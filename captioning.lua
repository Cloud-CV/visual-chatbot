require 'torch'
require 'nn'
require 'image'
utils = require 'misc.utils'

local preprocess = utils.preprocess

local TorchModel = torch.class('CaptioningTorchModel')

function TorchModel:__init(model_path, backend, input_sz, layer, seed, gpuid)
  self.model_path = model_path
  self.backend = backend
  self.input_sz = input_sz
  self.layer = layer
  self.seed = seed
  self.gpuid = gpuid

  if self.gpuid >= 0 then
    require 'cunn'
    require 'cudnn'
    require 'cutorch'
    cutorch.setDevice(1)
    cutorch.manualSeed(self.seed)
  end
  
  self:loadModel(model_path)
  torch.manualSeed(self.seed)
  torch.setdefaulttensortype('torch.FloatTensor')

end


function TorchModel:loadModel(model_path)

  -- Load the models
  local lm_misc_utils = require 'neuraltalk2.misc.utils'
  require 'neuraltalk2.misc.LanguageModel'
  local net_utils = require 'neuraltalk2.misc.net_utils'


  self.net = torch.load(model_path)
  print(self.net)
  local cnn_lm_model = self.net
  local cnn = cnn_lm_model.protos.cnn
  local lm = cnn_lm_model.protos.lm
  local vocab = cnn_lm_model.vocab


  net_utils.unsanitize_gradients(cnn)
  local lm_modules = lm:getModulesList()
  for k,v in pairs(lm_modules) do
    net_utils.unsanitize_gradients(v)
  end


  -- Set to evaluate mode
  lm:evaluate()
  cnn:evaluate()
  self.cnn = cnn
  self.lm = lm
  self.net_utils = net_utils
  self.vocab = vocab

end


function TorchModel:predict(input_image_path, input_sz, input_sz, out_path)
  print(input_image_path)
  local img = utils.preprocess(input_image_path, input_sz, input_sz)

  -- Clone & replace ReLUs for Guided Backprop
  local cnn_gb = self.cnn:clone()
  cnn_gb:replace(utils.guidedbackprop)

  -- Ship model to GPU
  if self.gpuid >= 0 then
    self.cnn:cuda()
    cnn_gb:cuda()
    img = img:cuda()
    self.lm:cuda()
  end

  -- Forward pass
  im_feats = self.cnn:forward(img)
  im_feat = im_feats:view(1, -1)
  im_feat_gb = cnn_gb:forward(img)

  -- get the prediction from model
  local seq, seqlogps = self.lm:sample(im_feat, sample_opts)
  seq[{{}, 1}] = seq

  local caption = self.net_utils.decode_sequence(self.vocab, seq)

  result = {}
  result['input_image'] = input_image_path
  result['pred_caption'] = caption[1]

  return result

end
