require 'nn'
require 'image'
require 'rnn'
require 'loadcaffe'
require 'nngraph'
utils = dofile('utils.lua');

local TorchModel = torch.class('VisDialTorchModel')

function TorchModel:__init(inputJson, loadPath, beamSize, beamLen, sampleWords, temperature, gpuid, backend, proto_file, model_file, maxThreads, encoder, decoder)
  self.input_json = inputJson
  self.loadPath = loadPath
  self.beamSize = beamSize
  self.beamLen = beamLen
  self.sampleWords = sampleWords
  self.temperature = temperature
  self.gpuid = gpuid
  self.backend = backend
  self.proto_file = proto_file
  self.model_file = model_file
  self.maxThreads = maxThreads

  ------------------------------------------------------------------------
  -- seed for reproducibility
  ------------------------------------------------------------------------
  torch.manualSeed(1234);

  ------------------------------------------------------------------------
  -- set default tensor based on gpu usage
  ------------------------------------------------------------------------
  if self.gpuid >= 0 then
      require 'cutorch'
      require 'cunn'
      if self.backend == 'cudnn' then require 'cudnn' end
      cutorch.setDevice(self.gpuid+1)
      cutorch.manualSeed(1234)
  end

  ------------------------------------------------------------------------
  -- Set default tensor type to FloatTensor
  ------------------------------------------------------------------------
  torch.setdefaulttensortype('torch.FloatTensor');

  ------------------------------------------------------------------------
  -- load CNN
  ------------------------------------------------------------------------
  cnn = loadcaffe.load(self.proto_file, self.model_file, self.backend)
  cnn:evaluate()
  cnn:remove()
  cnn:remove()
  cnn:add(nn.Normalize(2))

  if self.gpuid >= 0 then
    cnn = cnn:cuda()
  end

  ------------------------------------------------------------------------
  -- Read saved model and parameters
  ------------------------------------------------------------------------
  local savedModel = torch.load(self.loadPath)

  ------------------------------------------------------------------------
  -- transfer all options to model
  ------------------------------------------------------------------------
  local modelParams = savedModel.modelParams
  self.imgNorm = modelParams.imgNorm
  self.encoder = modelParams.encoder
  self.decoder = modelParams.decoder

  modelParams.gpuid = self.gpuid

  ------------------------------------------------------------------------
  -- add flags for various configurations
  -- additionally check if its imitation of discriminative model
  ------------------------------------------------------------------------
  if string.match(self.encoder, 'hist') then 
      self.useHistory = true;
  end
  if string.match(self.encoder, 'im') then self.useIm = true; end

  ------------------------------------------------------------------------
  -- Loading dataset
  ------------------------------------------------------------------------
  local dataloader = dofile('dataloader.lua')
  dataloader:initialize(self);
  collectgarbage();

  ------------------------------------------------------------------------
  -- Setup the model
  ------------------------------------------------------------------------
  require 'model'
  local model = Model(modelParams)

  ------------------------------------------------------------------------
  -- copy the weights from loaded model
  ------------------------------------------------------------------------
  model.wrapperW:copy(savedModel.modelW);

  ------------------------------------------------------------------------
  -- Create the Class attributes to access them while doing prediction
  ------------------------------------------------------------------------
  self.dataloader = dataloader
  self.cnn = cnn
  self.model = model

end


function TorchModel:predict(img, history, question)
  img_path = img -- storing the image path to send back to client side

  local startToken = self.dataloader.word2ind['<START>'];
  local endToken = self.dataloader.word2ind['<END>'];

  history_concat = ''
  for i = 1, #history do
      history_concat = history_concat .. history[i] .. '||||'
  end

  local cmd = 'python prepro_ques.py -question "' .. question .. '" -history "' .. history_concat .. '"'
  os.execute(cmd)
  file = io.open('ques_feat.json')
  text = file:read()
  file:close()
  feats = cjson.decode(text)

  ques_vector = utils.wordsToId(feats.question, self.dataloader.word2ind)
  
  hist_tensor = torch.LongTensor(10, 30):zero()
  for i = 1, #feats.history do
      hist_tensor[i] = utils.wordsToId(feats.history[i], self.dataloader.word2ind, 30)
  end

  ques_tensor = torch.LongTensor(10, 15):zero()
  for i = 1, #feats.questions do
      ques_tensor[i] = utils.wordsToId(feats.questions[i], self.dataloader.word2ind, 15)
  end
  ques_tensor[#feats.questions+1] = ques_vector

  local iter = #feats.questions+1

  img = utils.preprocess(img, 224, 224)

  if self.gpuid >= 0 then
      img = img:cuda()
      ques_vector = ques_vector:cuda()
      hist_tensor = hist_tensor:cuda()
  end

  img_feats = self.cnn:forward(img)

  sampleParams = {
      beamSize = self.beamSize,
      beamLen = self.beamLen,
      maxThreads = self.maxThreads,
      sampleWords = self.sampleWords,
      temperature = self.temperature
  }

  local batch = {ques_fwd = ques_tensor, img_feat = img_feats, hist = hist_tensor}
  local output = self.model:generateAnswers(self.dataloader, sampleParams, batch, iter)

  question = output[1]
  ansText = output[2]

  result = {}
  result['answer'] = ansText
  result['question'] = question
  if history_concat == "||||" then
    history_concat = ""
  end
  result['history'] = history_concat .. question .. " " .. ansText
  result['input_image'] = img_path
  return result

end
