from django.conf import settings
import os

BOT_INTORDUCTION_MESSAGE = [
    "Hi, I am a Visual Chatbot, capable of answering a sequence of questions about images. Please upload an image and fire away!",
]

# VISDIAL_CONFIG = {
#     'input_json': 'data/chat_processed_params.json',
#     # 'load_path': 'models/mn-qih-g-102.t7',
#     'load_path': 'models/hre-qih-g-10.t7',
#     'result_path': 'results',
#     'gpuid': 0,
#     'backend': 'cudnn',
#     'proto_file': 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
#     'model_file': 'models/VGG_ILSVRC_16_layers.caffemodel',
#     # 'encoder': 'hre-ques-im-hist',
#     # 'decoder': 'disc',
# }

VISDIAL_CONFIG = {
    'input_json': 'data/chat_processed_params_0.9.json',
    'load_path': 'models/hre-qih-g-new.t7',
    'result_path': 'results',
    'gpuid': 0,
    'backend': 'cudnn',
    'proto_file': 'models/VGG_ILSVRC_16_layers_deploy.prototxt',
    'model_file': 'models/VGG_ILSVRC_16_layers.caffemodel',
    'beamSize': 5,
    'beamLen': 20,
    'sampleWords': 0,
    'temperature': 1.0,
    'maxThreads': 500,
    'encoder': 'hre-ques-im-hist',
    'decoder': 'disc'
}

VISDIAL_LUA_PATH = "evaluate.lua"

CAPTIONING_GPUID = 2

CAPTIONING_CONFIG = {
    'input_sz': 224,
    'backend': 'cudnn',
    'layer': 30,
    'model_path': 'neuraltalk2/model_id1-501-1448236541.t7',
    'seed': 123,
    'image_dir': os.path.join(settings.BASE_DIR, 'media', 'captioning')
}

CAPTIONING_LUA_PATH = "captioning.lua"

if CAPTIONING_GPUID == -1:
    CAPTIONING_CONFIG['backend'] = "nn"
    CAPTIONING_CONFIG['model_path'] = "neuraltalk2/model_id1-501-1448236541.t7_cpu.t7"
else:
    CAPTIONING_CONFIG['backend'] = "cudnn"
