import torch
import yaml
from pythia.common.registry import registry
from pythia.models.butd import BUTD
from pythia.tasks.processors import VocabProcessor, CaptionProcessor
from pythia.utils.configuration import ConfigNode


# TODO: Docstrings and hints
# Build Captioning Model

def build_caption_model(caption_config, cuda_device):
    with open(caption_config["butd_model"]["config_yaml"]) as f:
        config = yaml.load(f)
    config = ConfigNode(config)
    config.training_parameters.evalai_inference = True
    registry.register("config", config)

    caption_processor, text_processor = init_processors(caption_config, config)
    state_dict = torch.load(caption_config["butd_model"]["model_pth"])
    model_config = config.model_attributes.butd
    model_config.model_data_dir = caption_config["model_data_dir"]
    model = BUTD(model_config)
    model.build()
    model.init_losses_and_metrics()

    if list(state_dict.keys())[0].startswith('module') and \
            not hasattr(model, 'module'):
        state_dict = multi_gpu_state_to_single(state_dict)

    model.load_state_dict(state_dict)
    model.to(cuda_device)
    model.eval()

    return model, caption_processor, text_processor


def init_processors(caption_config, config):
    captioning_config = config.task_attributes.captioning \
        .dataset_attributes.coco
    text_processor_config = captioning_config.processors.text_processor
    caption_processor_config = captioning_config.processors \
        .caption_processor
    vocab_file_path = caption_config[
        "text_caption_processor_vocab_txt"]
    text_processor_config.params.vocab.vocab_file = vocab_file_path
    caption_processor_config.params.vocab.vocab_file = vocab_file_path
    text_processor = VocabProcessor(text_processor_config.params)
    caption_processor = CaptionProcessor(
        caption_processor_config.params)

    registry.register("coco_text_processor", text_processor)
    registry.register("coco_caption_processor", caption_processor)

    return caption_processor, text_processor


def multi_gpu_state_to_single(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if not k.startswith('module.'):
            raise TypeError("Not a multiple GPU state of dict")
        k1 = k[7:]
        new_sd[k1] = v
    return new_sd


