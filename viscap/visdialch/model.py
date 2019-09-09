import torch
from torch import nn

from viscap.visdialch.decoders import Decoder
from viscap.visdialch.encoders import Encoder
from viscap.visdialch.utils.checkpointing import load_checkpoint

# TODO: Modify docstring and hints
class EncoderDecoderModel(nn.Module):
    """Convenience wrapper module, wrapping Encoder and Decoder modules.

    Parameters
    ----------
    encoder: nn.Module
    decoder: nn.Module
    """

    def __init__(self,
                 model_config=None,
                 vocabulary=None,
                 encoder=None,
                 decoder=None
                 ):
        super().__init__()

        if model_config is not None and vocabulary is not None:
            self.encoder = Encoder(model_config, vocabulary)
            self.decoder = Decoder(model_config, vocabulary)
        elif encoder is not None and decoder is not None:
            self.encoder = encoder
            self.decoder = decoder
        else:
            raise ValueError("No constructor found for passed arguments")

        print("Encoder: {}".format(model_config["encoder"]))
        print("Decoder: {}".format(model_config["decoder"]))
        # Share word embedding between encoder and decoder.
        self.decoder.word_embed = self.encoder.word_embed

    def forward(self, batch):
        encoder_output = self.encoder(batch)
        decoder_output = self.decoder(encoder_output, batch)
        return decoder_output

    def load_checkpoint(self, load_pthpath, device):
        model_state_dict, _ = load_checkpoint(load_pthpath, device)
        if isinstance(self, nn.DataParallel):
            self.module.load_state_dict(model_state_dict)
        else:
            self.load_state_dict(model_state_dict)
        print("Loaded model from {}".format(load_pthpath))

