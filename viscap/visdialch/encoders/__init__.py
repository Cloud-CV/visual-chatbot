from visdialch.encoders.lf import LateFusionEncoder


def Encoder(model_config, *args):
    name_enc_map = {"lf": LateFusionEncoder}
    return name_enc_map[model_config["encoder"]](model_config, *args)
