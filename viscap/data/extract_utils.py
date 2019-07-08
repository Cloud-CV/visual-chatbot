import torch
import sys

sys.path.append("..")  # add parent to import modules
from captioning.utils import pad_raw_image_batch


def image_id_from_path(image_path):
    """Given a path to an image, return its id.

    Parameters
    ----------
    image_path : str
        Path to image, e.g.: coco_train2014/COCO_train2014/000000123456.jpg

    Returns
    -------
    int
        Corresponding image id (123456)
    """

    return int(image_path.split("/")[-1][-16:-4])

def get_range_path(save_path, restrict_range):
    start_range, stop_range = restrict_range[0], restrict_range[1]
    save_path = save_path.split('.')[0] + "_range_" + str(
        start_range) + "_to_" + str(stop_range) + '.h5'
    return save_path


# used to handle variable size images inside the dataloader
def collate_function(batch):
    # print(f"Debug: collate_function, target shape {len(batch)}, {type(batch[0])}")
    # I think batch is (n, item)
    item_batch = {}
    item_batch["image"] = []
    item_batch["im_scales"] = []
    item_batch["image_ids"] = []
    for item in batch:
        item_batch["image"].append(item["image"])
        item_batch["im_scales"].append(item["im_scale"])
        item_batch["image_ids"].append(item["image_id"])

    item_batch["image"], item_batch["image_size"] = pad_raw_image_batch(
        item_batch["image"],
        32
    )
    return item_batch


# rearrange output from the model from process_feat_extraction function
def rearrange_ouput(output):
    for key in output[0].keys():
        if isinstance(output[0][key], torch.Tensor):
            embed_dim = output[0][key].shape[-1]
            output[0][key] = output[0][key].view(-1, embed_dim)
    return output
