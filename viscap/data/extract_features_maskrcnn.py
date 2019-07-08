import argparse
import torch
import cv2  # must import before importing caffe2 due to bug in cv2
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import sys
import numpy as np
import os
import warnings
# disable warning
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

sys.path.append("..")  # add parent to import modules
# path to packages inside captioning are already available to interpreter
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import (
    build_detection_model
)
from maskrcnn_benchmark.utils.model_serialization import (
    load_state_dict
)
from captioning.utils import process_feature_extraction, pad_raw_image_batch
from visdialch.data.dataset import RawImageDataset
from extract_utils import collate_function, rearrange_ouput, get_range_path

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

parser = argparse.ArgumentParser(
    description="Extract bottom-up features from a model trained by Detectron"
)
parser.add_argument(
    "--image-root",
    nargs="+",
    help="Path to a directory containing COCO/VisDial images. Note that this "
         "directory must have images, and not sub-directories of splits. "
         "Each HDF file should contain features from a single split."
         "Multiple paths are supported to account for VisDial v1.0 train.",
)
parser.add_argument(
    "--config",
    help="Path to model config file used by Detectron (.yaml)",
    default="extract_config.yaml",
)

parser.add_argument(
    "--save-path",
    help="Path to output file for saving bottom-up features (.h5)",
    default="data_img_mask_rcnn_x101.h5",
)
parser.add_argument(
    "--max-boxes",
    help="Maximum number of bounding box proposals per image",
    type=int,
    default=100
)
parser.add_argument(
    "--feat-name",
    help="The name of the layer to extract features from.",
    default="fc6",
)
parser.add_argument(
    "--feat-dims",
    help="Length of bottom-upfeature vectors.",
    type=int,
    default=2048,
)
parser.add_argument(
    "--split",
    choices=["train", "val", "test"],
    help="Which split is being processed.",
)
parser.add_argument(
    "--gpu-ids",
    help="The GPU id to use (-1 for CPU execution)",
    type=int,
    default=[0],
)
parser.add_argument(
    "--start-range",
    help="Truncate the dataset, from the beginning",
    type=int,
    default=None,
)
parser.add_argument(
    "--stop-range",
    help="Truncate the dataset, from the end",
    type=int,
    default=None,
)
parser.add_argument(
    "--batch-size",
    help="Number of images to be processed in one iteration",
    type=int,
    default=1,
)

# For reproducibility.
# Refer https://pytorch.org/docs/stable/notes/randomness.html
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


def main(args):
    """Extract bottom-up features from all images in a directory using
    a pre-trained Detectron model, and save them in HDF format.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """
    # load config
    # TODO: Also add batch_size and other extraction specific configs to file
    # TODO: Fix path resolution below
    visdial_path = os.getcwd() + "/../"
    config = yaml.load(open(args.config), Loader=yaml.FullLoader)
    caption_config = config["captioning"]
    cfg.merge_from_file(
        visdial_path + caption_config["detectron_model"]["config_yaml"]
    )
    cfg.freeze()

    # use this to extract features from different subsets of the datasets
    # by spawning multiple processes and finally merging the files,
    # this is recommended instead of using dataparallel model
    restrict_range = None
    if args.start_range is not None and args.stop_range is not None:
        restrict_range = [args.start_range, args.stop_range]
        args.save_path = get_range_path(args.save_path, restrict_range)

    if isinstance(args.gpu_ids, int):
        args.gpu_ids = [args.gpu_ids]

    device = (
        torch.device("cuda", args.gpu_ids[0])
        if args.gpu_ids[0] >= 0
        else torch.device("cpu")
    )

    print("Loading Model...")

    # TODO: pretty print config and usedse get_abspath, put config file in cwd
    # build mask-rcnn detection model
    detection_model = build_detection_model(cfg)
    detection_model.to(device)
    if -1 not in args.gpu_ids:
        detection_model = torch.nn.DataParallel(detection_model, args.gpu_ids)

    checkpoint = torch.load(
        visdial_path + caption_config["detectron_model"]["model_pth"],
        map_location=device)

    load_state_dict(detection_model, checkpoint.pop("model"))
    detection_model.eval()

    print("Model Loaded, Loading Dataset now...")

    raw_image_dataset = RawImageDataset(
        args.image_root,
        args.split,
        transform=True,
        in_mem=False,
        restrict_range=restrict_range
    )
    raw_image_dataloader = DataLoader(
        raw_image_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        shuffle=False,
        collate_fn=collate_function
    )
    print("Dataset Loaded")

    # create an output HDF to save extracted features
    save_h5 = h5py.File(args.save_path, "w")
    image_ids_h5d = save_h5.create_dataset(
        "image_ids", (len(raw_image_dataset),), dtype=int
    )

    boxes_h5d = save_h5.create_dataset(
        "boxes", (len(raw_image_dataset), args.max_boxes, 4),
    )
    features_h5d = save_h5.create_dataset(
        "features", (len(raw_image_dataset), args.max_boxes, args.feat_dims),
    )
    classes_h5d = save_h5.create_dataset(
        "classes", (len(raw_image_dataset), args.max_boxes,),
    )
    scores_h5d = save_h5.create_dataset(
        "scores", (len(raw_image_dataset), args.max_boxes,),
    )

    for i, batch in enumerate(tqdm(raw_image_dataloader)):

        # calculate idx_start and idx_end
        batch_size = args.batch_size
        idx_start, idx_end = i * batch_size, (i + 1) * batch_size

        # get the image_ids present in the batch
        image_ids = batch.pop("image_ids")

        for key in batch:
            if isinstance(batch[key], list):
                batch[key] = torch.Tensor(batch[key])
            batch[key] = batch[key].to(device)

        with torch.no_grad():
            output = detection_model(batch, mode="extract")

        output = rearrange_ouput(output)
        feat_name = caption_config["detectron_model"]["feat_name"]
        get_boxes = True

        boxes, features, classes, scores = process_feature_extraction(
            output,
            batch["im_scales"],
            args.max_boxes,
            get_boxes=get_boxes,
            feat_name=feat_name,
            conf_thresh=0.2
        )

        # store to extracted features to HDF file
        image_ids_h5d[idx_start:idx_end] = np.array(image_ids)
        boxes_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in boxes])
        features_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in features])
        classes_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in classes])
        scores_h5d[idx_start:idx_end] = np.array(
            [item.cpu().numpy() for item in classes])

    # set current split name in attributrs of file, for tractability
    save_h5.attrs["split"] = args.split
    save_h5.close()


if __name__ == "__main__":
    # set higher log level to prevent terminal spam
    args = parser.parse_args()
    main(args)
