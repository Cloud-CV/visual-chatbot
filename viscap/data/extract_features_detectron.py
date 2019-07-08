import argparse
import glob
import os

import cv2  # must import before importing caffe2 due to bug in cv2
from caffe2.python import workspace
from tqdm import tqdm
import h5py
import numpy as np

from detectron.core.config import assert_and_infer_cfg, merge_cfg_from_file
from detectron.core.config import cfg as detectron_config
from detectron.utils.boxes import nms as detectron_nms
import detectron.core.test as detectron_test
import detectron.core.test_engine as infer_engine
import detectron.utils.c2 as c2_utils


c2_utils.import_detectron_ops()
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
    default="data/config_faster_rcnn_x101.yaml",
)
parser.add_argument(
    "--weights",
    help="Path to model weights file saved by Detectron (.pkl)",
    default="data/model_faster_rcnn_x101.pkl",
)
parser.add_argument(
    "--save-path",
    help="Path to output file for saving bottom-up features (.h5)",
    default="data/data_img_faster_rcnn_x101.h5",
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
    default="fc7",
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
    "--gpu-id",
    help="The GPU id to use (-1 for CPU execution)",
    type=int,
    default=0,
)


def detect_image(detectron_model, image, args):
    """Given an image and a detectron model, extract object boxes,
    classes, confidences and features from the image using the model.

    Parameters
    ----------
    detectron_model
        Detectron model.
    image : np.ndarray
        Image in BGR format.
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
        Object bounding boxes, classes, confidence and features.
    """

    scores, cls_boxes, im_scale = detectron_test.im_detect_bbox(
        detectron_model,
        image,
        detectron_config.TEST.SCALE,
        detectron_config.TEST.MAX_SIZE,
        boxes=None,
    )
    num_proposals = scores.shape[0]

    rois = workspace.FetchBlob(f"gpu_{args.gpu_id}/rois")
    features = workspace.FetchBlob(
        f"gpu_{args.gpu_id}/{args.feat_name}"
    )

    cls_boxes = rois[:, 1:5] / im_scale
    max_conf = np.zeros((num_proposals,), dtype=np.float32)
    max_cls = np.zeros((num_proposals,), dtype=np.int32)
    max_box = np.zeros((num_proposals, 4), dtype=np.float32)

    for cls_ind in range(1, detectron_config.MODEL.NUM_CLASSES):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(
            np.float32
        )
        keep = np.array(detectron_nms(dets, detectron_config.TEST.NMS))
        idxs_update = np.where(cls_scores[keep] > max_conf[keep])
        keep_idxs = keep[idxs_update]
        max_conf[keep_idxs] = cls_scores[keep_idxs]
        max_cls[keep_idxs] = cls_ind
        max_box[keep_idxs] = dets[keep_idxs][:, :4]

    keep_boxes = np.argsort(max_conf)[::-1][:args.max_boxes]
    boxes = max_box[keep_boxes, :]
    classes = max_cls[keep_boxes]
    confidence = max_conf[keep_boxes]
    features = features[keep_boxes, :]
    return boxes, features, classes, confidence


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


def main(args):
    """Extract bottom-up features from all images in a directory using
    a pre-trained Detectron model, and save them in HDF format.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.
    """

    # specifically for visual genome
    detectron_config.MODEL.NUM_ATTRIBUTES = -1
    merge_cfg_from_file(args.config)

    # override some config options and validate the config
    detectron_config.NUM_GPUS = 1
    detectron_config.TRAIN.CPP_RPN = "none"
    assert_and_infer_cfg(cache_urls=False)

    # initialize model
    detectron_model = infer_engine.initialize_model_from_cfg(
        args.weights, args.gpu_id
    )

    # list of paths (example: "coco_train2014/COCO_train2014_000000123456.jpg")
    image_paths = []
    for image_root in args.image_root:
        image_paths.extend(
            [
                os.path.join(image_root, name)
                for name in glob.glob(os.path.join(image_root, "*.jpg"))
                if name not in {".", ".."}
            ]
        )

    # create an output HDF to save extracted features
    save_h5 = h5py.File(args.save_path, "w")
    image_ids_h5d = save_h5.create_dataset(
        "image_ids", (len(image_paths),), dtype=int
    )

    boxes_h5d = save_h5.create_dataset(
        "boxes", (len(image_paths), args.max_boxes, 4),
    )
    features_h5d = save_h5.create_dataset(
        "features", (len(image_paths), args.max_boxes, args.feat_dims),
    )
    classes_h5d = save_h5.create_dataset(
        "classes", (len(image_paths), args.max_boxes, ),
    )
    scores_h5d = save_h5.create_dataset(
        "scores", (len(image_paths), args.max_boxes, ),
    )

    with c2_utils.NamedCudaScope(args.gpu_id):
        for idx, image_path in enumerate(tqdm(image_paths)):
            try:
                image_ids_h5d[idx] = image_id_from_path(image_path)

                image = cv2.imread(image_path)
                boxes, features, classes, scores = detect_image(detectron_model, image, args)

                boxes_h5d[idx] = boxes
                features_h5d[idx] = features
                classes_h5d[idx] = classes
                scores_h5d[idx] = scores
            except:
                print(f"\nWarning: Failed to extract features from {idx}, {image_path}.\n")

    # set current split name in attributrs of file, for tractability
    save_h5.attrs["split"] = args.split
    save_h5.close()


if __name__ == "__main__":
    # set higher log level to prevent terminal spam
    workspace.GlobalInit(["caffe2", "--caffe2_log_level=3"])
    args = parser.parse_args()
    main(args)
