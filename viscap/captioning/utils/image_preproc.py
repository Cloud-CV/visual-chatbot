import os
import cv2
import numpy as np
import torch
import requests
import math
from PIL import Image
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.structures.image_list import to_image_list

# TODO: Comments, Docstrings, Cleanup.

def read_image(image_path):
    if image_path.startswith('http'):
        path = requests.get(image_path, stream=True).raw
    else:
        path = image_path
    img = Image.open(path)
    img.load() # close the loaded image
    return img


def image_transform(img):
    im = np.array(img).astype(np.float32)

    # Handle B&W images
    if len(im.shape) == 2:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    
    # Handle images with alpha channel
    if im.shape[2] == 4:
        im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)

    im = im[:, :, ::-1]
    
    # Transform used for BUTD model in pythia trained on coco
    im -= np.array([102.9801, 115.9465, 122.7717])
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(800) / float(im_size_min)
    # Prevent the biggest axis from being more than max_size
    if np.round(im_scale * im_size_max) > 1333:
        im_scale = float(1333) / float(im_size_max)
    im = cv2.resize(
        im,
        None,
        None,
        fx=im_scale,
        fy=im_scale,
        interpolation=cv2.INTER_LINEAR
    )
    img = torch.from_numpy(im).permute(2, 0, 1)
    return img, im_scale


def process_feature_extraction(output,
                               im_scales,
                               max_boxes=100,
                               get_boxes=False,
                               feat_name='fc6',
                               conf_thresh=0.2):
    # TODO: Add docstring and explain get_boxes
    batch_size = len(output[0]["proposals"])
    n_boxes_per_image = [len(_) for _ in output[0]["proposals"]]
    score_list = output[0]["scores"].split(n_boxes_per_image)
    score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
    feats = output[0][feat_name].split(n_boxes_per_image)
    cur_device = score_list[0].device

    feat_list = []
    boxes_list = []
    classes_list = []
    conf_list = []

    for i in range(batch_size):
        # bbox below stays on the device where it was generated
        dets = output[0]["proposals"][i].bbox.to(cur_device) / im_scales[i]
        scores = score_list[i]

        max_conf = torch.zeros((scores.shape[0])).to(cur_device)

        if get_boxes:
            max_cls = torch.zeros((scores.shape[0]), dtype=torch.long).to(
                cur_device)
            max_box = torch.zeros((scores.shape[0], 4)).to(cur_device)

        for cls_ind in range(1, scores.shape[1]):
            cls_scores = scores[:, cls_ind]
            keep = nms(dets, cls_scores, 0.5)

            if get_boxes:
                max_cls[keep] = torch.where(
                    cls_scores[keep] > max_conf[keep],
                    torch.tensor(cls_ind).to(cur_device),
                    max_cls[keep]
                )

                max_box[keep] = torch.where(
                    (cls_scores[keep] > max_conf[keep]).view(-1, 1),
                    dets[keep],
                    max_box[keep]
                )

            max_conf[keep] = torch.where(
                cls_scores[keep] > max_conf[keep],
                cls_scores[keep],
                max_conf[keep]
            )

        keep_boxes = torch.argsort(max_conf, descending=True)[:max_boxes]
        feat_list.append(feats[i][keep_boxes])

        if not get_boxes:
            return feat_list

        conf_list.append(max_conf[keep_boxes])
        boxes_list.append(max_box[keep_boxes])
        classes_list.append(max_cls[keep_boxes])

    return [boxes_list, feat_list, classes_list, conf_list]


# Given a list of image(s) returns features, bboxes, scores and classes
def get_detectron_features(image_paths,
                           detection_model,
                           get_boxes,
                           feat_name,
                           device):
    img_tensor, im_scales = [], []

    for img_path in image_paths:
        im = read_image(img_path)
        im, im_scale = image_transform(im)
        img_tensor.append(im)
        im_scales.append(im_scale)

    current_img_list = to_image_list(img_tensor, size_divisible=32)
    current_img_list = current_img_list.to(device)
    with torch.no_grad():
        output = detection_model(current_img_list)

    feat_list = process_feature_extraction(
        output,
        im_scales,
        get_boxes=get_boxes,
        feat_name=feat_name,
        conf_thresh=0.2
    )
    return feat_list


def get_abspath(path):
    if not os.path.isabs(path):
        return os.path.abspath(path)
    else:
        return path

def pad_raw_image_batch(images: torch.Tensor, size_divisible: int = 0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    if size_divisible > 0:
        stride = size_divisible
        max_size = list(max_size)
        max_size[1] = int(math.ceil(max_size[1] / stride) * stride)
        max_size[2] = int(math.ceil(max_size[2] / stride) * stride)
        max_size = tuple(max_size)

    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).zero_()
    for img, pad_img in zip(images, batched_imgs):
        pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)

    image_sizes = [im.shape[-2:] for im in batched_imgs]

    return batched_imgs, image_sizes
