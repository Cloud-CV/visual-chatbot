from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
from torch.nn.functional import normalize
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from .readers import (
    DialogsReader,
    DenseAnnotationsReader,
    ImageFeaturesHdfReader,
    RawImageReader
)
from .vocabulary import Vocabulary


class RawImageDataset(Dataset):
    """
    Represents dataset built from a list folders containing raw images. This
    is used inside feature extraction script to load images and extract
    features.

    Attributes
    ----------
    image_dirs : ``List[str]``
        List of paths to read images from.
    split : ``str``
        Type of split of the dataset associated with the images.
    transform : ``bool``
        Apply `self.image_transform` to images loaded.
    in_mem : ``bool``
        Load all the images in memory during initialization else go lazy.
    restrict_range: ``List[int]``
        List of two integers with the start and end index of truncated dataset.

    """

    def __init__(self,
                 image_dirs: List[str],
                 split: str,
                 transform: bool,
                 in_mem: bool,
                 restrict_range: List[int]=None) -> None:
        super().__init__()
        self.in_mem = in_mem
        self.raw_image_reader = RawImageReader(image_dirs, split, in_mem)
        self.image_ids = self.raw_image_reader.image_ids
        self.transform = transform
        if restrict_range is not None:
            assert(len(restrict_range) == 2, "Range should contain only two ints")
            start_idx, end_idx = restrict_range[0], restrict_range[1]
            self.image_ids = self.image_ids[start_idx:end_idx]
            print(f"Truncated Dataset with start_idx: {start_idx} and end_idx: {end_idx}")

    def __getitem__(self, index):
        item = {}
        image_id = self.image_ids[index]
        # apply transforms here and pack image in a dict
        image = self.raw_image_reader[image_id]
        item["image"] = image
        if self.transform:
            item["image"], item["im_scale"] = self.image_transform(
                image,
                image_id
            )
        item["image_id"] = image_id
        return item

    def __len__(self):
        return len(self.image_ids)

    @property
    def split(self):
        return self.raw_image_reader.split

    def image_transform(self, image):
        r""" Apply image transformation. This is used internally by the
        ``self.__getitem__`` method.

        Parameters
        ----------
        image : ``np.array``
            Image passed as numpy array.

        Returns
        -------
        [torch.Tensor, float]
            Tuple of image and rescaling factor.

        """

        im = np.array(image).astype(np.float32)
        
        # handle b/w and four channeled images, also log them
        if len(im.shape) == 2:
            im = np.stack([im,im,im], axis=2)
            print(f"Found a grayscale image: image-id = {image_id}")
        elif len(im.shape) == 3 and im.shape[2] == 4:
            im = im[:,:,:-1]
            print(f"Found a four channeled image: image-id = {image_id}")

        im = im[:, :, ::-1]
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
