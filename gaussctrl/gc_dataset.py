# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
GaussCtrl dataset.
"""

import json
from pathlib import Path
from typing import Dict, Union, Literal

import numpy.typing as npt
import numpy as np
import torch, cv2
import skimage.io as io
from PIL import Image
from rich.progress import track

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.datasets.base_dataset import InputDataset
from nerfstudio.model_components import losses
from nerfstudio.utils.misc import torch_compile
from nerfstudio.utils.rich_utils import CONSOLE

def get_depth_z_0_image_from_path(
    filepath: Path,
    height: int,
    width: int,
    scale_factor: float,
    read_type: str = 'depth',
    interpolation: int = cv2.INTER_NEAREST,
) -> torch.Tensor:
    """Loads, rescales and resizes depth images.
    Filepath points to a 16-bit or 32-bit depth image, or a numpy array `*.npy`.

    Args:
        filepath: Path to depth image.
        height: Target depth image height.
        width: Target depth image width.
        scale_factor: Factor by which to scale depth image.
        interpolation: Depth value interpolation for resizing.

    Returns:
        Depth image torch tensor with shape [height, width, 1].
    """
    if read_type == "depth":
        image = np.load(filepath)[:,:,0] * scale_factor
        # image = cv2.resize(image, (width, height), interpolation=interpolation)
        return image[None]
    elif read_type == "z_0":
        image = np.load(filepath) * scale_factor
        return image
    elif read_type == "mask":
        image = np.load(filepath) * scale_factor
        return image
    else:
        raise TypeError("Wrong depth type, depth files should be .npy files")
    # return torch.from_numpy(image[:, :, np.newaxis])

class GCDataset(InputDataset):
    """Dataset that returns images and depths. If no depths are found, then we generate them with Zoe Depth.

    Args:
        dataparser_outputs: description of where and how to read input images.
        scale_factor: The scaling factor for the dataparser outputs.
    """

    def __init__(self, dataparser_outputs: DataparserOutputs, scale_factor: float = 1.0):
        super().__init__(dataparser_outputs, scale_factor)
        if 'depth_filenames' in self.metadata.keys():
            self.depth_filenames = self.metadata["depth_filenames"]
        if 'z_0_filenames' in self.metadata.keys():
            self.z_0_filenames = self.metadata["z_0_filenames"]
        if 'unedited_image_filenames' in self.metadata.keys():
            self.unedited_image_filenames = self.metadata["unedited_image_filenames"]
        if 'mask_filenames' in self.metadata.keys():
            self.mask_filenames = self.metadata["mask_filenames"]

    def get_data(self, image_idx: int, image_type: Literal["uint8", "float32"] = "float32") -> Dict:
        """Returns the ImageDataset data as a dictionary.

        Args:
            image_idx: The image index in the dataset.
            image_type: the type of images returned
        """
        if image_type == "float32":
            image = self.get_image_float32(image_idx) # [512 512 3] 0-1
        elif image_type == "uint8":
            image = self.get_image_uint8(image_idx)
        else:
            raise NotImplementedError(f"image_type (={image_type}) getter was not implemented, use uint8 or float32")
        
        data = {"image_idx": image_idx, "image": image}
        metadata = self.get_metadata(data)
        data.update(metadata)
        return data

    def get_unedited_numpy_image(self, image_idx: int) -> npt.NDArray[np.uint8]:
        """Returns the image of shape (H, W, 3 or 4).

        Args:
            image_idx: The image index in the dataset.
        """
        image_filename = self._dataparser_outputs.metadata['unedited_image_filenames'][image_idx]
        pil_image = Image.open(image_filename)
        if self.scale_factor != 1.0:
            width, height = pil_image.size
            newsize = (int(width * self.scale_factor), int(height * self.scale_factor))
            pil_image = pil_image.resize(newsize, resample=Image.BILINEAR)
        image = np.array(pil_image, dtype="uint8")  # shape is (h, w) or (h, w, 3 or 4)
        if len(image.shape) == 2:
            image = image[:, :, None].repeat(3, axis=2)
        assert len(image.shape) == 3
        assert image.dtype == np.uint8
        assert image.shape[2] in [3, 4], f"Image shape of {image.shape} is in correct."
        return image

    def get_metadata(self, data: Dict) -> Dict:
        meta = {}
        height = int(self._dataparser_outputs.cameras.height[data["image_idx"]])
        width = int(self._dataparser_outputs.cameras.width[data["image_idx"]])
        
        if 'depth_filenames' in self.metadata.keys():
            depth_filepath = self.depth_filenames[data["image_idx"]]
            # Scale depth images to meter units and also by scaling applied to cameras
            depth_image = get_depth_z_0_image_from_path(
                filepath=depth_filepath, height=height, width=width, scale_factor=1, read_type='depth'
            )
            meta["depth_image"] = depth_image
        if 'z_0_filenames' in self.metadata.keys():
            z_0_filepath = self.z_0_filenames[data["image_idx"]]
            z_0_image = get_depth_z_0_image_from_path(
                filepath=z_0_filepath, height=height, width=width, scale_factor=1, read_type='z_0'
            )
            meta["z_0_image"] = z_0_image
        if 'mask_filenames' in self.metadata.keys():
            mask_filepath = self.mask_filenames[data["image_idx"]]
            mask_image = get_depth_z_0_image_from_path(
                filepath=mask_filepath, height=height, width=width, scale_factor=1, read_type='mask'
            ) # boolean
            meta["mask_image"] = mask_image
        if 'unedited_image_filenames' in self.metadata.keys():
            unedited_image = self.get_unedited_numpy_image(data["image_idx"])
            unedited_image = torch.from_numpy(unedited_image.astype("float32") / 255.0) 
            meta["unedited_image"] = unedited_image 
            
        return meta

    def _find_transform(self, image_path: Path) -> Union[Path, None]:
        while image_path.parent != image_path:
            transform_path = image_path.parent / "transforms.json"
            if transform_path.exists():
                return transform_path
            image_path = image_path.parent
        return None
