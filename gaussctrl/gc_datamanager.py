# Copyright 2022 The Nerfstudio Team. All rights reserved.
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
GaussCtrl Datamanager.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, Type, Union, Literal, Generic, TypeVar, get_args, get_origin, cast, ForwardRef, List
import random, torch
from rich.progress import Console
from copy import deepcopy
from functools import cached_property
from tqdm import tqdm
import numpy as np
import random

from diffusers.models.attention_processor import AttnProcessor
from nerfstudio.cameras.rays import RayBundle
from nerfstudio.data.utils.dataloaders import CacheDataloader
from nerfstudio.utils.misc import get_orig_class
from nerfstudio.data.datamanagers.base_datamanager import (
    VanillaDataManager,
    VanillaDataManagerConfig,
)
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
    _undistort_image
)
from nerfstudio.data.datamanagers.base_datamanager import DataManager, DataManagerConfig, TDataset
from gaussctrl.gc_dataset import GCDataset
from nerfstudio.data.datasets.base_dataset import InputDataset
from gaussctrl import utils



CONSOLE = Console(width=120)

@dataclass
class GaussCtrlDataManagerConfig(FullImageDatamanagerConfig):
    """Configuration for the GaussCtrlDataManager."""

    _target: Type = field(default_factory=lambda: GaussCtrlDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""
    subset_num: int = 4
    """Subset sample split number: We sample 40 (subset_num*sampled_views_every_subset) views here randomly from the entire dataset to accelerate the method. We split the entire set into 4 subsets, and randomly sampled 10 images in each subset split. Feel free to reduce the number to make the method faster.""" 
    sampled_views_every_subset: int = 10
    """Number of images sampled in each subset split"""
    load_all: bool = False
    """Set it to True if you want to edit all the images in the dataset"""


class GaussCtrlDataManager(FullImageDatamanager, Generic[TDataset]):
    """Data manager for GaussCtrl."""

    config: GaussCtrlDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(self,
                config: GaussCtrlDataManagerConfig,
                device: Union[torch.device, str] = "cpu",
                test_mode: Literal["test", "val", "inference"] = "val",
                world_size: int = 1,
                local_rank: int = 0,
                **kwargs,):
        super().__init__(config, device, test_mode, world_size, local_rank)

        self.sample_idx = []
        self.step_every = 1
        self.edited_image_dict = {}
        
        # Sample data
        if len(self.train_dataset._dataparser_outputs.image_filenames) <= self.config.subset_num * self.config.sampled_views_every_subset or self.config.load_all:
            self.cameras = self.train_dataset.cameras
            self.train_data = self.cached_train 
            self.train_unseen_cameras = list(range(len(self.train_data)))
        else:
            view_num = len(self.train_dataset._dataparser_outputs.image_filenames)
            anchors = list(range(0, view_num, view_num // self.config.subset_num))[:4] + [view_num]
            sampled_indices = []
            for idx in range(len(anchors[:-1])):
                cur_anchor = anchors[idx]
                next_anchor = anchors[idx+1] 
                selected = sorted(random.sample(list(range(cur_anchor, next_anchor)), self.config.sampled_views_every_subset))
                sampled_indices += selected

            self.cameras = [self.train_dataset.cameras[i: i+1] for i in sampled_indices]
            self.train_data_temp = [self.cached_train[i] for i in sampled_indices]
            self.train_data = []
            for i, data in enumerate(self.train_data_temp):
                data['image_idx'] = i
                self.train_data.append(data)
            self.train_unseen_cameras = list(range(self.config.subset_num * self.config.sampled_views_every_subset))
        
    def cache_images(self, cache_images_option):
        cached_train = []
        cached_eval = []
        CONSOLE.log("Caching / undistorting train images")
        for i in tqdm(range(len(self.train_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.train_dataset.get_data(i, image_type=self.config.cache_images_type)
            camera = self.train_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                cached_train.append(data)
                continue
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            cached_train.append(data)

            self.train_dataset.cameras.fx[i] = float(K[0, 0])
            self.train_dataset.cameras.fy[i] = float(K[1, 1])
            self.train_dataset.cameras.cx[i] = float(K[0, 2])
            self.train_dataset.cameras.cy[i] = float(K[1, 2])
            self.train_dataset.cameras.width[i] = image.shape[1]
            self.train_dataset.cameras.height[i] = image.shape[0]

        CONSOLE.log("Caching / undistorting eval images")
        for i in tqdm(range(len(self.eval_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.eval_dataset.get_data(i, image_type=self.config.cache_images_type)
            camera = self.eval_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                cached_eval.append(data)
                continue
            distortion_params = camera.distortion_params.numpy()
            image = data["image"].numpy()

            K, image, mask = _undistort_image(camera, distortion_params, data, image, K)
            data["image"] = torch.from_numpy(image)
            if mask is not None:
                data["mask"] = mask

            cached_eval.append(data)

            self.eval_dataset.cameras.fx[i] = float(K[0, 0])
            self.eval_dataset.cameras.fy[i] = float(K[1, 1])
            self.eval_dataset.cameras.cx[i] = float(K[0, 2])
            self.eval_dataset.cameras.cy[i] = float(K[1, 2])
            self.eval_dataset.cameras.width[i] = image.shape[1]
            self.eval_dataset.cameras.height[i] = image.shape[0]

        if cache_images_option == "gpu":
            for cache in cached_train:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
            for cache in cached_eval:
                cache["image"] = cache["image"].to(self.device)
                if "mask" in cache:
                    cache["mask"] = cache["mask"].to(self.device)
        else:
            for cache in cached_train:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()
            for cache in cached_eval:
                cache["image"] = cache["image"].pin_memory()
                if "mask" in cache:
                    cache["mask"] = cache["mask"].pin_memory()

        return cached_train, cached_eval

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[GaussCtrlDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is GaussCtrlDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is GaussCtrlDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is GaussCtrlDataManager:
                for value in get_args(base):
                    if isinstance(value, ForwardRef):
                        if value.__forward_evaluated__:
                            value = value.__forward_value__
                        elif value.__forward_module__ is None:
                            value.__forward_module__ = type(self).__module__
                            value = getattr(value, "_evaluate")(None, None, set())
                    assert isinstance(value, type)
                    if issubclass(value, InputDataset):
                        return cast(Type[TDataset], value)
        return default

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Returns the next training batch

        Returns a Camera instead of raybundle"""

        image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1)) 
        
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_data))]
        
        data = deepcopy(self.train_data[image_idx]) # image torch.Size([512, 512, 3]) cpu 0-1
        data["image"] = data["image"].to(self.device)
        
        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        if len(self.train_dataset._dataparser_outputs.image_filenames) <= self.config.subset_num * self.config.sampled_views_every_subset or self.config.load_all:
            camera = self.cameras[image_idx : image_idx + 1].to(self.device)
        else:
            camera = self.cameras[image_idx : image_idx + 1][0].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data
