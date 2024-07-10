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
Instruct-NeRF2NeRF Datamanager.
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
from gs_edit.gs_dataset import GSDataset
from nerfstudio.data.datasets.base_dataset import InputDataset
from gs_edit import utils



CONSOLE = Console(width=120)

@dataclass
class GaussEditDataManagerConfig(FullImageDatamanagerConfig):
    """Configuration for the InstructNeRF2NeRFDataManager."""

    _target: Type = field(default_factory=lambda: GaussEditDataManager)
    patch_size: int = 32
    """Size of patch to sample from. If >1, patch-based sampling will be used."""
    sampled_view_split: int = 4
    sampled_view_num_every: int = 10
    load_all: bool = False
    


class GaussEditDataManager(FullImageDatamanager, Generic[TDataset]):
    """Data manager for GaussEdit."""

    config: GaussEditDataManagerConfig
    train_dataset: TDataset
    eval_dataset: TDataset

    def __init__(self,
                config: GaussEditDataManagerConfig,
                device: Union[torch.device, str] = "cpu",
                test_mode: Literal["test", "val", "inference"] = "val",
                world_size: int = 1,
                local_rank: int = 0,
                **kwargs,):
        super().__init__(config, device, test_mode, world_size, local_rank)
        # random.seed(13789)

        self.sample_idx = []
        self.step_every = 1
        self.edited_image_dict = {}
        # self.good_views = [1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 38, 40, 41, 42, 43, 45, 46, 48, 49, 50, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 69, 70, 72, 78]

        # self.cameras = self.train_dataset.cameras
        # self.train_data = self.cached_train
        
        # Sample data
        if len(self.train_dataset._dataparser_outputs.image_filenames) <= self.config.sampled_view_split * self.config.sampled_view_num_every or self.config.load_all:
            self.cameras = self.train_dataset.cameras
            self.train_data = self.cached_train
            self.train_unseen_cameras = list(range(len(self.train_data)))
        else:
            view_num = len(self.train_dataset._dataparser_outputs.image_filenames)
            anchors = list(range(0, view_num, view_num // self.config.sampled_view_split))[:4] + [view_num]
            sampled_indices = []
            for idx in range(len(anchors[:-1])):
                cur_anchor = anchors[idx]
                next_anchor = anchors[idx+1] 
                selected = sorted(random.sample(list(range(cur_anchor, next_anchor)), self.config.sampled_view_num_every))
                sampled_indices += selected

            self.cameras = [self.train_dataset.cameras[i: i+1] for i in sampled_indices]
            self.train_data_temp = [self.cached_train[i] for i in sampled_indices]
            self.train_data = []
            for i, data in enumerate(self.train_data_temp):
                data['image_idx'] = i
                self.train_data.append(data)
            self.train_unseen_cameras = list(range(self.config.sampled_view_split * self.config.sampled_view_num_every))
        

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

    def cache_images_old(self, cache_images_option):
        cached_train = []
        cached_eval = []
        CONSOLE.log("Caching train data")
        for i in tqdm(range(len(self.train_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.train_dataset.get_data(i, image_type=self.config.cache_images_type)
            camera = self.train_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                cached_train.append(data)
                continue
            else:
                raise ValueError(f"{camera.distortion_params} is not None. Unsupported distorted cameras")

        CONSOLE.log("Caching eval data")
        for i in tqdm(range(len(self.eval_dataset)), leave=False):
            # cv2.undistort the images / cameras
            data = self.eval_dataset.get_data(i, image_type=self.config.cache_images_type)
            camera = self.eval_dataset.cameras[i].reshape(())
            K = camera.get_intrinsics_matrices().numpy()
            if camera.distortion_params is None:
                cached_eval.append(data)
                continue
            else:
                raise ValueError(f"{camera.distortion_params} is not None. Unsupported distorted cameras")
            
        # if cache_images_option == "gpu":
        #     for cache in cached_train:
        #         cache["depth_image"] = cache["depth_image"].to(self.device)
        #         cache["z_0_image"] = cache["z_0_image"].to(self.device)
        #         if "image" in cache:
        #             cache["image"] = cache["image"].to(self.device)
        #     for cache in cached_eval:
        #         cache["depth_image"] = cache["depth_image"].to(self.device)
        #         cache["z_0_image"] = cache["z_0_image"].to(self.device)
        #         if "image" in cache:
        #             cache["image"] = cache["image"].to(self.device)
        # else:
        #     for cache in cached_train:
        #         cache["depth_image"] = cache["depth_image"].pin_memory()
        #         cache["z_0_image"] = cache["z_0_image"].pin_memory()
        #         if "image" in cache:
        #             cache["image"] = cache["image"].to(self.device)
        #     for cache in cached_eval:
        #         cache["depth_image"] = cache["depth_image"].pin_memory()
        #         cache["z_0_image"] = cache["z_0_image"].pin_memory()
        #         if "image" in cache:
        #             cache["image"] = cache["image"].to(self.device)
        return cached_train, cached_eval

    @cached_property
    def dataset_type(self) -> Type[TDataset]:
        """Returns the dataset type passed as the generic argument"""
        default: Type[TDataset] = cast(TDataset, TDataset.__default__)  # type: ignore
        orig_class: Type[GaussEditDataManager] = get_orig_class(self, default=None)  # type: ignore
        if type(self) is GaussEditDataManager and orig_class is None:
            return default
        if orig_class is not None and get_origin(orig_class) is GaussEditDataManager:
            return get_args(orig_class)[0]

        # For inherited classes, we need to find the correct type to instantiate
        for base in getattr(self, "__orig_bases__", []):
            if get_origin(base) is GaussEditDataManager:
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
        
        # rand_idx = random.choice(range(len(self.good_views))) 
        # selected_idx = self.good_views.pop(rand_idx) 
        # image_idx = self.train_unseen_cameras[selected_idx] 
        
        # Make sure to re-populate the unseen cameras list if we have exhausted it
        if len(self.train_unseen_cameras) == 0:
            self.train_unseen_cameras = [i for i in range(len(self.train_data))]
        # if len(self.good_views) == 0:
            # self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]
            # self.good_views = [1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 15, 16, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 38, 40, 41, 42, 43, 45, 46, 48, 49, 50, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 69, 70, 72, 78]
        
        data = deepcopy(self.train_data[image_idx]) # image torch.Size([512, 512, 3]) cpu 0-1
        data["image"] = data["image"].to(self.device)
        
        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        if len(self.train_dataset._dataparser_outputs.image_filenames) <= self.config.sampled_view_split * self.config.sampled_view_num_every:
            camera = self.cameras[image_idx : image_idx + 1].to(self.device)
        else:
            camera = self.cameras[image_idx : image_idx + 1][0].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = image_idx
        return camera, data

    def next_train_serial(self, step: int) -> Tuple[Cameras, Dict]:
        """Serial Sampling"""

        if (step % self.step_every == 0):
            image_idx = self.train_unseen_cameras.pop(random.randint(0, len(self.train_unseen_cameras) - 1))

            self.sample_idx.append(image_idx)
            # self.step_every += 1 * len(self.sample_idx) 
            if len(self.sample_idx) < 10:
                self.step_every += 1 * len(self.sample_idx) 
            elif len(self.sample_idx) >= 10 and len(self.sample_idx) < 50:
                self.step_every = 10 * len(self.sample_idx) 
            else:
                self.step_every = 30 * len(self.sample_idx) 
        # breakpoint()
        print(step)
        print(self.step_every)
        print(self.train_unseen_cameras)
        print(self.sample_idx)
        
        current_idx = self.sample_idx[random.randint(0, len(self.sample_idx) - 1)]
        print(current_idx)

        # # Make sure to re-populate the unseen cameras list if we have exhausted it
        # if len(self.train_unseen_cameras) == 0:
        #     self.train_unseen_cameras = [i for i in range(len(self.train_dataset))]

        data = deepcopy(self.cached_train[current_idx])
        data["image"] = data["image"].to(self.device)

        assert len(self.train_dataset.cameras.shape) == 1, "Assumes single batch dimension"
        camera = self.train_dataset.cameras[current_idx : current_idx + 1].to(self.device)
        if camera.metadata is None:
            camera.metadata = {}
        camera.metadata["cam_idx"] = current_idx
        return camera, data
