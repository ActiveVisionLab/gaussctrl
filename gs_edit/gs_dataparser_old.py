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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio, os
import numpy as np
import torch

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import Nerfstudio
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json
from nerfstudio.cameras.camera_paths import get_path_from_json
from typing import Literal, Optional, Type
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class GaussEditDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: GaussEditDataParser)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "white"
    """alpha color of background"""
    filename: str = "camera_path_outer240.json"
    load_3D_points: bool = False
    """Whether to load the 3D points from the colmap reconstruction."""
    orientation_method: Literal["pca", "up", "vertical", "none"] = "up"


@dataclass
class GaussEditDataParser(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: GaussEditDataParserConfig

    def __init__(self, config: GaussEditDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.file_name = config.filename
        if self.alpha_color is not None:
            self.alpha_color_tensor = get_color(self.alpha_color)
        else:
            self.alpha_color_tensor = None

    def _generate_dataparser_outputs(self, split="train"):
        '''load data to train from the sample trajectory''' 
        nerfstudio_dataparser = Nerfstudio
        assert self.config.data.exists(), f"Data directory {self.config.data} does not exist."
        
        sampled_json = load_from_json(self.data / self.file_name) 
        image_filenames = [self.data / Path(f'images/frame_{idx+1:05d}.jpg') for idx in range(len(sampled_json["camera_path"]))]
        unedited_image_filenames = [self.data / Path(f'unedited/frame_{idx+1:05d}.jpg') for idx in range(len(sampled_json["camera_path"]))]
        depth_filenames = [self.data / Path(f'depth_npy/frame_{idx+1:05d}.npy') for idx in range(len(sampled_json["camera_path"]))]
        z_0_filenames = [self.data / Path(f'z_0/frame_{idx+1:05d}.npy') for idx in range(len(sampled_json["camera_path"]))]
        if os.path.isdir(self.data / Path(f'masks_npy')):
            mask_filenames = [self.data / Path(f'masks_npy/frame_{idx+1:05d}.npy') for idx in range(len(sampled_json["camera_path"]))]
        
        cameras = get_path_from_json(sampled_json) 
        scene_box = SceneBox(aabb=torch.tensor([[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]], dtype=torch.float32))

        metadata = {
                "unedited_image_filenames": unedited_image_filenames,
                "depth_filenames": depth_filenames, # [PosixPath('data/bear/depths/frame_00096.npy')]
                "z_0_filenames": z_0_filenames,
                "mask_filenames": mask_filenames,
                "data": self.data
            } if os.path.isdir(self.data / Path(f'masks_npy')) else {
                "unedited_image_filenames": unedited_image_filenames,
                "depth_filenames": depth_filenames, # [PosixPath('data/bear/depths/frame_00096.npy')]
                "z_0_filenames": z_0_filenames,
                "data": self.data
            }

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames, # [PosixPath('data/bear_512/images/frame_00096.jpg')]
            cameras=cameras,
            alpha_color=self.alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            metadata=metadata
        )
        

        return dataparser_outputs

   