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
Instruct-NeRF2NeRF configuration file.
"""
from pathlib import Path
from dataclasses import dataclass
from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig
from nerfstudio.configs.base_config import ViewerConfig
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.data.datasets.depth_dataset import DepthDataset

from gaussctrl.gs_datamanager import GaussEditDataManagerConfig
from gaussctrl.gs import GaussEditModelConfig
from gaussctrl.gs_pipeline import GaussEditPipelineConfig
from gaussctrl.gs_trainer import GaussEditTrainerConfig
from gaussctrl.gs_datamanager import GaussEditDataManager
from gaussctrl.gs_dataparser_ns import GaussEditDataParserConfig
from gaussctrl.gs_dataparser_colmap import ColmapDataParserConfig
from gaussctrl.gs_dataset import GSDataset
from nerfstudio.data.datamanagers.base_datamanager import VanillaDataManager, VanillaDataManagerConfig
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanager, FullImageDatamanagerConfig
from nerfstudio.plugins.registry_dataparser import DataParserSpecification    


gaussctrl_method = MethodSpecification(
    config=GaussEditTrainerConfig(
        method_name="gs-edit",
        steps_per_eval_image=100,
        steps_per_eval_batch=0,
        steps_per_save=250,
        max_num_iterations=1000,
        steps_per_eval_all_images=1000,
        save_only_latest_checkpoint=True,
        mixed_precision=False,
        gradient_accumulation_steps={"camera_opt": 100},
        pipeline=GaussEditPipelineConfig(
            datamanager=GaussEditDataManagerConfig(
                _target=GaussEditDataManager[GSDataset],
                dataparser=GaussEditDataParserConfig(load_3D_points=False,
                                            #   data=Path('data/bear60'),
                                            #   filename='outer_60.json'
                                              ),
                # dataparser=ColmapDataParserConfig(load_3D_points=False,
                #                             #   data=Path('data/bear60'),
                #                             #   filename='outer_60.json'
                #                               ),
                # dataparser=NerfstudioDataParserConfig(load_3D_points=False,
                #                             #   data=Path('data/bear60'),
                #                             #   filename='outer_60.json'
                #                               ),
                # camera_optimizer=CameraOptimizerConfig(
                #     mode="SO3xR3",
                #     optimizer=AdamOptimizerConfig(lr=6e-4, eps=1e-8, weight_decay=1e-2),
                #     scheduler=ExponentialDecaySchedulerConfig(lr_final=6e-6, max_steps=200000),
                # ),
            ),
            model=GaussEditModelConfig(),
        ),
        optimizers={
            "xyz": {
                "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1.6e-6,
                    max_steps=30000,
                ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacity": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scaling": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "rotation": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=5e-5, max_steps=30000),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),
        vis="viewer",
    ),
    description="GaussCtrl",
)

