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

"""GaussCtrl Pipeline and trainer"""

import os
from dataclasses import dataclass, field
from itertools import cycle
from typing import Optional, Type, List
from rich.progress import Console
from copy import deepcopy
import numpy as np 
from PIL import Image
import open3d as o3d
import mediapy as media

import torch, random
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal
from nerfstudio.pipelines.base_pipeline import VanillaPipeline, VanillaPipelineConfig
from nerfstudio.viewer.server.viewer_elements import ViewerNumber, ViewerText
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
from gaussctrl.gs_datamanager import (
    GaussEditDataManagerConfig,
)
from diffusers.models.attention_processor import AttnProcessor
from gaussctrl import utils
from nerfstudio.viewer_legacy.server.utils import three_js_perspective_camera_focal_length
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.utils import colormaps

from diffusers import StableDiffusionInstructPix2PixPipeline, StableDiffusionControlNetPipeline, ControlNetModel, UNet2DConditionModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler

CONSOLE = Console(width=120)

@dataclass
class GaussEditPipelineConfig(VanillaPipelineConfig):
    """Configuration for pipeline instantiation"""

    _target: Type = field(default_factory=lambda: GaussEditPipeline)
    """target class to instantiate"""
    datamanager: GaussEditDataManagerConfig = GaussEditDataManagerConfig()
    """specifies the datamanager config"""
    render_rate: int = 1000
    """how many gauss steps for gauss training"""
    prompt: str = "a photo of a polar bear in the forest"
    # bear dataset parameters
    dataset_update_num: int = 1
    guidance_scale: float = 5
    # ref_views_indices: List = [20, 55] #face: 20, 55 bear: [22, 45] fangzhou: [4, 29], stone horse [14, 65], fern: [4, 11], garden: [18, 23]
    num_inference_steps: int = 20
    chunk_size: int = 5
    

class GaussEditPipeline(VanillaPipeline):
    """InstructNeRF2NeRF pipeline"""

    config: GaussEditPipelineConfig

    def __init__(
        self,
        config: GaussEditPipelineConfig,
        device: str,
        test_mode: Literal["test", "val", "inference"] = "val",
        world_size: int = 1,
        local_rank: int = 0,
        grad_scaler: Optional[GradScaler] = None,
    ):
        super().__init__(config, device, test_mode, world_size, local_rank)
        self.prompt = self.config.prompt
        # self.device = 'cuda:0'
        self.pipe_device = 'cuda:0'
        self.ddim_scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet).to(self.device).to(torch.float16)
        self.pipe.scheduler = self.ddim_scheduler
        self.pipe.unet.set_attn_processor(
                        processor=utils.CrossFrameAttnProcessor(self_attn_coeff=0.6,
                        unet_chunk_size=2))
        self.pipe.controlnet.set_attn_processor(
                        processor=utils.CrossFrameAttnProcessor(self_attn_coeff=0,
                        unet_chunk_size=2)) 
        self.pipe.to(self.pipe_device)

        added_prompt = 'best quality, extremely detailed'
        self.positive_prompt = self.prompt + ', ' + added_prompt
        self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        # if str(self.datamanager.train_dataset._dataparser_outputs.metadata['data']).split('/')[-1].split('_')[0] in ['bear', 'garden']:
        
        view_num = len(self.datamanager.train_data) # self.config.datamanager.sampled_view_split * self.config.datamanager.sampled_view_num_every # len(self.datamanager.train_dataset._dataparser_outputs.image_filenames) 
        anchors = list(range(0, view_num, view_num // 4)) + [view_num]
        
        random.seed(13789)
        self.ref_indices = [random.randint(anchor, anchors[idx+1]) for idx, anchor in enumerate(anchors[:-1])]  # [18, 23]
        self.num_ref_views = len(self.ref_indices)

        self.num_inference_steps = self.config.num_inference_steps
        self.guidance_scale = self.config.guidance_scale
        self.controlnet_conditioning_scale = 1.0
        self.eta = 0.0
        self.chunk_size = self.config.chunk_size
        
        if test_mode == "val":
            print("#############################")
            CONSOLE.print("Start doing initial generation: ", style="bold yellow")
            CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}, counting from 1", style="bold yellow")
            print("#############################")
            ref_disparity_list = []
            ref_z0_list = []
            for ref_idx in self.ref_indices:
                ref_data = deepcopy(self.datamanager.train_data[ref_idx]) 
                ref_disparity = self.depth2disparity(ref_data['depth_image']) # 
                ref_z0 = ref_data['z_0_image']
                ref_disparity_list.append(ref_disparity)
                ref_z0_list.append(ref_z0) 
                
            ref_disparities = np.concatenate(ref_disparity_list, axis=0)
            ref_z0s = np.concatenate(ref_z0_list, axis=0)
            ref_disparity_torch = torch.from_numpy(ref_disparities.copy()).to(torch.float16).to(self.pipe_device)
            ref_z0_torch = torch.from_numpy(ref_z0s.copy()).to(torch.float16).to(self.pipe_device)

            for idx in range(len(self.datamanager.train_data) // self.chunk_size):
                chunked_data = self.datamanager.train_data[idx*self.chunk_size: (idx+1)*self.chunk_size]
                
                indices = [current_data['image_idx'] for current_data in chunked_data]
                mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()] 
                unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
                CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

                depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
                disparities = np.concatenate(depth_images, axis=0)
                disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

                z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
                z0s = np.concatenate(z_0_images, axis=0)
                latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

                disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
                latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)
                
                chunk_edited = self.pipe(
                                    prompt=[self.positive_prompt] * (self.num_ref_views+self.chunk_size),
                                    negative_prompt=[self.negative_prompts] * (self.num_ref_views+self.chunk_size),
                                    latents=latents_chunk,
                                    image=disp_ctrl_chunk,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=self.guidance_scale,
                                    controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                                    eta=self.eta,
                                    output_type='pt',
                                ).images[self.num_ref_views:]
                chunk_edited = chunk_edited.cpu() 

                # insert edited images back to cache image for training
                for local_idx, edited_image in enumerate(chunk_edited):
                    global_idx = indices[local_idx]

                    bg_cntrl_edited_image = edited_image
                    if mask_images != []:
                        mask = torch.from_numpy(mask_images[local_idx])
                        bg_mask = 1 - mask

                        unedited_image = unedited_images[local_idx].permute(2,0,1)
                        bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None] 

                    # # ###########################################
                    # m = mask.numpy() * 255
                    # m_np = m.astype(np.uint8)
                    # m_pil = Image.fromarray(m_np)
                    # m_pil.save('temp_mask/'+ f'{global_idx}.jpg')
                                
                    # a = bg_cntrl_edited_image * 255
                    # a_np = a.permute(1,2,0).numpy().astype(np.uint8)
                    # a_pil = Image.fromarray(a_np)
                    # a_pil.save('temp/'+ f'{global_idx}.jpg')

                    save_mid_dir = 'mid_results'
                    dataset_name = str(self.datamanager.train_dataset._dataparser_outputs.image_filenames[0]).split('/')[1]
                    mid_save_path = f'{save_mid_dir}/{dataset_name}_gs7.5/'+self.prompt
                    os.makedirs(mid_save_path, exist_ok=True)
                    b = edited_image * 255
                    b_np = b.permute(1,2,0).numpy().astype(np.uint8)
                    b_pil = Image.fromarray(b_np)
                    b_pil.save(os.path.join(mid_save_path, f'frame_{global_idx+1:05d}.jpg'))

                    # b = unedited_image * 255
                    # b_np = b.permute(1,2,0).numpy().astype(np.uint8)
                    # b_pil = Image.fromarray(b_np)
                    # b_pil.save(f'try.jpg')
                    # breakpoint()
                    # ###########################################

                    self.datamanager.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32) # [3 512 512]
                    # self.datamanager.cached_eval[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)
                    self.datamanager.train_data[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)
                    # self.datamanager.eval_dataset[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)
            # breakpoint()
            print("#############################")
            CONSOLE.print("Done initial generation: ", style="bold yellow")
            print("#############################")

    def sample_traj(self):
        ################################
        output_dir = 'temp'
        image_format = 'jpeg'

        input_file = "exports/splat/garden.ply"
        pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
        pcd_np = np.asarray(pcd.points) 
        obj_mid = np.mean(pcd_np, axis=0) 

        cameras = self.datamanager.train_dataset.cameras
        c2ws = cameras.camera_to_worlds

        pose_center_point = c2ws[..., 3].mean(0)  # (3)
        pose_center_point[-1] = (c2ws[..., -1][...,-1].min() + c2ws[..., -1][...,-1].max()) / 2
        pose_avg = self.avg_pose(c2ws)
        # pose_avg[:,-1] = pose_center_point
        
        up = c2ws[..., 1].mean(0)

        # dist = numpy.linalg.norm(a-b)
        depth_of_views = [np.linalg.norm(obj_mid-i) for i in c2ws[..., -1].clone().numpy()]
        avg_depth = sum(depth_of_views) / len(depth_of_views) 
        
        n_frames = 40
        radii = np.array([avg_depth, avg_depth, 1., 1.])
        render_poses = []
        for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
            t = radii * [np.cos(theta), -np.sin(theta), 0, 1.]
            position = np.array([avg_depth, avg_depth, 1.]) * [np.cos(theta), -np.sin(theta), 0] + pose_center_point.numpy()  # pose_avg @ t # (3)
            lookat = obj_mid
            z_axis = position - lookat
            render_poses.append(self.viewmatrix(z_axis, up, position))
        render_poses = torch.from_numpy(np.stack(render_poses, axis=0))

        fov = 50 
        image_height = 512
        image_width = 512
        focal_length = three_js_perspective_camera_focal_length(fov, image_height)
        fxs = fys = [focal_length] * n_frames

        fx = torch.tensor(fxs)
        fy = torch.tensor(fys)
        render_cam_traj = Cameras(
                            fx=fx.to(torch.float32),
                            fy=fy.to(torch.float32),
                            cx=torch.tensor(image_width / 2, dtype=torch.float32),
                            cy=torch.tensor(image_height / 2, dtype=torch.float32),
                            camera_to_worlds=render_poses.to(torch.float32),
                            camera_type=CameraType.PERSPECTIVE, 
                        ).to(self.pipe_device)
        
        for camera_idx in range(render_cam_traj.size):
            outputs = self.model.get_outputs_for_camera(
                            render_cam_traj[camera_idx : camera_idx + 1], obb_box=None
                        )
            
            render_image = []
            render_image_dict = {}
            for rendered_output_name in ['rgb', 'depth']:
                output_image = outputs[rendered_output_name]
                is_depth = rendered_output_name.find("depth") != -1
                if is_depth:
                    depth_dir = os.path.join(output_dir, 'depth_npy')
                    depth_path = os.path.join(depth_dir, f"frame_{camera_idx+1:05d}.npy")
                    os.makedirs(depth_dir, exist_ok=True)
                    np.save(depth_path, output_image.cpu().numpy())

                    output_image = (
                        colormaps.apply_depth_colormap(
                            output_image,
                            accumulation=outputs["accumulation"],
                            near_plane=None,
                            far_plane=None,
                            colormap_options=colormaps.ColormapOptions(),
                        )
                        .cpu()
                        .numpy()
                    )
                else:
                    output_image = (
                        colormaps.apply_colormap(
                            image=output_image,
                            colormap_options=colormaps.ColormapOptions(),
                        )
                        .cpu()
                        .numpy()
                    )
                render_image_dict[rendered_output_name] = output_image
                render_image.append(output_image)
            
            render_image = np.concatenate(render_image, axis=1)
            os.makedirs(os.path.join(output_dir, 'render'), exist_ok=True)
            
            if image_format == "jpeg":
                media.write_image(
                    os.path.join(output_dir, 'render', f"{camera_idx:05d}.jpg"), render_image, fmt="jpeg", quality=100
                )
                for key in render_image_dict.keys():
                    os.makedirs(os.path.join(output_dir, key), exist_ok=True)
                    media.write_image(
                        os.path.join(output_dir, key, f"frame_{camera_idx+1:05d}.jpg"), render_image_dict[key], fmt="jpeg", quality=100
                    )
        breakpoint()
        ################################

    def sample_traj1(self):
        ################################
        output_dir = 'temp'
        image_format = 'jpeg'

        input_file = "exports/splat/garden.ply"
        pcd = o3d.io.read_point_cloud(input_file) # Read the point cloud
        pcd_np = np.asarray(pcd.points) 
        obj_mid = np.mean(pcd_np, axis=0) 

        cameras = self.datamanager.train_dataset.cameras
        c2ws = cameras.camera_to_worlds

        pose_center_point = c2ws[..., 3].mean(0)  # (3)
        pose_center_point[-1] = (c2ws[..., -1][...,-1].min() + c2ws[..., -1][...,-1].max()) / 2
        pose_avg = self.avg_pose(c2ws)
        # pose_avg[:,-1] = pose_center_point
        
        up = c2ws[..., 1].mean(0)

        # dist = numpy.linalg.norm(a-b)
        depth_of_views = [np.linalg.norm(obj_mid-i) for i in c2ws[..., -1].clone().numpy()]
        avg_depth = sum(depth_of_views) / len(depth_of_views) 
        
        n_frames = 40
        radii = np.array([avg_depth, avg_depth, 1., 1.])
        render_poses = []
        for theta in np.linspace(0., 2. * np.pi, n_frames, endpoint=False):
            t = radii * [np.cos(theta), -np.sin(theta), 0, 1.]
            position = np.array([avg_depth, avg_depth, 1.]) * [np.cos(theta), -np.sin(theta), 0] + pose_center_point.numpy()  # pose_avg @ t # (3)
            lookat = obj_mid
            z_axis = position - lookat
            render_poses.append(self.viewmatrix(z_axis, up, position))
        render_poses = torch.from_numpy(np.stack(render_poses, axis=0))

        fov = 50 
        image_height = 512
        image_width = 512
        focal_length = three_js_perspective_camera_focal_length(fov, image_height)
        fxs = fys = [focal_length] * n_frames

        fx = torch.tensor(fxs)
        fy = torch.tensor(fys)
        render_cam_traj = Cameras(
                            fx=fx.to(torch.float32),
                            fy=fy.to(torch.float32),
                            cx=torch.tensor(image_width / 2, dtype=torch.float32),
                            cy=torch.tensor(image_height / 2, dtype=torch.float32),
                            camera_to_worlds=render_poses.to(torch.float32),
                            camera_type=CameraType.PERSPECTIVE, 
                        ).to(self.pipe_device)
        
        for camera_idx in range(render_cam_traj.size):
            outputs = self.model.get_outputs_for_camera(
                            render_cam_traj[camera_idx : camera_idx + 1], obb_box=None
                        )
            
            render_image = []
            render_image_dict = {}
            for rendered_output_name in ['rgb', 'depth']:
                output_image = outputs[rendered_output_name]
                is_depth = rendered_output_name.find("depth") != -1
                if is_depth:
                    depth_dir = os.path.join(output_dir, 'depth_npy')
                    depth_path = os.path.join(depth_dir, f"frame_{camera_idx+1:05d}.npy")
                    os.makedirs(depth_dir, exist_ok=True)
                    np.save(depth_path, output_image.cpu().numpy())

                    output_image = (
                        colormaps.apply_depth_colormap(
                            output_image,
                            accumulation=outputs["accumulation"],
                            near_plane=None,
                            far_plane=None,
                            colormap_options=colormaps.ColormapOptions(),
                        )
                        .cpu()
                        .numpy()
                    )
                else:
                    output_image = (
                        colormaps.apply_colormap(
                            image=output_image,
                            colormap_options=colormaps.ColormapOptions(),
                        )
                        .cpu()
                        .numpy()
                    )
                render_image_dict[rendered_output_name] = output_image
                render_image.append(output_image)
            
            render_image = np.concatenate(render_image, axis=1)
            os.makedirs(os.path.join(output_dir, 'render'), exist_ok=True)
            
            if image_format == "jpeg":
                media.write_image(
                    os.path.join(output_dir, 'render', f"{camera_idx:05d}.jpg"), render_image, fmt="jpeg", quality=100
                )
                for key in render_image_dict.keys():
                    os.makedirs(os.path.join(output_dir, key), exist_ok=True)
                    media.write_image(
                        os.path.join(output_dir, key, f"frame_{camera_idx+1:05d}.jpg"), render_image_dict[key], fmt="jpeg", quality=100
                    )
        breakpoint()
        ################################

    def viewmatrix(self, lookdir: np.ndarray, up: np.ndarray,
               position: np.ndarray) -> np.ndarray:
        """Construct lookat view matrix."""
        vec2 = self.normalize(lookdir)
        vec0 = self.normalize(np.cross(up, vec2))
        vec1 = self.normalize(np.cross(vec2, vec0))
        m = np.stack([vec0, vec1, vec2, position], axis=1)
        return m

    def normalize(self, v):
        """Normalize a vector."""
        return v / np.linalg.norm(v)

    def avg_pose(self, c2ws):
        center = c2ws[..., 3].mean(0)
        z = self.normalize(c2ws[..., 2].mean(0))  # (3)
        y_ = c2ws[..., 1].mean(0)  # (3)
        x = self.normalize(np.cross(y_, z))  # (3)
        y = np.cross(z, x)  # (3)
        pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)
        
        return pose_avg

    @torch.no_grad()
    def image2latent(self, image):
        image = image * 2 - 1
        image = image.permute(2, 0, 1).unsqueeze(0) # torch.Size([1, 3, 512, 512]) -1~1
        latents = self.pipe.vae.encode(image)['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    def depth2disparity(self, depth):
        """
        Args: depth numpy array [1 512 512]
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=0)
        return disparity_map[None]
    
    def depth2disparity_torch(self, depth):
        """
        Args: depth torch tensor
        Return: disparity
        """
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / torch.max(disparity) # 0.00233~1
        disparity_map = torch.concatenate([disparity_map, disparity_map, disparity_map], dim=0)
        return disparity_map[None]

    def update_datasets(self, cam_idx, depth, latent):
        self.datamanager.cached_train[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy() # [1 512 512]
        # self.datamanager.cached_eval[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        self.datamanager.train_dataset[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()
        # self.datamanager.eval_dataset[cam_idx]["depth_image"] = depth.permute(2,0,1).cpu().to(torch.float32).numpy()

        self.datamanager.cached_train[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        # self.datamanager.cached_eval[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        self.datamanager.train_dataset[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        # self.datamanager.eval_dataset[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()

    def get_train_loss_dict(self, step: int, count: int):
        """This function gets your training loss dict and performs image editing.
        Args:
            step: current iteration step to update sampler if using DDP (distributed)
        """
        ray_bundle, batch = self.datamanager.next_train(step) # camera, data
        model_outputs = self._model(ray_bundle)  # train distributed data parallel model if world_size > 1
        
        metrics_dict = self.model.get_metrics_dict(model_outputs, batch)
        loss_dict = self.model.get_loss_dict(model_outputs, batch, metrics_dict)
        # breakpoint()
        # if step > 30500:
        #     a = model_outputs['rgb'].clone().detach().cpu().numpy() * 255
        #     a_np = a.astype(np.uint8)
        #     a_pil = Image.fromarray(a_np)
        #     a_pil.save('try.jpg')
        #     batch_img = batch['image'].clone().detach().cpu().numpy() * 255
        #     batch_img_np = batch_img.astype(np.uint8)
        #     batch_img_pil = Image.fromarray(batch_img_np)
        #     batch_img_pil.save('try.jpg')
        #     breakpoint()

        if ((step+1) % self.config.render_rate == 0) and count < self.config.dataset_update_num - 1:
            for cam_idx in range(len(self.datamanager.train_dataset.cameras)):
                CONSOLE.print(f"Rendering view {cam_idx}, counting from 0", style="bold yellow")
                current_cam = self.datamanager.train_dataset.cameras[cam_idx: cam_idx+1].to(self.device)
                if current_cam.metadata is None:
                    current_cam.metadata = {}
                current_cam.metadata["cam_idx"] = cam_idx
                rendered_image = self._model.get_outputs_for_camera(current_cam)

                image = rendered_image['rgb'].to(torch.float16) # [512 512 3] 0-1
                depth = rendered_image['depth'].to(torch.float16) # [512 512 1]

                # save_img = (image.cpu().numpy() * 255).astype(np.uint8)
                # frame_pil = Image.fromarray(save_img) 
                # frame_pil.save(os.path.join(f'test/render', f'frame_{cam_idx+1:05d}.jpg'))
                
                # reverse the images to noises
                self.pipe.unet.set_attn_processor(processor=AttnProcessor())
                self.pipe.controlnet.set_attn_processor(processor=AttnProcessor()) 
                init_latent = self.image2latent(image)
                disparity = self.depth2disparity_torch(depth[:,:,0][None]) 
                
                self.pipe.scheduler = self.ddim_inverser
                latent, _ = self.pipe(prompt=self.positive_prompt,
                                    negative_prompt=self.negative_prompts,
                                    num_inference_steps=self.num_inference_steps, 
                                    latents=init_latent, 
                                    image=disparity, return_dict=False, guidance_scale=0, output_type='latent')
                                
                self.update_datasets(cam_idx, depth, latent)

                # latent = self.datamanager.cached_train[cam_idx]["z_0_image"]
                # depth = self.datamanager.cached_train[cam_idx]["depth_image"]
                # np.save(os.path.join('test/inv_z0', f'frame_{cam_idx+1:05}.npy'), latent)
                # np.save(os.path.join('test/inv_depth', f'frame_{cam_idx+1:05}.npy'), depth)
                # self.datamanager.pipe.scheduler = self.datamanager.ddim_scheduler
                # image, _ = self.datamanager.pipe(prompt=self.datamanager.positive_prompt,
                #                     negative_prompt=self.datamanager.negative_prompts,
                #                     num_inference_steps=self.datamanager.num_inference_steps, 
                #                     latents=latent, 
                #                     image=disparity, return_dict=False, guidance_scale=0, output_type='pil')
                
                # image[0].save(os.path.join(f'test/ddim_back2img', f'frame_{cam_idx+1:05d}.jpg'))
            
            # Generating images
            print("#############################")
            CONSOLE.print("Start doing editing generation: ", style="bold yellow")
            CONSOLE.print(f"Reference views are {[j+1 for j in self.ref_indices]}, counting from 1", style="bold yellow")
            print("#############################")
            self.pipe.scheduler = self.ddim_scheduler
            self.pipe.unet.set_attn_processor(
                            processor=utils.CrossFrameAttnProcessor(self_attn_coeff=0.6,
                            unet_chunk_size=2))
            self.pipe.controlnet.set_attn_processor(
                            processor=utils.CrossFrameAttnProcessor(self_attn_coeff=0,
                            unet_chunk_size=2)) 

            ref_disparity_list = []
            ref_z0_list = []
            for ref_idx in self.ref_indices:
                ref_data = deepcopy(self.datamanager.cached_train[ref_idx]) 
                ref_disparity = self.depth2disparity(ref_data['depth_image'])
                ref_z0 = ref_data['z_0_image']
                ref_disparity_list.append(ref_disparity)
                ref_z0_list.append(ref_z0) 
                
            ref_disparities = np.concatenate(ref_disparity_list, axis=0)
            ref_z0s = np.concatenate(ref_z0_list, axis=0)
            
            ref_disparity_torch = torch.from_numpy(ref_disparities.copy()).to(torch.float16).to(self.pipe_device)
            ref_z0_torch = torch.from_numpy(ref_z0s.copy()).to(torch.float16).to(self.pipe_device)

            for idx in range(len(self.datamanager.cached_train) // self.chunk_size):
                chunked_data = self.datamanager.cached_train[idx*self.chunk_size: (idx+1)*self.chunk_size]
                
                indices = [current_data['image_idx'] for current_data in chunked_data]
                mask_images = [current_data['mask_image'] for current_data in chunked_data if 'mask_image' in current_data.keys()] 
                unedited_images = [current_data['unedited_image_filenames'] for current_data in chunked_data]
                CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

                depth_images = [self.depth2disparity(current_data['depth_image']) for current_data in chunked_data]
                disparities = np.concatenate(depth_images, axis=0)
                disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

                z_0_images = [current_data['z_0_image'] for current_data in chunked_data] # list of np array
                z0s = np.concatenate(z_0_images, axis=0)
                latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

                disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
                latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)
                
                chunk_edited = self.pipe(
                                    prompt=[self.positive_prompt] * (self.num_ref_views+self.chunk_size),
                                    negative_prompt=[self.negative_prompts] * (self.num_ref_views+self.chunk_size),
                                    latents=latents_chunk,
                                    image=disp_ctrl_chunk,
                                    num_inference_steps=self.num_inference_steps,
                                    guidance_scale=0,
                                    controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                                    eta=self.eta,
                                    output_type='pt',
                                ).images[self.num_ref_views:]
                chunk_edited = chunk_edited.cpu() 
                
                # ############# TEMP
                # save_img = self.pipe.image_processor.pt_to_numpy(chunk_edited)
                # for k, frame in enumerate(save_img):
                #     # if i == 0: 
                #     idx_temp = idx*6 + k
                #     frame = (frame * 255).astype(np.uint8)
                #     frame_pil = Image.fromarray(frame) 
                #     frame_pil.save(os.path.join(f'test', f'frame_{idx_temp+1:05d}.jpg'))
                # ############# TEMP

                # insert edited images back to cache image for training
                for local_idx, edited_image in enumerate(chunk_edited):
                    global_idx = indices[local_idx]

                    bg_cntrl_edited_image = edited_image
                    if mask_images != []:
                        mask = torch.from_numpy(mask_images[local_idx])
                        bg_mask = 1 - mask

                        unedited_image = unedited_images[local_idx].permute(2,0,1)
                        bg_cntrl_edited_image = edited_image * mask[None] + unedited_image * bg_mask[None] 

                    # ###########################################
                    # m = mask.numpy() * 255
                    # m_np = m.astype(np.uint8)
                    # m_pil = Image.fromarray(m_np)
                    # m_pil.save('temp_mask/'+ f'{global_idx}.jpg')
                                
                    # a = bg_cntrl_edited_image * 255
                    # a_np = a.permute(1,2,0).numpy().astype(np.uint8)
                    # a_pil = Image.fromarray(a_np)
                    # a_pil.save('temp/'+ f'{global_idx}.jpg')

                    # b = edited_image * 255
                    # b_np = b.permute(1,2,0).numpy().astype(np.uint8)
                    # b_pil = Image.fromarray(b_np)
                    # b_pil.save('test/'+ f'{global_idx}.jpg')
                    # # breakpoint()
                    # ###########################################

                    self.datamanager.cached_train[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32) # [3 512 512]
                    # self.datamanager.cached_eval[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)
                    self.datamanager.train_dataset[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)
                    # self.datamanager.eval_dataset[global_idx]["image"] = bg_cntrl_edited_image.permute(1,2,0).to(torch.float32)

        return model_outputs, loss_dict, metrics_dict

    def forward(self):
        """Not implemented since we only want the parameter saving of the nn module, but not forward()"""
        raise NotImplementedError
