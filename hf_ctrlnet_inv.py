from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline,UniPCMultistepScheduler, DDIMInverseScheduler, DDIMScheduler, StableDiffusionPipeline
from diffusers.utils import load_image
from torch import autocast, inference_mode
import numpy as np
import torch
from annotator.midas import MidasDetector
import cv2, os
from PIL import Image
import imageio
from annotator.util import resize_image, HWC3
from typing import Any, Callable, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import argparse
import torch.nn.functional as F
from diffusers.utils.torch_utils import randn_tensor
# from torch.utils.tensorboard import SummaryWriter

# writer = SummaryWriter('runs/log1')

def depth2disparity(depth):
    # depth: np array [512 512 1]
    disparity = 1 / (depth + 1e-5)
    disparity_map = disparity * 255 / np.max(disparity) 
    disparity_map = disparity_map.astype(np.uint8)[:,:,0]
    disparity_map = HWC3(disparity_map)
    return disparity_map

def load_512(image_path, left=0, right=0, top=0, bottom=0):
        if type(image_path) is str:
            image = np.array(Image.open(image_path))[:, :, :3]
        else:
            image = image_path
        h, w, c = image.shape
        if h == 512 and w == 512:
            return image
        
        left = min(left, w-1)
        right = min(right, w - left - 1)
        top = min(top, h - left - 1)
        bottom = min(bottom, h - top - 1)
        image = image[top:h-bottom, left:w-right]
        h, w, c = image.shape
        if h < w:
            offset = (w - h) // 2
            image = image[:, offset:offset + h]
        elif w < h:
            offset = (h - w) // 2
            image = image[offset:offset + w]
        image = np.array(Image.fromarray(image).resize((512, 512)))
        return image

@torch.no_grad()
def image2latent(image, pipe):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(device) # torch.Size([1, 3, 512, 512]) -1~1
            latents = pipe.vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents

@torch.no_grad()
def latent2image(latents, pipe, return_type='np'):
    latents = 1 / 0.18215 * latents.detach()
    image = pipe.vae.decode(latents)['sample']
    if return_type == 'np':
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
        image = (image * 255).astype(np.uint8)
    return image


if __name__ == "__main__":

    parser = argparse.ArgumentParser(prog='generation')
    parser.add_argument('--imgdir', type=str, default=None,                     
                        help="images to edit (here we don't use these images just be more convinient), i.e. data/bear512_innertraj/rgb")
    parser.add_argument('--depthdir', type=str, default=None,
                        help='i.e. data/bear512_innertraj/depth_npy')
    parser.add_argument('--savedir', type=str, default=None,
                        help='i.e. output/bear_controlnet_240_inner')
    parser.add_argument('--noisedir', type=str, default=None,
                        help='i.e. data/bear512_innertraj/bear512_noise_inner240')
    parser.add_argument('--prompt', type=str, default="a photo of panda in the forest")
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--device', type=str, default='cuda:1')
    args = parser.parse_args()

    # config
    IMGDIR = 'data_wholeset_views/bear_512/rgb'
    DEPTHDIR = 'data_wholeset_views/bear_512/depth_npy'
    SAVEDIR = 'data_wholeset_views/bear_512'
    # NOISEDIR = 'data/tnt_horse_fov75/z_0'
    prompt = 'a photo of a bear statue in the forest'
    num_inference_steps = 20
    seed = None
    device = 'cuda:0'
    guidance_scale = 0

    img_savedir = os.path.join(SAVEDIR, 'imgs_recover')
    disparity_savedir = os.path.join(SAVEDIR, 'disparity')
    z_0_savedir = os.path.join(SAVEDIR, 'z_0')
    depth_savedir = os.path.join(SAVEDIR, 'depth')
    coefimg_savedir = os.path.join(SAVEDIR, 'coef_exp')
    edited_savedir = os.path.join(SAVEDIR, 'edited')

    ablation_save = os.path.join('ablation_results/polar_bear/ctrlnet_no_depth_ctrl', 'inv_gs0')
    os.makedirs(ablation_save, exist_ok=True)
    # os.makedirs(img_savedir, exist_ok=True)
    # # os.makedirs(disparity_savedir, exist_ok=True)
    # os.makedirs(z_0_savedir, exist_ok=True)
    # # os.makedirs(depth_savedir, exist_ok=True)
    # # os.makedirs(coefimg_savedir, exist_ok=True)
    # # os.makedirs(edited_savedir, exist_ok=True)

    # # Save config
    # import yaml
    # config = {'IMGROOT': IMGDIR,
    #         'DEPTHDIR': DEPTHDIR,
    #         # 'NOISEDIR': NOISEDIR, 
    #         'num_inference_steps': num_inference_steps,
    #         'prompt':prompt,
    #         'script_name': os.path.basename(__file__)}

    # config_path = os.path.join(SAVEDIR, 'config.yml')
    # with open(config_path, 'w') as outfile:
    #     yaml.dump(config, outfile)
    

    # load control net and stable diffusion v1-5
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
    # controlnet = ControlNetModel.from_pretrained('lllyasviel/control_v11e_sd15_ip2p')
    pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet)
    pipe.to(device)
    # speed up diffusion process with faster scheduler and memory optimization
    # pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    # pipe.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
    # scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
    # ldm_stable = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", scheduler=scheduler).to(device)
    
    # inversepipe = Inversion(NUM_DDIM_STEPS=num_inference_steps, prompt=prompt, device=device)
    

    imgfiles = sorted(os.listdir(IMGDIR))
    latent_list = []
    for i, imgfile in enumerate(imgfiles):
        # imgfile = 'frame_00019.jpg'
        filename = imgfile.split('.')[0]
        imgpath = os.path.join(IMGDIR, imgfile) 
        imgsavepath = os.path.join(img_savedir, imgfile) 
        disparitysavepath = os.path.join(disparity_savedir, imgfile) 
        tar_view_idx = int(imgfile.split('.')[0].split('_')[-1]) - 1
        # image = Image.open(imgpath).convert("RGB") 

        # noisepath = os.path.join(NOISEDIR, imgfile.split('.')[0] + '.npy') 
        # if seed == None:
        #     noisepath = os.path.join(NOISEDIR, imgfile.split('.')[0] + '.npy') 
        #     noise = torch.Tensor(np.load(noisepath)).to(device)

        ############################################
        # depthpath = os.path.join(DEPTHDIR, imgfile.split('.')[0] + '.npy') 
        # depth = np.load(depthpath)
        # depth = np.zeros((512, 512, 1))
        

        disparity = np.ones((512, 512, 1))
        # disparity = 1 / (depth + 1e-5)
        disparity_map = disparity * 255 / np.max(disparity) 
        disparity_map = disparity_map.astype(np.uint8)[:,:,0]
        disparity_map = HWC3(disparity_map)
        # imageio.imwrite(disparitysavepath, disparity_map)

        disparity = Image.fromarray(disparity_map) 


        image = load_512(imgpath)
        init_latent = image2latent(image=image, pipe=pipe) 
        pipe.scheduler = DDIMInverseScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        # pipe.scheduler.set_timesteps(50)
        
        latent, latents_list = pipe(prompt, 
                                    num_inference_steps=num_inference_steps, 
                                    latents=init_latent, 
                                    image=disparity, return_dict=False, guidance_scale=0, output_type='latent')

        # # Load noise
        # latent = torch.Tensor(np.load(noisepath)).to(device)

        # # save noise
        # latent_save = latent.clone().cpu().numpy()
        # np.save(os.path.join(z_0_savedir, filename+'.npy'),latent_save) # os.path.join(z_0_savedir, filename+'.npy')
        # mask = -(np.array(Image.open('bear_mask_view_215_ref19_wrt_214.jpg')) - 255) > 0 # (512 512)
        # mask = torch.Tensor(mask)[None,None,...]
        # breakpoint()
        # mask = F.interpolate(mask, size=(64, 64), mode='nearest').to(device)
        
        # random_latent = randn_tensor((1,4,64,64), generator=None, device=device, dtype=torch.float32)
        # latent_mask = mask 
        # random_mask = 1 - latent_mask
        # # breakpoint()
        # masked_latent = random_latent * random_mask + latent * latent_mask
        # # mask1 = (mask[0,0,:,:].cpu().numpy() * 255).astype(np.uint8)
        # # mask1 = Image.fromarray(mask1)
        # # mask1.save('mask.jpg')
        # # masked_latent = latent * mask

        # bear_statue_latent = np.load('data/bear512_innertraj/bear512_noise_inner240/frame_00215.npy')
        # bear_statue_latent = torch.Tensor(bear_statue_latent).to(device)
    
        # alpha = 0.01
        # latents = alpha * latent_list[0] + (1-alpha) * latent_list[2]
        pipe.scheduler = DDIMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        image, latent_list = pipe(prompt, num_inference_steps=num_inference_steps, image=disparity, latents=latent, return_dict=False, guidance_scale=0)

        # # masked_latent = latent * mask
        # # image = pipe.vae.decode(masked_latent / pipe.vae.config.scaling_factor, return_dict=False, generator=None)[0].detach()
        # # image = pipe.image_processor.postprocess(image, output_type='pil', do_denormalize=None)  

        edited_savepath = os.path.join(ablation_save, imgfile)
        image[0].save(edited_savepath) 
        # breakpoint()
        # print(f'save to {imgsavepath}') 
