import os

import PIL.Image
import numpy as np
import torch
import torchvision
from torchvision.transforms import Resize, InterpolationMode
import imageio
from einops import rearrange
import cv2, glob
from PIL import Image
# from annotator.canny import CannyDetector
# from annotator.openpose import OpenposeDetector
# from annotator.midas import MidasDetector
# import decord
from diffusers.utils import USE_PEFT_BACKEND

# apply_canny = CannyDetector()
# apply_openpose = OpenposeDetector()
# apply_midas = MidasDetector()

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def add_watermark(image, watermark_path, wm_rel_size=1/16, boundary=5):
    '''
    Creates a watermark on the saved inference image.
    We request that you do not remove this to properly assign credit to
    Shi-Lab's work.
    '''
    watermark = Image.open(watermark_path)
    w_0, h_0 = watermark.size
    H, W, _ = image.shape
    wmsize = int(max(H, W) * wm_rel_size)
    aspect = h_0 / w_0
    if aspect > 1.0:
        watermark = watermark.resize((wmsize, int(aspect * wmsize)), Image.LANCZOS)
    else:
        watermark = watermark.resize((int(wmsize / aspect), wmsize), Image.LANCZOS)
    w, h = watermark.size
    loc_h = H - h - boundary
    loc_w = W - w - boundary
    image = Image.fromarray(image)
    mask = watermark if watermark.mode in ('RGBA', 'LA') else None
    image.paste(watermark, (loc_w, loc_h), mask)
    return image


def pre_process_canny(input_video, low_threshold=100, high_threshold=200):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def pre_process_depth(input_video, apply_depth_detect: bool = True):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        img = HWC3(img)
        if apply_depth_detect:
            detected_map, _ = apply_midas(img) # (512, 512) 0~255
            a = Image.fromarray(detected_map)
            a.save('try1.jpg')
            breakpoint()
        else:
            detected_map = img
        detected_map = HWC3(detected_map) # 0~255
        breakpoint()
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps) # (8, 512, 512, 3) 0~255 
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0 # 0~1
    breakpoint()
    return rearrange(control, 'f h w c -> f c h w')

def read_depth2disparity(depth_dir):
    depth_paths = sorted(glob.glob(depth_dir + '/*.npy'))
    disparity_list = []
    for depth_path in depth_paths:
        depth = np.load(depth_path) # [512,512,1] 
        
        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity) # 0.00233~1
        # disparity_map = disparity_map.astype(np.uint8)[:,:,0]
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=2)
        disparity_list.append(disparity_map[None]) 

    detected_maps = np.concatenate(disparity_list, axis=0)
    
    control = torch.from_numpy(detected_maps.copy()).float()
    return rearrange(control, 'f h w c -> f c h w')

def read_depth(depth_dir):
    depth_paths = sorted(glob.glob(depth_dir + '/*.npy'))
    depth_list = []
    for depth_path in depth_paths:
        depth = np.load(depth_path) # [512,512,1] 
        
        depth_list.append(depth[None]) 

    detected_maps = np.concatenate(depth_list, axis=0)
    
    control = torch.from_numpy(detected_maps.copy()).float()
    return rearrange(control, 'f h w c -> f c h w')    

def pre_process_pose(input_video, apply_pose_detect: bool = True):
    detected_maps = []
    for frame in input_video:
        img = rearrange(frame, 'c h w -> h w c').cpu().numpy().astype(np.uint8)
        img = HWC3(img)
        if apply_pose_detect:
            detected_map, _ = apply_openpose(img)
        else:
            detected_map = img
        detected_map = HWC3(detected_map)
        H, W, C = img.shape
        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)
        detected_maps.append(detected_map[None])
    detected_maps = np.concatenate(detected_maps)
    control = torch.from_numpy(detected_maps.copy()).float() / 255.0
    return rearrange(control, 'f h w c -> f c h w')


def create_video(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'movie.mp4')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)

        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def create_gif(frames, fps, rescale=False, path=None, watermark=None):
    if path is None:
        dir = "temporal"
        os.makedirs(dir, exist_ok=True)
        path = os.path.join(dir, 'canny_db.gif')

    outputs = []
    for i, x in enumerate(frames):
        x = torchvision.utils.make_grid(torch.Tensor(x), nrow=4)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        if watermark is not None:
            x = add_watermark(x, watermark)
        outputs.append(x)
        # imageio.imsave(os.path.join(dir, os.path.splitext(name)[0] + f'_{i}.jpg'), x)

    imageio.mimsave(path, outputs, fps=fps)
    return path

def prepare_video(video_path:str, resolution:int, device, dtype, normalize=True, start_t:float=0, end_t:float=-1, output_fps:int=-1):
    vr = decord.VideoReader(video_path)
    initial_fps = vr.get_avg_fps()
    if output_fps == -1:
        output_fps = int(initial_fps)
    if end_t == -1:
        end_t = len(vr) / initial_fps
    else:
        end_t = min(len(vr) / initial_fps, end_t)
    assert 0 <= start_t < end_t
    assert output_fps > 0
    start_f_ind = int(start_t * initial_fps)
    end_f_ind = int(end_t * initial_fps)
    num_f = int((end_t - start_t) * output_fps)
    sample_idx = np.linspace(start_f_ind, end_f_ind, num_f, endpoint=False).astype(int)
    video = vr.get_batch(sample_idx)
    if torch.is_tensor(video):
        video = video.detach().cpu().numpy()
    else:
        video = video.asnumpy()
    _, h, w, _ = video.shape
    video = rearrange(video, "f h w c -> f c h w")
    video = torch.Tensor(video).to(device).to(dtype)

    # Use max if you want the larger side to be equal to resolution (e.g. 512)
    # k = float(resolution) / min(h, w)
    k = float(resolution) / max(h, w)
    h *= k
    w *= k
    h = int(np.round(h / 64.0)) * 64
    w = int(np.round(w / 64.0)) * 64

    video = Resize((h, w), interpolation=InterpolationMode.BILINEAR, antialias=True)(video)
    if normalize:
        video = video / 127.5 - 1.0
    return video, output_fps


def post_process_gif(list_of_results, image_resolution):
    output_file = "/tmp/ddxk.gif"
    imageio.mimsave(output_file, list_of_results, fps=4)
    return output_file

def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask):
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    key_ref_cross = key_ref_cross[:, ref_frame_index]
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")
    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")

    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)
    attention_probs = attn.get_attention_scores(query, key_ref_cross, attention_mask)
    hidden_states_ref_cross = torch.bmm(attention_probs, value_ref_cross) 
    return hidden_states_ref_cross

class CrossFrameAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff

    def diffuser014(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None):
        batch_size, sequence_length, _ = hidden_states.shape
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.cross_attention_norm:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query) 
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        # query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states_cross = torch.bmm(attention_probs, value)

        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * hidden_states_cross if not is_cross_attention else hidden_states_cross 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def condat1view(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):

        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        query = attn.head_to_batch_dim(query)
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            former_frame_index = [0] * video_length
            
            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, former_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, former_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        # query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states_cross = torch.bmm(attention_probs, value)

        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * hidden_states_cross if not is_cross_attention else hidden_states_cross 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    
    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0,):

        residual = hidden_states
        
        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        # breakpoint()
        query = attn.head_to_batch_dim(query)
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size
            # former_frame_index = torch.arange(video_length) - 1
            # former_frame_index[0] = 0
            ref_frame_index = [0] * video_length
            ang1_frame_index = [1] * video_length
            ang2_frame_index = [2] * video_length
            ang3_frame_index = [3] * video_length
            
            hidden_states_ref = compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask)
            hidden_states_ang1 = compute_attn(attn, query, key, value, video_length, ang1_frame_index, attention_mask)
            hidden_states_ang2 = compute_attn(attn, query, key, value, video_length, ang2_frame_index, attention_mask)

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, ang3_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, ang3_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")


        # query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        # breakpoint()
        hidden_states_ang3 = torch.bmm(attention_probs, value)

        coef = 0.25
        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * (0.25 * hidden_states_ang1 + 0.25 * hidden_states_ang2 + 0.25 * hidden_states_ang3 + 0.25 * hidden_states_ref) if not is_cross_attention else hidden_states_ang3 
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
