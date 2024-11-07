<p align="center">
  
  <h1 align="center"><strong>🎥 [ECCV 2024] GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing</strong></h3>

  <p align="center">
    <a href="https://jingwu2121.github.io/" class="name-link" target="_blank">Jing Wu<sup>*1</sup> </a>,
    <a href="https://jwbian.net/" class="name-link" target="_blank">Jia-Wang Bian<sup>*2</sup> </a>,
    <a href="https://xinghui-li.github.io/" class="name-link" target="_blank">Xinghui Li<sup>1</sup></a>,
    <a href="https://wanggrun.github.io/" class="name-link" target="_blank">Guangrun Wang<sup>1</sup></a>,
    <a href="https://mbzuai.ac.ae/study/faculty/ian-reid/" class="name-link" target="_blank">Ian Reid<sup>2</sup></a>,
    <a href="https://www.robots.ox.ac.uk/~phst/" class="name-link" target="_blank">Philip Torr<sup>1</sup></a>,
    <a href="https://www.robots.ox.ac.uk/~victor/" class="name-link" target="_blank">Victor Adrian Prisacariu<sup>1</sup></a>
    <br>
    * denotes equal contribution
    <br>
    <sup>1</sup>University of Oxford,
    <br>
<sup>2</sup>Mohamed bin Zayed University of Artificial Intelligence
</p>

<div align="center">

[![Badge with Logo](https://img.shields.io/badge/arXiv-2403.08733-red?logo=arxiv)
](https://arxiv.org/abs/2403.08733)
[![Badge with Logo](https://img.shields.io/badge/Project-Page-blue?logo=homepage)](https://gaussctrl.active.vision/)
[![Badge with Logo](https://img.shields.io/badge/Download-Data-cyan)](https://github.com/jingwu2121/gaussctrl/tree/main/data)
[![Badge with Logo](https://img.shields.io/badge/BSD-License-green)](LICENSE.txt)
</div>

![teaser](./assets/teaser.png)

## ✨ News
- [9.4.2024] Our original results utilise stable-diffusion-v1-5 from runwayml for editing, which is now unavailable. Please change the diffusion checkpoint to other available models, e.g. `CompVis/stable-diffusion-v1-4`, by using `--pipeline.diffusion_ckpt "CompVis/stable-diffusion-v1-4"`. Reproduce our original results by using the checkpoint `--pipeline.diffusion_ckpt "jinggogogo/gaussctrl-sd15"` 

## ⚙️ Installation

- Tested on CUDA11.8 + Ubuntu22.04 + NeRFStudio1.0.0 (NVIDIA RTX A5000 24G)

Clone the repo. 
```bash
git clone https://github.com/ActiveVisionLab/gaussctrl.git
cd gaussctrl
```

### 1. NeRFStudio and Lang-SAM

```bash
conda create -n gaussctrl python=3.8
conda activate gaussctrl
conda install cuda -c nvidia/label/cuda-11.8.0
```

GaussCtrl is built upon NeRFStudio, follow [this link](https://docs.nerf.studio/quickstart/installation.html) to install NeRFStudio first. If you are failing to build tiny-cuda-nn, try building from scratch, see [here](https://github.com/NVlabs/tiny-cuda-nn/?tab=readme-ov-file#compilation-windows--linux). We recommend using NeRFStudio v1.0.0 with gsplat v0.1.3. 

```bash
pip install nerfstudio==1.0.0

# Try either of these two if one is not working
pip install gsplat==0.1.2
pip install gsplat==0.1.3
```

Install Lang-SAM for mask extraction. 

```bash
pip install -U git+https://github.com/luca-medeiros/lang-segment-anything.git

pip install -r requirements.txt
```

### 2. Install GaussCtrl
```bash 
pip install -e .
```

### 3. Verify the install
```bash
ns-train -h
```

## 🗄️ Data

### Use Our Preprocessed Data

Our preprocessed data are under the `data` folder, where
- `fangzhou` is from [NeRF-Art](https://github.com/cassiePython/NeRF-Art/tree/main/data/fangzhou_nature) 
- `bear`, `face` are from [Instruct-NeRF2NeRF](https://drive.google.com/drive/folders/1v4MLNoSwxvSlWb26xvjxeoHpgjhi_s-s?usp=share_link) 
- `garden` is from [Mip-NeRF 360](http://storage.googleapis.com/gresearch/refraw360/360_v2.zip) 
- `stone horse` and `dinosaur` are from [BlendedMVS](https://github.com/YoYo000/BlendedMVS) 

We thank these authors for their great work!

### Customize Your Data

We recommend to pre-process your data to 512x512, and following [this page](https://docs.nerf.studio/quickstart/custom_dataset.html) to process your data. 

## :arrow_forward: Get Started
![Method](./assets/method.png)

### 1. Train a 3DGS
To get started, you first need to train your 3DGS model. We use `splatfacto` from NeRFStudio. 

```bash 
ns-train splatfacto --output-dir {output/folder} --experiment-name EXPEIMENT_NAME nerfstudio-data --data {path/to/your/data}
```

### 2. Edit your model
Once you finish training the `splatfacto` model, the checkpoints will be saved to `output/folder/EXPEIMENT_NAME` folder. 

Start editing your model by running:

```bash
ns-train gaussctrl --load-checkpoint {output/folder/.../nerfstudio_models/step-000029999.ckpt} --experiment-name EXPEIMENT_NAME --output-dir {output/folder} --pipeline.datamanager.data {path/to/your/data} --pipeline.edit_prompt "YOUR PROMPT" --pipeline.reverse_prompt "PROMPT TO DESCRIBE THE UNEDITED SCENE" --pipeline.guidance_scale 5 --pipeline.chunk_size {batch size of images during editing} --pipeline.langsam_obj 'OBJECT TO BE EDITED' 
```

Please note that the Lang-SAM is optional here. If you are editing the environment, please remove this argument. 

```bash
ns-train gaussctrl --load-checkpoint {output/folder/.../nerfstudio_models/step-000029999.ckpt} --experiment-name EXPEIMENT_NAME --output-dir {output/folder} --pipeline.datamanager.data {path/to/your/data} --pipeline.edit_prompt "YOUR PROMPT" --pipeline.reverse_prompt "PROMPT TO DESCRIBE THE UNEDITED SCENE" --pipeline.guidance_scale 5 --pipeline.chunk_size {batch size of images during editing} 
```

Here, `--pipeline.guidance_scale` denotes the classifier-free guidance used when editing the images. `--pipeline.chunk_size` denotes the number of images edited together during 1 batch. We are using **NVIDIA RTX A5000** GPU (24G), and the maximum chunk size is 3. (~22G) 

Control the number of reference views using `--pipeline.ref_view_num`, by default, it is set to 4. 

### Small Tips
- If your editings are not as expected, please check the images edited by ControlNet. 
- Normally, conditioning your editing on the good ControlNet editing views is very helpful, which means choosing those good ControlNet editing views as reference views is better. 

## :wrench: Reproduce Our Results

Experiments in the main paper are included in the `scripts` folder. To reproduce the results, first train the `splatfacto` model. We take the `bear` case as an example here. 
```bash
ns-train splatfacto --output-dir unedited_models --experiment-name bear nerfstudio-data --data data/bear
```

Then edit the 3DGS by running:
```bash
ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-07-10_170906/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a polar bear in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True 
```

In our experiments, We sampled 40 views randomly from the entire dataset to accelerate the method, which is set in `gc_datamanager.py` by default. We split the entire set into 4 subsets, and randomly sampled 10 images in each subset split. Feel free to decrease/increase the number to see the difference by modifying `--pipeline.datamanager.subset-num` and `--pipeline.datamanager.sampled-views-every-subset`. Set `--pipeline.datamanager.load-all` to `True`, if you want to edit all the images in the dataset. 

## :camera: View Results Using NeRFStudio Viewer
```bash
ns-viewer --load-config {outputs/.../config.yml} 
```

## :movie_camera: Render Your Results
- Render all the dataset views. 
```bash 
ns-gaussctrl-render dataset --load-config {outputs/.../config.yml} --output_path {render/EXPEIMENT_NAME} 
```

- Render a mp4 of a camera path
```bash
ns-gaussctrl-render camera-path --load-config {outputs/.../config.yml} --camera-path-filename data/EXPEIMENT_NAME/camera_paths/render-path.json --output_path render/EXPEIMENT_NAME.mp4
```

## Evaluation
We use [this code](https://github.com/ayaanzhaque/instruct-nerf2nerf/tree/main/metrics) to evaluate our method. 

## Citation
If you find this code or find the paper useful for your research, please consider citing:
```
@article{gaussctrl2024,
author = {Wu, Jing and Bian, Jia-Wang and Li, Xinghui and Wang, Guangrun and Reid, Ian and Torr, Philip and Prisacariu, Victor},
title = {{GaussCtrl: Multi-View Consistent Text-Driven 3D Gaussian Splatting Editing}},
journal = {ECCV},
year = {2024},
}
```
