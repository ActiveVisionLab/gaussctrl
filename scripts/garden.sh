# ns-train splatfacto --output-dir unedited_models --experiment-name garden --viewer.quit-on-train-completion True nerfstudio-data --data data/garden

ns-train gaussctrl --load-checkpoint unedited_models/garden/splatfacto/2024-07-11_173647/nerfstudio_models/step-000029999.ckpt --experiment-name garden --output-dir outputs --pipeline.datamanager.data data/garden --pipeline.prompt "a photo of a fake plant on a table in the garden in the snow" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 