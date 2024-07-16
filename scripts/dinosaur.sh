ns-train splatfacto --output-dir unedited_models --experiment-name dinosaur --viewer.quit-on-train-completion True nerfstudio-data --data data/dinosaur

ns-train gaussctrl --load-checkpoint unedited_models/dinosaur/splatfacto/2024-07-11_173113/nerfstudio_models/step-000029999.ckpt --experiment-name dinosaur --output-dir outputs --pipeline.datamanager.data data/dinosaur --pipeline.prompt "a photo of a robot dinosaur on the road side" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'dinosaur statue' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/dinosaur/splatfacto/2024-07-11_173113/nerfstudio_models/step-000029999.ckpt --experiment-name dinosaur --output-dir outputs --pipeline.datamanager.data data/dinosaur --pipeline.prompt "a photo of a dinosaur statue under the water" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/dinosaur/splatfacto/2024-07-11_173113/nerfstudio_models/step-000029999.ckpt --experiment-name dinosaur --output-dir outputs --pipeline.datamanager.data data/dinosaur --pipeline.prompt "a photo of a dinosaur statue in the snow" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/dinosaur/splatfacto/2024-07-11_173113/nerfstudio_models/step-000029999.ckpt --experiment-name dinosaur --output-dir outputs --pipeline.datamanager.data data/dinosaur --pipeline.prompt "a photo of a dinosaur statue at night" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/dinosaur/splatfacto/2024-07-11_173113/nerfstudio_models/step-000029999.ckpt --experiment-name dinosaur --output-dir outputs --pipeline.datamanager.data data/dinosaur --pipeline.prompt "a photo of a dinosaur statue in the storm" --pipeline.guidance_scale 7.5 --pipeline.chunk_size 3 --viewer.quit-on-train-completion True 
