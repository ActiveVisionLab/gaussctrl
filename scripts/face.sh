ns-train splatfacto --output-dir unedited_models --experiment-name face --viewer.quit-on-train-completion True nerfstudio-data --data data/face

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of a face of a man with a moustache" --pipeline.guidance_scale 3 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of bronze bust statue of a man" --pipeline.guidance_scale 3 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of a man wearing a pair of glasses" --pipeline.guidance_scale 3 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of a face of a Jocker with green hair" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of a face of an old man with wrinkles" --pipeline.guidance_scale 3 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of a man wearing a pair of sunglasses" --pipeline.guidance_scale 3 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of a face of a woman with thick made-up" --pipeline.guidance_scale 3 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/face/splatfacto/2024-07-11_173339/nerfstudio_models/step-000029999.ckpt --experiment-name face --output-dir outputs --pipeline.datamanager.data data/face --pipeline.prompt "a photo of a face of a man with red hair" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'man' --viewer.quit-on-train-completion True 
