from diffusers import StableDiffusionPipeline
import torch
from utils_sd import register_normal_pipeline, register_faster_forward, register_parallel_pipeline, seed_everything  # 1.import package

import time

seed_everything(2)
model_id = "KBlueLeaf/kohaku-v2.1"
pipe = StableDiffusionPipeline.from_pretrained("KBlueLeaf/kohaku-v2.1", torch_dtype=torch.float16,safety_checker = None)
pipe = pipe.to("cuda")

#------------------------------
# register_parallel_pipeline(pipe) # 2. enable parallel. If memory is limited, replace it with  `register_normal_pipeline(pipe)`
# register_faster_forward(pipe.unet,5,'50ls')  # 3. encoder propagation
#------------------------------
prompt = "Girl with panda ears wearing a hood"

start = time.time()
# image = pipe.call(prompt).images[0]  
image = pipe(prompt).images[0]
end = time.time()

print(f"Time: {end-start}")
    
image.save("cat.png")