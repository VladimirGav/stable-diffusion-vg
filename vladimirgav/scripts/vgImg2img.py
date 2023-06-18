#!/usr/bin/env python # [1]
"""
 * VladimirGav
 * GitHub Website: https://vladimirgav.github.io/
 * GitHub: https://github.com/VladimirGav
 * Source https://github.com/VladimirGav/stable-diffusion-vg
 * Copyright (c)
"""

import time
import os
import json
import requests
from sys import argv
from os.path import abspath
from PIL import Image
from io import BytesIO

DirCur = os.path.dirname(os.path.abspath(__file__))
DirRoot = DirCur+"/../.."

FilePathJson = DirCur+"/../inputdata/img2img.json"
if len(argv) > 1:
    FilePathJson = argv[1]

file_inputdata = FilePathJson
file = open(file_inputdata, 'r')
arrData = json.loads(file.read())
file.close()

img_id = 0
if 'img_id' in arrData:
    img_id = arrData['img_id']

prompt = 'fox'
if 'prompt' in arrData:
    prompt = arrData['prompt']

negative_prompt = ''
if 'negative_prompt' in arrData:
    negative_prompt = arrData['negative_prompt']

model_id = 'stabilityai/stable-diffusion-2-1-base'
if 'model_id' in arrData:
    model_id = arrData['model_id']

imgs_dir = DirCur+"/../imgs"
if not os.path.exists(imgs_dir): os.makedirs(imgs_dir)

if 'imgs_dir' in arrData:
    imgs_dir = arrData['imgs_dir']
    imgs_dir = imgs_dir.replace("sdroot", DirRoot)

imgs_count = 1
if 'imgs_count' in arrData:
    imgs_count = arrData['imgs_count']

img_width = 512
if 'img_width' in arrData:
    img_width = arrData['img_width']

img_height = 768
if 'img_height' in arrData:
    img_height = arrData['img_height']

img_num_inference_steps = 25
if 'img_num_inference_steps' in arrData:
    img_num_inference_steps = arrData['img_num_inference_steps']

img_guidance_scale = 7.5
if 'img_guidance_scale' in arrData:
    img_guidance_scale = arrData['img_guidance_scale']

#safetensors file
model_lora_weights = ''
if 'model_lora_weights' in arrData:
    model_lora_weights = arrData['model_lora_weights']

img_original = DirCur+"/../inputdata/img2img.jpg"
if 'img_original' in arrData:
    img_original = arrData['img_original']
    img_original = img_original.replace("sdroot", DirRoot)

#img_original = img_original.replace("sdroot", DirCur+"/../..")
#print(img_original)
#print(2233)

#url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"
#response = requests.get(url)
#init_image = Image.open(BytesIO(response.content)).convert("RGB")

file_img_content = img_original
init_image = Image.open(file_img_content).convert("RGB")
init_image.thumbnail((768, 768))

resultArr = {
'img_id': img_id,
'prompt': prompt,
'negative_prompt': negative_prompt,
'model_id': model_id,
'imgs_dir': imgs_dir,
'imgs_count': imgs_count,
'img_width': img_width,
'img_height': img_height,
'img_num_inference_steps': img_num_inference_steps,
'img_guidance_scale': img_guidance_scale,
'model_lora_weights': model_lora_weights,
'img_original': img_original
}
#print(resultArr)
#exit()

#exit()

import torch
from diffusers import StableDiffusionImg2ImgPipeline, DPMSolverMultistepScheduler

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
device = "cuda"
pipe = pipe.to(device)

#Low GPU pipe.enable_attention_slicing()
pipe.enable_attention_slicing()

#generator = torch.Generator(device=device).manual_seed(1024)
for i in range(imgs_count):
    resultArr['imgs'] = {}
    resultArr['imgs'][i] = {}
    FileName = str(int(time.time()))+str(".png")
    resultArr['imgs'][i]['FileName'] = FileName
    FilePath = imgs_dir+"/"+FileName
    resultArr['imgs'][i]['FilePath'] = FilePath
    image = pipe(prompt=prompt, image=init_image, negative_prompt=negative_prompt, num_inference_steps=img_num_inference_steps, guidance_scale=img_guidance_scale).images[0]
    image.save(FilePath)

print(json.dumps(resultArr))