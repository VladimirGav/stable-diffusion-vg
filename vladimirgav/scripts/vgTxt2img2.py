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
from sys import argv
from os.path import abspath

DirCur = os.path.dirname(os.path.abspath(__file__))

FilePathJson = DirCur+"/../inputdata/txt2img.json"
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

model_id = 'stabilityai/stable-diffusion-2-1-base'
if 'model_id' in arrData:
    model_id = arrData['model_id']

imgs_dir = DirCur+"/../imgs"
if not os.path.exists(imgs_dir): os.makedirs(imgs_dir)

if 'imgs_dir' in arrData:
    imgs_dir = arrData['imgs_dir']

imgs_count = 1
if 'imgs_count' in arrData:
    imgs_count = arrData['imgs_count']


resultArr = {'img_id': img_id, 'prompt': prompt, 'model_id': model_id, 'imgs_dir': imgs_dir, 'imgs_count': imgs_count}
#print(resultArr)
#exit()

model_path = 'D:\sd\stable-diffusion\vladimirgav\models\dungeonsNWaifusNew_dungeonsNWaifus22.safetensors'

from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler, StableDiffusionImg2ImgPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)

#scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

#Low GPU pipe.enable_attention_slicing()
pipe.enable_attention_slicing()

for i in range(imgs_count):
    resultArr['imgs'] = {}
    resultArr['imgs'][i] = {}
    FileName = str(int(time.time()))+str(".png")
    resultArr['imgs'][i]['FileName'] = FileName
    FilePath = imgs_dir+"/"+FileName
    resultArr['imgs'][i]['FilePath'] = FilePath
    image = pipe(prompt).images[0]
    image.save(FilePath)

print(json.dumps(resultArr))