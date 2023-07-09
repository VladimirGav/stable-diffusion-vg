#!/usr/bin/env python # [1]
"""
 * VladimirGav
 * GitHub Website: https://vladimirgav.github.io/
 * GitHub: https://github.com/VladimirGav
 * Source https://github.com/VladimirGav/stable-diffusion-vg
 * Copyright (c)
"""

import sys, os
import time
import os
import json
from sys import argv
from os.path import abspath
import random

DirCur = os.path.dirname(os.path.abspath(__file__))
DirRoot = DirCur+"/../.."

FilePathJson = DirCur+"/../inputdata/txt2img.json"
if len(argv) > 1:
    FilePathJson = argv[1]

file_inputdata = FilePathJson
file = open(file_inputdata, 'r')
arrData = json.loads(file.read())
file.close()

lora_dir = DirCur+"/../lora"

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
if not os.path.exists(imgs_dir):os.makedirs(imgs_dir)

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

#safetensors file
SamplerStr = 'DPM++ SDE Karras'
if 'sampler' in arrData:
    SamplerStr = arrData['sampler']

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
'model_lora_weights': model_lora_weights
}

#print(resultArr)
#exit()

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, DiffusionPipeline
from diffusers import EulerDiscreteScheduler, DDPMScheduler, DPMSolverMultistepScheduler, DPMSolverSDEScheduler
from compel import Compel

from safetensors.torch import load_file
from collections import defaultdict
from diffusers.loaders import LoraLoaderMixin
import torch

#scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")
#pipe = StableDiffusionPipeline.from_pretrained(model_id, scheduler=scheduler, torch_dtype=torchDtype, safety_checker=None)

#pipe = DiffusionPipeline.from_pretrained(model_id)

torchDtype = torch.float16
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torchDtype, safety_checker=None, requires_safety_checker = False)
device = "cuda"
pipe = pipe.to(device)
pipe.enable_attention_slicing() #Low GPU pipe.enable_attention_slicing()
pipe.enable_sequential_cpu_offload() # my graphics card VRAM is very low

#pipeline = DiffusionPipeline.from_pretrained("id/id")
#pipeline.save_pretrained("local_path", safe_serialization=True) # save in safetensors format

def set_scheduler(pipe, SamplerStr):
    SamplerStr = SamplerStr.lower()
    SamplerNewStr = ''

    if "euler" in SamplerStr:
        scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        SamplerNewStr = 'euler'
    elif "ddpm" in SamplerStr:
        scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
        SamplerNewStr = 'ddpm'
    elif "dpm++ sde" in SamplerStr:
        scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        scheduler.config.algorithm_type = 'sde-dpmsolver++'
        SamplerNewStr = 'dpm++ sde'
    elif "dpm++" in SamplerStr:
        scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        scheduler.config.algorithm_type = 'dpmsolver++'
        SamplerNewStr = 'dpm++'
    else:
        scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        SamplerNewStr = 'dpm++'

    if "karras" in SamplerStr:
            scheduler.config.use_karras_sigmas = True
            SamplerNewStr = SamplerNewStr+' karras'

    return scheduler, SamplerNewStr

#pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

#SamplerStr = 'DPM++ SDE Karras'
scheduler, SamplerNewStr = set_scheduler(pipe=pipe, SamplerStr=SamplerStr)
pipe.scheduler = scheduler
resultArr['sampler']=SamplerNewStr


# LoRA Start
#https://github.com/huggingface/diffusers/issues/3064
current_pipeline = None
original_weights = {}

def load_lora_weights(pipeline, checkpoint_path, multiplier, device, dtype):
    LORA_PREFIX_UNET = "lora_unet"
    LORA_PREFIX_TEXT_ENCODER = "lora_te"
    # load LoRA weight from .safetensors
    state_dict = load_file(checkpoint_path, device=device)

    updates = defaultdict(dict)
    for key, value in state_dict.items():
        # it is suggested to print out the key, it usually will be something like below
        # "lora_te_text_model_encoder_layers_0_self_attn_k_proj.lora_down.weight"

        layer, elem = key.split('.', 1)
        updates[layer][elem] = value

    # directly update weight in diffusers model
    for layer, elems in updates.items():

        if "text" in layer:
            layer_infos = layer.split(LORA_PREFIX_TEXT_ENCODER + "_")[-1].split("_")
            curr_layer = pipeline.text_encoder
        else:
            layer_infos = layer.split(LORA_PREFIX_UNET + "_")[-1].split("_")
            curr_layer = pipeline.unet

        # find the target layer
        temp_name = layer_infos.pop(0)
        while len(layer_infos) > -1:
            try:
                curr_layer = curr_layer.__getattr__(temp_name)
                if len(layer_infos) > 0:
                    temp_name = layer_infos.pop(0)
                elif len(layer_infos) == 0:
                    break
            except Exception:
                if len(temp_name) > 0:
                    temp_name += "_" + layer_infos.pop(0)
                else:
                    temp_name = layer_infos.pop(0)

        # get elements for this layer
        weight_up = elems['lora_up.weight'].to(dtype)
        weight_down = elems['lora_down.weight'].to(dtype)
        alpha = elems['alpha']
        if alpha:
            alpha = alpha.item() / weight_up.shape[1]
        else:
            alpha = 1.0

        # update weight
        if len(weight_up.shape) == 4:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up.squeeze(3).squeeze(2), weight_down.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
        else:
            curr_layer.weight.data += multiplier * alpha * torch.mm(weight_up, weight_down)

    return pipeline

# LoRa End

# add LoRA
#model_path = "sayakpaul/sd-model-finetuned-lora-t4"
#pipe.unet.load_attn_procs(model_path)
#image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5, cross_attention_kwargs={"scale": 0.5}).images[0]]

lora_path = lora_dir+'/hairdetailer.safetensors'
#pipe = load_lora_weights(pipe, lora_path, 1.0, device, torchDtype)
#sys.exit(0)

# use civitai.com lora soon
#model_lora_weights = "D:\\sd\\stablediffusion\\vladimirgav\\models\\aZovyaRPGArtistTools_sd21768V1.safetensors"
#if model_lora_weights != '':
#    pipe.load_lora_weights(".", weight_name=model_lora_weights)


#pipe(prompt=prompt, image=img, negative_prompt=None, strength=0.7).images[0]

# For long prompt
max_length_prompt = 365
# If one is short then add ...
def adjust_prompt(prompt, negative_prompt):
    if len(prompt) > max_length_prompt and len(negative_prompt) < max_length_prompt:
        negative_prompt += ' ' + '.' * (max_length_prompt - len(negative_prompt) + 10)
    elif len(negative_prompt) > max_length_prompt and len(prompt) < max_length_prompt:
        prompt += ' ' + '.' * (max_length_prompt - len(prompt) + 10)

    return prompt, negative_prompt

def get_pipeline_embeds(pipeline, prompt, negative_prompt, device):
    # If one is short then add ...
    prompt, negative_prompt = adjust_prompt(prompt, negative_prompt)

    """ Get pipeline embeds for prompts bigger than the maxlength of the pipe
    :param pipeline:
    :param prompt:
    :param negative_prompt:
    :param device:
    :return:
    """
    max_length = pipeline.tokenizer.model_max_length

    # simple way to determine length of tokens
    count_prompt = len(prompt.split(" "))
    count_negative_prompt = len(negative_prompt.split(" "))

    # create the tensor based on which prompt is longer
    if count_prompt >= count_negative_prompt:
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = input_ids.shape[-1]
        negative_ids = pipeline.tokenizer(negative_prompt, truncation=False, padding="max_length", max_length=shape_max_length, return_tensors="pt").input_ids.to(device)

    else:
        negative_ids = pipeline.tokenizer(negative_prompt, return_tensors="pt", truncation=False).input_ids.to(device)
        shape_max_length = negative_ids.shape[-1]
        input_ids = pipeline.tokenizer(prompt, return_tensors="pt", truncation=False, padding="max_length", max_length=shape_max_length).input_ids.to(device)

    concat_embeds = []
    neg_embeds = []
    for i in range(0, shape_max_length, max_length):
        concat_embeds.append(pipeline.text_encoder(input_ids[:, i: i + max_length])[0])
        neg_embeds.append(pipeline.text_encoder(negative_ids[:, i: i + max_length])[0])

    return torch.cat(concat_embeds, dim=1), torch.cat(neg_embeds, dim=1)

#prompt = (22 + 10) * prompt
#negative_prompt = (22 + 10) * negative_prompt

#print("Our inputs ", prompt, negative_prompt, len(prompt.split(" ")), len(negative_prompt.split(" ")))
# For long prompt END

#prompt = (22 + random.randint(1, 10)) * "a photo of an astronaut riding a horse on mars"
#negative_prompt = (22 + random.randint(1, 10)) * "some negative texts"

#print("Our inputs ", prompt, negative_prompt, len(prompt.split(" ")), len(negative_prompt.split(" ")))
#prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, device)

# For long prompt
#compel_proc = Compel(tokenizer=pipe.tokenizer, text_encoder=pipe.text_encoder, truncate_long_prompts=False)
#prompt_embeds = compel_proc(prompt)
#negative_prompt_embeds = compel_proc(negative_prompt)
#[prompt_embeds, negative_prompt_embeds] = compel_proc.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_prompt_embeds])


#generator = torch.Generator(device=device).manual_seed(1024)
for i in range(imgs_count):
    resultArr['imgs'] = {}
    resultArr['imgs'][i] = {}
    FileName = str(int(time.time()))+str(".png")
    resultArr['imgs'][i]['FileName'] = FileName
    FilePath = imgs_dir+"/"+FileName
    resultArr['imgs'][i]['FilePath'] = FilePath
    if len(prompt) > max_length_prompt or len(negative_prompt) > max_length_prompt:
        prompt_embeds, negative_prompt_embeds = get_pipeline_embeds(pipe, prompt, negative_prompt, device)
        image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, width=img_width, height=img_height, num_inference_steps=img_num_inference_steps, guidance_scale=img_guidance_scale).images[0]
    else:
        image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=img_width, height=img_height, num_inference_steps=img_num_inference_steps, guidance_scale=img_guidance_scale).images[0]

    #image = pipe(prompt=prompt, negative_prompt=negative_prompt, width=img_width, height=img_height, num_inference_steps=img_num_inference_steps, guidance_scale=img_guidance_scale).images[0]
    #image = pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, width=img_width, height=img_height, num_inference_steps=img_num_inference_steps, guidance_scale=img_guidance_scale).images[0]
    image.save(FilePath)

print(json.dumps(resultArr))