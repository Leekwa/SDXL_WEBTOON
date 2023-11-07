from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
import torch
import os
from PIL import Image
from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
import sys
import requests
import io
from base64 import encodebytes
from PIL import Image
from flask import jsonify



app = Flask(__name__)



# 클라이언트에서 받은 데이터로 변수 설정
# gpu_count = torch.cuda.device_count()
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
device2 = "cuda:1" if torch.cuda.is_available() else "cpu"



print(torch.cuda.device_count())
print(device2)




prj_path = "Leekp/toonmaker5"  # 기본 prj_path 설정
# prj_path = "Leekp/toonmaker7"  # 기본 prj_path 설정

model = "stabilityai/stable-diffusion-xl-base-1.0"

pipe = DiffusionPipeline.from_pretrained(
    model,
    torch_dtype=torch.float16,
    use_safetensors=True
)
pipe.to(device2)
pipe.load_lora_weights(prj_path, weight_name="pytorch_lora_weights.safetensors")

refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    torch_dtype=torch.float16,
)
refiner.to(device2)





def get_response_image(image_path):
    pil_img = Image.open(image_path, mode='r') # reads the PIL image
    byte_arr = io.BytesIO()
    pil_img.save(byte_arr, format='PNG') # convert the PIL image to byte array
    encoded_img = encodebytes(byte_arr.getvalue()).decode('ascii') # encode as base64
    return encoded_img

@app.route('/')
def upload():

    return render_template('file_upload.html')


@app.route('/file_upload', methods=['POST'])
def file_upload():
    text_input = request.form.get("text_input")
    if not text_input:
        return jsonify({"message": "No text input provided"})

    prompt = text_input.strip()
    negative_prompt = "text, watermark, low-quality, signature, moiré pattern, downsampling, aliasing, distorted, blurry, glossy, blur, jpeg artifacts, compression artifacts, poorly drawn, low-resolution, bad, distortion, twisted, excessive, exaggerated pose, exaggerated limbs, grainy, symmetrical, duplicate, error, pattern, beginner, pixelated, fake, hyper, glitch, overexposed, high-contrast, bad-contrast"
    guide = 7.5
    steps = 10
    seed_value = 12345
    num_images_per_prompt = 4
    strength = 0.3

    all_encoded_images_list = []

    for seed in range(num_images_per_prompt):
        torch.set_default_dtype(torch.float32)
        generator = torch.Generator(device=device2).manual_seed(seed + seed_value)

        # Generate initial image using pipe object.
        image_initial = pipe(prompt=prompt, generator=generator).images[0]

        # Refine the generated image using refiner object.
        image_output = refiner(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guide,
            num_inference_steps=steps,
            generator=generator,
            strength=strength,
            image=image_initial).images[0]  # Add the initial image as an argument.

        output_image_path = f"/home/metaai/PycharmProjects/SDXL_FLASK/sdsd/static/results_{seed}.png"
        image_output.save(output_image_path)

        encoded_image_list = [get_response_image(output_image_path)]

        all_encoded_images_list.append(encoded_image_list)


    torch.cuda.empty_cache()

    return jsonify(all_encoded_images_list)







