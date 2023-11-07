# from flask import Flask, render_template, request, jsonify
# from werkzeug.utils import secure_filename
# import torch
# from PIL import Image
# from diffusers import DiffusionPipeline, StableDiffusionXLImg2ImgPipeline
#
# app = Flask(__name__)
#
# # 클라이언트에서 받은 데이터로 변수 설정
# device = "cuda:1" if torch.cuda.is_available() else "cpu"
#
# # 모델 및 파이프라인 설정
# prj_path = "Leekp/toonmaker5"  # 기본 prj_path 설정
# model = "stabilityai/stable-diffusion-xl-base-1.0"
#
# pipe = DiffusionPipeline.from_pretrained(
#     model,
#     torch_dtype=torch.float32,
#     use_safetensors=True
# )
# pipe.to(device)
#
# refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
#     "stabilityai/stable-diffusion-xl-refiner-1.0",
#     torch_dtype=torch.float16,
# )
# refiner.to(device)
#
# def resize(value, img):
#     img = Image.open(img)
#     img = img.resize((value, value))
#     return img
#
# @app.route('/file_upload', methods=['POST'])
# def file_upload():
#     if 'file' not in request.files:
#         return jsonify({"message": "No file part"})
#
#     f = request.files['file']
#     if f.filename == '':
#         return jsonify({"message": "No selected file"})
#
#     # 이미지 업로드 처리
#     if f:
#         filename = secure_filename(f.filename)
#         file_path = f"uploads/{filename}"
#         f.save(file_path)
#
#         # 텍스트 입력 받기
#         text_input = request.form.get("text_input", "")
#
#         # Define your parameters here.
#         prompt = text_input
#         negative_prompt = ""
#         guide = 7.5
#         steps = 10
#         seed_value = 12345
#         num_images_per_prompt = 4
#
#         # Load source image and resize it to required dimensions (768x768).
#         source_image_path_resized = resize(768, file_path)
#
#         for seed in range(num_images_per_prompt):
#             # Set the default data type to float32
#             torch.set_default_dtype(torch.float32)
#
#             # Create the generator with manual seed
#             generator = torch.Generator(device=device).manual_seed(seed + seed_value)
#
#             # Run the model with the loaded image as input.
#             image_output = pipe(prompt=prompt,
#                                 negative_prompt=negative_prompt,
#                                 guidance_scale=guide,
#                                 num_inference_steps=steps,
#                                 generator=generator
#                                 ).images[0]
#
#             # Save output image to a file.
#             output_image_path = f"results/output_{seed}.png"
#             image_output.save(output_image_path)
#
#         torch.cuda.empty_cache()  # Clear unused memory from cache
#         return jsonify({"message": "이미지 처리 및 저장 완료"})
#
#     return render_template('file_upload.html')
#
# if __name__ == '__main__':
#     app.run()