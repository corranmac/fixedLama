import os
os.system("gdown https://drive.google.com/uc?id=1-95IOJ-2y9BtmABiffIwndPqNZD_gLnV")
os.system("unzip big-lama.zip")
import cv2
import paddlehub as hub
import gradio as gr
import torch
from PIL import Image
import numpy as np
os.mkdir("data")
os.mkdir("dataout")
torch.hub.download_url_to_file('https://images.pexels.com/photos/103123/pexels-photo-103123.jpeg', 'person.jpeg')
model = hub.Module(name='U2Net')
def infer(img):
  img.save("./data/data.png")
  os.system("ls data")
  result = model.Segmentation(
      images=[cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)],
      paths=None,
      batch_size=1,
      input_size=320,
      output_dir='output',
      visualization=True)
  im = Image.fromarray(result[0]['mask'])
  im.save("./data/data_mask.png")
  os.system("ls data")
  os.system('python predict.py model.path=/home/user/app/big-lama/ indir=/home/user/app/data/ outdir=/home/user/app/dataout/ device=cpu')
  os.system("ls dataout")
  return "./dataout/data_mask.png"
inputs = gr.inputs.Image(type='pil', label="Original Image")
outputs = gr.outputs.Image(type="file",label="output")
title = "LaMa Image Inpainting"
description = "Gradio demo for LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below."
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2109.07161' target='_blank'>Resolution-robust Large Mask Inpainting with Fourier Convolutions</a> | <a href='https://github.com/saic-mdal/lama' target='_blank'>Github Repo</a></p>"
examples = [
  ['person.jpeg']
]
gr.Interface(infer, inputs, outputs, title=title, description=description, article=article, examples=examples).launch()