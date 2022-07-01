import os
os.system("wget https://huggingface.co/akhaliq/lama/resolve/main/best.ckpt")
os.system("pip install imageio")
import cv2
import paddlehub as hub
import gradio as gr
import torch
from PIL import Image, ImageOps
import numpy as np
import imageio
os.mkdir("data")
os.rename("best.ckpt", "models/best.ckpt")
os.mkdir("dataout")
model = hub.Module(name='U2Net')
def infer(img,option):
  print(type(img))
  print(type(img["image"]))
  print(type(img["mask"]))
  imageio.imwrite("./data/data.png", img["image"])
  if option == "automatic (U2net)":
      result = model.Segmentation(
          images=[cv2.cvtColor(img["image"], cv2.COLOR_RGB2BGR)],
          paths=None,
          batch_size=1,
          input_size=320,
          output_dir='output',
          visualization=True)
      im = Image.fromarray(result[0]['mask'])
      im.save("./data/data_mask.png")
  else:
      imageio.imwrite("./data/data_mask.png", img["mask"])
  os.system('python predict.py model.path=/home/user/app/ indir=/home/user/app/data/ outdir=/home/user/app/dataout/ device=cpu')
  return "./dataout/data_mask.png",mask
  
inputs = [gr.Image(tool="sketch", label="Input",type="numpy"),gr.inputs.Radio(choices=["automatic (U2net)","manual"], type="value", default="manual", label="Masking option")]
outputs = [gr.outputs.Image(type="file",label="output"),gr.outputs.Image(type="pil",label="Mask")]
title = "LaMa Image Inpainting"
description = "Gradio demo for LaMa: Resolution-robust Large Mask Inpainting with Fourier Convolutions. To use it, simply upload your image, or click one of the examples to load them. Read more at the links below. Masks are generated by U^2net"
article = "<p style='text-align: center'><a href='https://arxiv.org/abs/2109.07161' target='_blank'>Resolution-robust Large Mask Inpainting with Fourier Convolutions</a> | <a href='https://github.com/saic-mdal/lama' target='_blank'>Github Repo</a></p>"
examples = [
  ['person512.png',"automatic (U2net)"],
  ['person512.png',"manual"]
]
gr.Interface(infer, inputs, outputs, title=title, description=description, article=article, examples=examples,cache_examples=False).launch()