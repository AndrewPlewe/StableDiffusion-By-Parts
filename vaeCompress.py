import torch
from diffusers import AutoencoderKL
from huggingface_hub import notebook_login
from pathlib import Path
from PIL import Image
from torchvision import transforms as tfms
import numpy as np
import argparse as ap
import tifffile as tf

parser = ap.ArgumentParser(
                    prog='vaeCompress',
                    description='this takes an image and compresses it and stores the compressed version as a npy file.',
                    epilog='Rock on and pic more!')

parser.add_argument('filename')
parser.add_argument('-o', '--outputFilename')
parser.add_argument('-v', '--vaeFileLocation', default='stabilityai/stable-diffusion-2-1')
args = vars(parser.parse_args())

if not (Path.home()/'.huggingface'/'token').exists(): notebook_login()
torch_device = "cuda" 
vae = AutoencoderKL.from_pretrained(args["vaeFileLocation"], subfolder="vae", torch_dtype=torch.float16)
vae = vae.to(torch_device, torch.float16)

input_image = Image.open(args["filename"])

with torch.no_grad():
    latent = vae.encode(tfms.ToTensor()(input_image).unsqueeze(0).to(torch_device, torch.float16)*2-1) # Note scaling
    
encoded = 0.19503 * latent.latent_dist.sample()

imagearray = encoded[0].cpu().numpy()
tf.imwrite(args["outputFilename"], imagearray, dtype='float16')