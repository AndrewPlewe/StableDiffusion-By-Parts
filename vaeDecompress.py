import torch
from diffusers import AutoencoderKL
from huggingface_hub import notebook_login
from pathlib import Path
from PIL import Image
import numpy as np
import argparse as ap
import tifffile as tf

parser = ap.ArgumentParser(
                    prog='vaeDecompress',
                    description='this takes a vaeCompressed image and makes it into a vaeDecompressed image.',
                    epilog='Rock on and pic more!')

parser.add_argument('filename')
parser.add_argument('-o', '--outputFilename')
parser.add_argument('-v', '--vaeFileLocation', default='stabilityai/stable-diffusion-2-1')

# args = parser.parse_args
args = vars(parser.parse_args())
if not (Path.home()/'.huggingface'/'token').exists(): notebook_login()
torch_device = "cuda" 
vae = AutoencoderKL.from_pretrained(args["vaeFileLocation"], subfolder="vae", torch_dtype=torch.float16)
vae = vae.to(torch_device, torch.float16)

encoded = tf.imread(args["filename"])
latents = torch.from_numpy(np.expand_dims(encoded, 0))

vae = vae.to(torch_device, torch.float16)

with torch.no_grad():
    image = vae.decode(latents.cuda()).sample

# Display
image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()

tf.imwrite(args["outputFilename"] + ".tiff", image[0], dtype='float16')

images = (image * 255).round().astype("uint8") #average rounding error
pil_images = [Image.fromarray(image) for image in images]

#decoded = vaeDecompress(encoded2.cuda())[0]
pil_images[0].save(args["outputFilename"], bitmap_format='png')