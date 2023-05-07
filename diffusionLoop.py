import numpy
import torch
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel #TODO: so far haven't found a good replacement, leaving for now
from huggingface_hub import notebook_login
from torch import autocast
from tqdm.auto import tqdm

import argparse as ap
import tifffile as tf

parser = ap.ArgumentParser(
                    prog='diffusionLoop',
                    description='pass in a the path and filename slug to your save state, and out pops a tiff ready for the VAE.',
                    epilog='Rock on and pic more!')

parser.add_argument('savestate')
parser.add_argument('-l', '--latent_image', default="")
parser.add_argument('-t', '--text_embedding', default="")
parser.add_argument('-i', '--inference_steps', default=20, type=int)
parser.add_argument('-g', '--guidance_scale', default=7, type=int)
parser.add_argument('-m', '--model_path', default="runwayml/stable-diffusion-v1-5")

# args = parser.parse_args
args = vars(parser.parse_args())

torch_device = "cuda"

num_inference_steps = args['inference_steps']
guidance_scale = args['guidance_scale']               
batch_size = 1

unet = UNet2DConditionModel.from_pretrained(args['model_path'], subfolder="unet", torch_dtype=torch.float16)
unet = unet.to(torch_device, torch.float16)

#TODO: get other types of schedulers and a good way to pick them:
scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler.set_timesteps(num_inference_steps)

latent_image = args["latent_image"]

if latent_image == "":
    latent_image = args["savestate"] + "_latent.tiff"

text_embedding = args["text_embedding"]

if text_embedding == "":
    text_embedding = args["savestate"] + "_prompt.tiff"

#encoded = tf.imread(args["savestate"] + "_state1.tiff")
encoded = tf.imread(latent_image)
encoded2 = torch.from_numpy(numpy.expand_dims(encoded, 0))

#tcoded = tf.imread(args["savestate"] + "_textstate1.tiff")
tcoded = tf.imread(args["text_embedding"])
text_embeddings = torch.from_numpy(tcoded)

with torch.no_grad():
    text_embeddings = text_embeddings.to(torch_device, torch.float16)

latents = encoded2.to(torch_device, torch.float16)

# Loop
with autocast("cuda"):
    for i, t in tqdm(enumerate(scheduler.timesteps)):

        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]

        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        # predict the noise
        with torch.no_grad():
            noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        # guidance step
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        
        #update latents
        latents = scheduler.step(noise_pred, t, latents).prev_sample

# scale and decode the image latents with vae
# my own calc for this:
latent_scale = (0.19503/7.5) * guidance_scale
latent_scale = latent_scale if guidance_scale > 7.5 else 0.18215
latents = (1 / latent_scale) * latents

imagearray = latents[0].cpu().numpy()
tf.imwrite(args["savestate"] + "_unet.tiff", imagearray, dtype='float16')
