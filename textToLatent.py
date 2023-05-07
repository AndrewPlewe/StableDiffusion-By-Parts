import torch
from diffusers import LMSDiscreteScheduler, UNet2DConditionModel
from huggingface_hub import notebook_login
from pathlib import Path
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, logging
import random

import argparse as ap
import tifffile as tf

parser = ap.ArgumentParser(
                    prog='textToLatent',
                    description='Put in a text prompt, and out pops the latent as a TIFF file (TIFFSD) for the latent and a TIFFSD file for the text embedding.',
                    epilog='Rock on and pic more!')

parser.add_argument('prompt')
parser.add_argument('-n', '--negative_prompt',default="")
parser.add_argument('-o', '--outputFilename')
parser.add_argument('-w', '--width', default=512, type=int)
parser.add_argument('-he', '--height', default=512, type=int)
parser.add_argument('-i', '--inference_steps', default=20, type=int)


# args = parser.parse_args
args = vars(parser.parse_args())

if not (Path.home()/'.huggingface'/'token').exists(): notebook_login()

# Supress some unnecessary warnings when loading the CLIPTextModel
logging.set_verbosity_error()

# Set device
torch_device = "cpu"

prompt = [args["prompt"]]
neg_prompt = [args["negative_prompt"]]
height = args["height"]                   # default height of Stable Diffusion
width = args["width"]                 # default width of Stable Diffusion
generator = torch.manual_seed(random.randint(1,99999999999999999))   # Seed generator to create the inital latent noise
batch_size = 1
num_inference_steps = args['inference_steps']

# Load the tokenizer and text encoder to tokenize and encode the text. 
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

# The UNet model for generating the latents, we only really need it for the config to get in_channels:
unet = UNet2DConditionModel.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="unet")

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
scheduler.set_timesteps(num_inference_steps)

text_encoder = text_encoder.to(torch_device)

# # Prep text 
text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
    uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

# Prep latents
latents = torch.randn(
  (batch_size, unet.config.in_channels, height // 8, width // 8),
  generator=generator)

latents = latents.to(torch_device, torch.float16)
latents = latents * scheduler.init_noise_sigma # Scaling (previous versions did latents = latents * self.scheduler.sigmas[0]

imagearray = latents[0].cpu().numpy()
#imagearray = numpy.vstack((imagearray, text_embeddings.cpu()))
tf.imwrite(args["outputFilename"] + "_latent.tiff", imagearray, dtype='float16')
tf.imwrite(args["outputFilename"] + "_prompt.tiff", text_embeddings.numpy())
