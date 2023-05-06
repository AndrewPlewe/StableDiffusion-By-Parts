# StableDiffusion-By-Parts
Slice and dice the Stable Diffusion pipeline, saving to a TIFF file (what I'm calling "TIFFSD") in between sections.

This is prototype code, so it has hard-coded values and requires manual love to use it for anything other than testing. A Huggingface account is required at this stage, unless you change that part of the code yourself, which is fine. 

Example Run:

`python textToLatent.py "two deer dancing by a lake" -o "c:\test\deer3" -w 768 -he 512 -i 4` -- This will output two files to the directory "test" with "deer3" prepended.
Those files will look something like this: 

![savestate1](https://user-images.githubusercontent.com/7604556/236593217-576cf858-3e40-44f6-95a9-f68023ceed06.png)

and this: 

![savetext](https://user-images.githubusercontent.com/7604556/236593243-67383773-5c82-49fc-894b-9ce73e730f65.png)

The first is the ramdomized latent space "image", and the second is the text "embedding".


`python diffusionLoop.py "d:\test\Deer3" -i 30 -g 7` -- This will take those two files and then run the diffusion loop using their contents, 30 iterations CFG 7

The output of that will look something like this: 

![savestate2](https://user-images.githubusercontent.com/7604556/236593303-a6e362b7-15bc-4c5c-b33f-6f240e3a2f21.png)

`python vaeDecompress.py "d:\test\Deer3_state2.tiff" -o "d:\test\Deer3_84.png" -v 'vae84' `-- This will upscale the latent "image" from the previous output to full size using a local VAE model in the "vae84" folder in the directory where this script is located. Can also be an absolute/relative path. The output will be your actual image.


TODO:
-----
1.) remove dependencies on diffusers

2.) more/better input params

3.) add support for .ckpt files and other similar bits

4.) frickin' vae tiling

5.) Marry the text embedding to the latents after running "textToLatent.py", as an option.
