# StableDiffusion-By-Parts
Slice and dice the Stable Diffusion pipeline, saving to a TIFF file (what I'm calling "TIFFSD") in between sections.

This is prototype code, so it has hard-coded values and requires manual love to use it for anything other than testing. A Huggingface account is required at this stage, unless you change that part of the code yourself, which is fine. 

Example Run:

python textToLatent.py "two deer dancing by a lake" -o "c:\test\deer3" -w 768 -he 512 -i 4 -- This will output two files to the directory "test" with "deer3" prepended.

python diffusionLoop.py "d:\test\Deer3" -i 30 -g 7 -- This will take those two files and then run the diffusion loop using their contents

python vaeDecompress.py "d:\test\Deer3_state2.tiff" -o "d:\test\Deer3_84.png" -v 'vae84' -- This will upscale the latent "image" from the previous output to full size


TODO:
-----
1.) remove dependencies on diffusers

2.) more/better input params

3.) add support for .ckpt files and other similar bits

4.) frickin' vae tiling

5.) Marry the text embedding to the latents after running "textToLatent.py", as an option.
