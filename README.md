# StableDiffusion-By-Parts
# WARNING USE AT YOUR OWN RISK PROTOTYPE SOFTWARE NOT FOR PEOPLE WHO DON'T UNDERSTAND CODE
nicer version will come later.

Slice and dice the Stable Diffusion pipeline, saving to a TIFF file (what I'm calling "TIFFSD" format) in between sections.

This is prototype code, so it has hard-coded values and requires manual love to use it for anything other than testing. A Huggingface account is required at this stage, unless you change that part of the code yourself, which is fine. 

Example Run:

`python textToLatent.py "a highly detailed photo of a cat" -o "c:\test\cat1" -w 768 -he 512 -i 30` -- This will output two files to the directory "test" with "cat1" prepended.
Those files will look something like this: 

![savestate1](https://user-images.githubusercontent.com/7604556/236593217-576cf858-3e40-44f6-95a9-f68023ceed06.png)

and this: 

![savetext](https://user-images.githubusercontent.com/7604556/236593243-67383773-5c82-49fc-894b-9ce73e730f65.png)

The first is the ramdomized latent space "image", and the second is the text "embedding". The adventerous can use the second TIFFSD file as a way to save a prompt, which can then be used with any latent space "image" in the diffusion loop below.


`python diffusionLoop.py "c:\test\Cat1" -i 30 -g 7` -- This will take those two files and then run the diffusion loop using their contents, 30 iterations CFG 7

The output of that will look something like this: 

![savestate2](https://user-images.githubusercontent.com/7604556/236593303-a6e362b7-15bc-4c5c-b33f-6f240e3a2f21.png)

`python vaeDecompress.py "c:\test\Cat1_state2.tiff" -o "c:\test\Cat1_84.png" -v 'vae84' `-- This will upscale the latent "image" from the previous output to full size using a local VAE model in the "vae84" folder in the directory where this script is located. Can also be an absolute/relative path. Or a ref to a Huggingface Diffusers model. The output will be your actual image. If you've read this far, you deserve a cat pic:


![savestate3](https://user-images.githubusercontent.com/7604556/236593446-4a6daa04-a649-4184-9fbe-f444a6cbf03e.png)

As written at the moment you'll get a 16 bit per RGBA channel per pixel .tiff file, as well as a .png The image above is the .tiff file, converted to .png in image editing software.

There's also a file in here called "vaeCompress.py". This can take a regular-size image (size, currently, limited to what will fit at once in the vram of your graphics card, or in your ram/cpu if you dare) and turn it into a latent space "image". Basically what the image2image process does, except the code (currently) doesn't add any noise to the image. I originally started this to explore VAE-only compression of images, but hey sometimes there's scope creep.

TODO:
-----
1.) remove dependencies on diffusers

2.) more/better input params

3.) add support for .ckpt files and other similar bits

4.) frickin' vae tiling

5.) Marry the text embedding to the latents after running "textToLatent.py", as an option.
