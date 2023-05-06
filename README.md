# StableDiffusion-By-Parts
# WARNING USE AT YOUR OWN RISK PROTOTYPE SOFTWARE NOT FOR PEOPLE WHO DON'T UNDERSTAND CODE
nicer version will come later.

# Update (05/06/2023): 
Added an "Example Artifacts" folder with TIFFSD format .tiff files. Stage 1 = creating the latents and the prompt, Stage 2 = after diffusion loop, before VAE, Stage 3 = "raw" .tiff containing the 16 bit per color channel output of the VAE, before conversion to unit8 as a png file. This can also be 32 bits, it comes directly from the VAE so a 32 bit VAE will produce a 32 bit output.

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

Quick and dirty tech specs for TIFFSD format:

TIFFs can save pixels using 16 bit and 32 bit floating point numbers for each RGBA channel, along with multiple "pages" of different dimensions. This is awesome, and I'm doing that here. Not entirely sure of the channel order, but generally it doesn't matter. IF you edit one of these (like, say, trimming off parts of a latent space "image" to do manual tiling), BE SURE YOU SAVE IN 16 OR 32 BIT FLOATING POINT PER RGBA CHANNEL PER PIXEL FORMAT. The things I've tried usually only save integers. This will ruin your day. So, abide by the rules of the TIFF and you'll be golden. Basically the data saved is straight off the GPU in the format Stable Diffusion likes, I only pull off a "wrapper" dimension that isn't really necessary (see the code for details). You do have to add that back on, though, when you re-load the .tiff



TODO:
-----
1.) remove dependencies on diffusers

2.) more/better input params

3.) add support for .ckpt files and other similar bits

4.) frickin' vae tiling

5.) Marry the text embedding to the latents after running "textToLatent.py", as an option.
