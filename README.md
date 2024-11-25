# diffusers/controlnet-canny-sdxl-1.0 Cog model

[![Try a demo on Replicate](https://replicate.com/pnyompen/sd-controlnet-lora/badge)](https://replicate.com/pnyompen/sd-controlnet-lora)

This is an implementation of the [diffusers/controlnet-canny-sdxl-1.0](https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0) as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights:

    cog run script/download-weights

Then, you can run predictions:

    cog predict -i image=@demo.png -i prompt="aerial view, a futuristic research complex in a bright foggy jungle, hard lighting" -i negative_prompt="low quality, bad quality, sketches"

## Example:

Input:

"aerial view, a futuristic research complex in a bright foggy jungle, hard lighting"

![alt text](demo.png)

Output:

![alt text](output.png)
