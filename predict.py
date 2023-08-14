from cog import BasePredictor, Input, Path
import os
import cv2
import time
import torch
import shutil
import numpy as np
from PIL import Image
from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline, AutoencoderKL
from diffusers.utils import load_image

CONTROL_MODEL = "diffusers/controlnet-canny-sdxl-1.0"
VAE_MODEL = "madebyollin/sdxl-vae-fp16-fix"
MODEL_NAME = "stabilityai/stable-diffusion-xl-base-1.0"
CONTROL_CACHE = "control-cache"
VAE_CACHE = "vae-cache"
MODEL_CACHE = "sdxl-cache"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        t1 = time.time()
        print("Loading controlnet model")
        controlnet = ControlNetModel.from_pretrained(
            CONTROL_CACHE,
            torch_dtype=torch.float16
        )
        print("Loading better VAE")
        better_vae = AutoencoderKL.from_pretrained(
            VAE_CACHE,
            torch_dtype=torch.float16
        )
        print("Loading sdxl")
        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            MODEL_CACHE,
            vae=better_vae,
            controlnet=controlnet,
            use_safetensors=True,
            variant="fp16",
            torch_dtype=torch.float16,
        )
        self.pipe = pipe.to("cuda")
        t2 = time.time()
        print("Setup took: ", t2 - t1)

    def load_image(self, path):
        shutil.copyfile(path, "/tmp/image.png")
        return load_image("/tmp/image.png").convert("RGB")

    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(
            description="Input image for img2img or inpaint mode",
            default=None,
        ),
        prompt: str = Input(
            description="Input prompt",
            default="aerial view, a futuristic research complex in a bright foggy jungle, hard lighting",
        ),
        negative_prompt: str = Input(
            description="Input Negative Prompt",
            default="low quality, bad quality, sketches",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        condition_scale: float = Input(
            description="controlnet conditioning scale for generalization",
            default=0.5,
            ge=0.0,
            le=1.0,
        ),
        seed: int = Input(
            description="Random seed. Set to 0 to randomize the seed", default=0
        ),
    ) -> Path:
        """Run a single prediction on the model"""
        if (seed is None) or (seed <= 0):
            seed = int.from_bytes(os.urandom(2), "big")
        generator = torch.Generator("cuda").manual_seed(seed)
        print(f"Using seed: {seed}")

        # self.pipe.enable_model_cpu_offload()

        image = self.load_image(image)
        image = np.array(image)
        image = cv2.Canny(image, 100, 200)
        image = image[:, :, None]
        image = np.concatenate([image, image, image], axis=2)
        image = Image.fromarray(image)

        images = self.pipe(
            prompt, 
            negative_prompt=negative_prompt, 
            image=image, 
            controlnet_conditioning_scale=condition_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).images

        output_path = f"/tmp/output.png"
        images[0].save(output_path)

        return Path(output_path)
