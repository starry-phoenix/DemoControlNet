from share import *

import os

import numpy as np
import matplotlib.pyplot as plt
import random
import imageio
import config
import einops
import torch
from annotator.canny import CannyDetector

from annotator.util import resize_image, HWC3
from pytorch_lightning import seed_everything
from skimage import feature, filters
from skimage.color import rgb2gray
from utils_postprocess import (take_luminance_from_first_chroma_from_second)

class DemoNetModel:
    def __init__(self, model, ddim_sampler, input_image_path=None, model_config=None, output_path=None):
        self.model = model
        self.ddim_sampler = ddim_sampler
        self.model_config = model_config
        self.output_path = output_path
        self.apply_canny = CannyDetector()
        self.input_image_path = input_image_path
    
    def get_input_image(self):
        return imageio.imread(self.input_image_path)

    def choose_input_image(self, input_image, condition_type = "canny_wthresholding"):
        if condition_type == "canny_wthresholding":
            global low_threshold, high_threshold, detected_map
            low_threshold = 50
            high_threshold = 100
            detected_map = self.apply_canny(input_image, low_threshold, high_threshold)
        elif condition_type == "canny":
            detected_map = feature.canny(rgb2gray(input_image), sigma=2).astype(np.float32)
        elif condition_type == "roberts":
            detected_map = filters.roberts(rgb2gray(input_image))
        elif condition_type == "sobel":
            detected_map = filters.sobel(rgb2gray(input_image))
        elif condition_type == "hessian":
            detected_map = filters.hessian(rgb2gray(input_image), range(1, 10))

        detected_map = np.clip(detected_map.astype(np.float32) * 255, 0, 255).astype(np.uint8)

        detected_map = HWC3(detected_map)

        return detected_map

    def plot_input_image(self, input_image):
        """ """
        print(input_image.shape)
        plt.imshow(input_image)
        detected_map = self.choose_input_image(input_image)
        # post-processing
        plt.imshow(255 - detected_map)
        # plt.imshow(detected_map)
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path, "input_image.png"))
        plt.show()
    
    def plot_compare_results_input(self, input_image, image_resolution, result, mode="lab"):
        index = -1
        test = take_luminance_from_first_chroma_from_second(
            resize_image(HWC3(input_image), image_resolution), result[index], mode=mode
        )
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].imshow(input_image)
        axs[1].imshow(result[index])
        axs[2].imshow(test)
        axs[0].axis(False)
        axs[1].axis(False)
        axs[2].axis(False)
        if self.output_path is not None:
            plt.savefig(os.path.join(self.output_path , "compare_results.png"))
        plt.show()

    def plot_result(self, result):
        for res in result:
            plt.imshow(res)
            plt.axis(False)
            if self.output_path is not None:
                plt.savefig(os.path.join(self.output_path , "result.png"))
            plt.show()

    def process(self,
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta,
        low_threshold,
        high_threshold,
    ) -> list:
        """
    Processes the input image using a control model with Canny edge detection.
        Args:
            input_image:
            prompt:
            a_prompt:
            n_prompt:
            num_samples:
            image_resolution:
            ddim_steps:
            guess_mode:
            strength:
            scale:
            seed:
            eta:
            low_threshold:
            high_threshold:

        Returns:
            A list containing the detected edge map and the generated samples.

        """
        with torch.no_grad():
            # resize image
            img = resize_image(HWC3(input_image), image_resolution)
            H, W, C = img.shape
            # apply image filter : canny to detect edges
            detected_map = self.apply_canny(img, low_threshold, high_threshold)
            # convert to HWC format
            detected_map = HWC3(detected_map)

            # convert to tensor
            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            # normalize to [0, 1]
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            # convert to CHW format
            control = einops.rearrange(control, "b h w c -> b c h w").clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
                ],
            }
            un_cond = {
                "c_concat": None if guess_mode else [control],
                "c_crossattn": [self.model.get_learned_conditioning([n_prompt] * num_samples)],
            }
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=True)

            self.model.control_scales = (
                [strength * (0.825 ** float(12 - i)) for i in range(13)]
                if guess_mode
                else ([strength] * 13)
            )  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            samples, intermediates = self.ddim_sampler.sample(
                ddim_steps,
                num_samples,
                shape,
                cond,
                verbose=False,
                eta=eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
            )

            if config.save_memory:
                self.model.low_vram_shift(is_diffusing=False)

            x_samples = self.model.decode_first_stage(samples)
            x_samples = (
                (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
                .cpu()
                .numpy()
                .clip(0, 255)
                .astype(np.uint8)
            )

            results = [x_samples[i] for i in range(num_samples)]
        return [255 - detected_map] + results