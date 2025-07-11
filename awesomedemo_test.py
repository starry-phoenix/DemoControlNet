# %%
from share import *

import numpy as np
import matplotlib.pyplot as plt
import random

import config
import einops
import torch

from annotator.util import resize_image, HWC3
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from pytorch_lightning import seed_everything

from main_process import DemoNetModel
from utils_config import ConfigSort

# %% Create model setup
# Initialize the Canny edge detector



# %% # Define the parameters for the model

# prompt = "mri brain scan"
# num_samples = 1
# image_resolution = 512
# strength = 1.0
# guess_mode = False
# low_threshold = 50
# high_threshold = 100
# ddim_steps = 10
# scale = 9.0
# seed = 1
# eta = 0.0
# a_prompt = "good quality"  # 'best quality, extremely detailed'
# n_prompt = "animal, drawing, painting, vivid colors, longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"


# %%

def create_init_model():
    model = create_model("./models/cldm_v15.yaml").cpu()
    model.load_state_dict(
        load_state_dict("./models/control_sd15_canny.pth", location="cuda")
    )
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)
    return model, ddim_sampler

def execute_main(config_raw, output_path=None):
    """Main function to execute the demo."""

    config_data = ConfigSort.getconfig_dataclass(
            config_raw
        )
    # creates a model from the config file and loads the state dict from the specified path
    model, ddim_sampler = create_init_model()

    input_image_path = "test_imgs//mri_brain.jpg"
    output_path = output_path

    # create process_model instance
    process_model = DemoNetModel(model, ddim_sampler, input_image_path, output_path=output_path)

    input_image = process_model.get_input_image()
    
    # Process the input image and generate results
    result = process_model.process(
        input_image=input_image,
        prompt=config_data.prompt,
        a_prompt=config_data.a_prompt,
        n_prompt=config_data.n_prompt,
        num_samples=config_data.num_samples,
        image_resolution=config_data.image_resolution,
        ddim_steps=config_data.ddim_steps,
        guess_mode=config_data.guess_mode,
        strength=config_data.strength,
        scale=config_data.scale,
        seed=config_data.seed,
        eta=config_data.eta,
        low_threshold=config_data.low_threshold,
        high_threshold=config_data.high_threshold
    )
    
    # Display the input image, detected map, and the result
    process_model.plot_input_image(input_image)
    process_model.plot_result(result)
    process_model.plot_compare_results_input(input_image, config_data.image_resolution, result, mode=config_data.mode)
