import contextlib
import logging
import sys

import hydra
from omegaconf import DictConfig

from awesomedemo_test import execute_main

"""
This script is the entry point for running the main demo application.
It uses Hydra for configuration management and sets up the environment
for executing the main function with the provided configuration.
"""


@hydra.main(version_base=None, config_path="conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    # Set the output directory for the Hydra configuration
    out_hydra_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    execute_main(cfg, out_hydra_dir)


if __name__ == "__main__":
    my_app()