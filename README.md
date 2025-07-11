# DemoControlNet
Demonstrates a simple use-case of a ControlNet application

# Getting started
In addition to the changes mentioned in the given task related to the repository, add the script files to the project root directory for the application to function.

Install hydra package:

```console
conda install conda-forge::hydra-core
```

Run the main script as seen below

```console
python run_main_demonet.py
```

Run multirun with 

```console
python run_main_demonet.py -m mode=lab
```

for more information regarding running the main script with

```console
python run_main_demonet.py --help
```

# Configuration setup 
The configuration setup is inside the conf directory with three setups
- prompt: User input prompt
- parameters: model process parameters
- mode: postprocess comparison of lab, yuv, and luminance. 
