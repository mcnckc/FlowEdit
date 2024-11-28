[![Zero-Shot Image Editing](https://img.shields.io/badge/zero%20shot-image%20editing-Green)]([https://github.com/topics/video-editing](https://github.com/topics/text-guided-image-editing))
[![Python](https://img.shields.io/badge/python-3.8+-blue?python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org/downloads/release/python-38/)
![PyTorch](https://img.shields.io/badge/torch-2.0.0-red?PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

# FlowEdit

[Project](https://matankleiner.github.io/flowedit/) | [Arxiv](https://arxiv.org/abs/) 

### Official Pytorch implementation of the paper: "FlowEdit: Inversion-Free Text-Based Editing Using Pre-Trained Flow Models"

![](imgs/teaser.png)

## Installation
1. Clone the repository

2. Install the required dependencies: `pip install -r requirements.txt`
	* Tested with CUDA version 12.0 and diffusers 0.21.2

## Usage

Upload images to a data folder. 

Create an edits file that include the path to each image, a source prompt, a traget promt and a target code. The target code summarize the changes between the source and target prompts. <br>
See `edits.yaml` for example.

Create an experiment file. This file include all the hyperparamaters needed for ruuning FlowEdit, such as `n_max`, `n_min`. <br>
See `FLUX_exp.yaml` for FLUX usage example and `SD3_exp.yaml` for Stable Diffusion 3 usage example. <br>
See our paper for discussion on the effect of different hyperparameters and values we used.

Run `python run_script.py --exp_yaml <path to your experiment yaml>`


## License
This project is licensed under the [MIT License](LICENSE).


### Citation
If you use this code for your research, please cite our paper:

```
```
