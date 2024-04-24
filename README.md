### Dataset Access
Access the dataset used to replicate the results presented in the paper at Hugging Face:
[Latent DNA Diffusion Dataset](https://huggingface.co/datasets/Zehui127127/latent-dna-diffusion)

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Zehui127/Latent-DNA-Diffusion.git
    ```
2. Install the required packages: diffusers, accelerate
   ```sh
   pip install torch torchvision diffusers accelerate einops easydict pytorch_lightning
   ```
### Code Usage

#### Download pre-trained models

A pre-trained vae model and unet model can be downloaded from the following links:

https://zenodo.org/records/11061611

#### Generate DNA sequences using the trained diffusion model

Set the path to the pretrained vae model and unet model in the following command. The number of sequences to generate can be set using the sequence_num argument in ```src/configs/generate.yaml```.

```
CUDA_VISIBLE_DEVICES="0" accelerate launch --main_process_port 12903 --multi_gpu main.py --model generate --gen_vae_path="" --gen_unet_path=""
```


#### Train the diffusion model

Set the path to the dataset in the config file ```src/configs/un_unet.yaml``` data_path field.
Set the path to the pretrained vae model in the config file ```src/configs/un_unet.yaml``` vae_path field.
```
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --main_process_port 12903 --multi_gpu main.py --model un_unet
```

#### Train the vae model

Set the path to the dataset in the config file
```
CUDA_VISIBLE_DEVICES="0,1" accelerate launch --main_process_port 12903 --multi_gpu main.py --model vanilla_vae
```



# How to Cite This Work

If you use this project or dataset in your research, please cite it as follows:

```bibtex
@article{li2023latent,
  title={Latent Diffusion Model for DNA Sequence Generation},
  author={Li, Zehui and Ni, Yuhao and Huygelen, Tim August B and Das, Akashaditya and Xia, Guoxuan and Stan, Guy-Bart and Zhao, Yiren},
  journal={arXiv preprint arXiv:2310.06150},
  year={2023}
}

@article{li2024discdiff,
  title={DiscDiff: Latent Diffusion Model for DNA Sequence Generation},
  author={Li, Zehui and Ni, Yuhao and Beardall, William AV and Xia, Guoxuan and Das, Akashaditya and Huygelen, Tim August B and Stan, Guy-Bart and Zhao, Yiren},
  journal={arXiv preprint arXiv:2402.06079},
  year={2024}
}
