# Improved Mean Flows

<p align="center">
  <img src="assets/teaser.png" width="60%">
</p>

This is the official JAX implementation for the paper [Improved Mean Flows: On the Challenges of Fastforward Generative Models](https://arxiv.org/abs/2512.02012). This code is written and tested on TPUs.

## Initialization

Run `install.sh` to install the dependencies (JAX+TPUs).

## Inference

You can quickly verify your setup with our provided checkpoint.
<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<td valign="bottom"></td>
<td valign="bottom" align="center">iMF-B/2</td>
<td valign="bottom" align="center">iMF-M/2</td>
<td valign="bottom" align="center">iMF-L/2</td>
<td valign="bottom" align="center">iMF-XL/2</td>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/iMF-B-2.zip">download</a></td>
<td align="center"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/iMF-M-2.zip">download</a></td>
<td align="center"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/iMF-L-2.zip">download</a></td>
<td align="center"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/iMF-XL-2.zip">download</a></td>
</tr>
<tr><td align="left">FID</td>
<td align="center">3.37/3.39</td>
<td align="center">2.27/2.27</td>
<td align="center">1.85/1.86</td>
<td align="center">1.70/1.72</td>
</tr>
<tr><td align="left">IS</td>
<td align="center">256.0/255.3</td>
<td align="center">260.9/257.7</td>
<td align="center">278.6/276.6</td>
<td align="center">282.0/282.0</td>
</tr>
</tbody></table>

The FID/IS numbers on the left are from our evaluation, and those on the right are reported in the original paper. Note that slight differences may arise due to randomness in evaluation.


#### Sanity Check

1. **Download the checkpoint and FID stats:** 
    - Download the pre-trained checkpoint from the table above.
    - Download the FID stats file from [here](https://huggingface.co/he-vision-group/iMF/blob/main/imagenet_256_fid_stats.npz). Our FID stats is computed on TPU and JAX, which may slightly differ from those computed on GPU and PyTorch.

2. **Unzip the checkpoint:**
    ```bash
    unzip <downloaded_checkpoint.zip> -d <your_ckpt_dir>
    ```
    Replace `<downloaded_checkpoint.zip>` and `<your_ckpt_dir>` with your actual paths.

3. **Set up the config:**
    - Set `load_from` in `configs/eval_config.yml` to the path of `<your_ckpt_dir>`.
    - Set `fid.cache_ref` to the path of the downloaded FID stats file.
    - Set CFG-related parameters for corresponding model.

4. **Launch evaluation:**
    ```bash
    bash scripts/eval.sh JOB_NAME
    ```
    Our default evaluation script generates 50,000 samples using pre-trained iMF-B/2 for FID and IS evaluation. The expected FID and IS is 3.37 and 256.0 for this checkpoint. (compared to 3.39 and 255.3 reported in the original paper)

## Data Preparation

Before training, you need to prepare the ImageNet dataset and compute latent representations:

#### 1. Download ImageNet

Download the [ImageNet](http://image-net.org/download) dataset and extract it to your desired location. The dataset should have the following structure:
```
imagenet/
├── train/
│   ├── n01440764/
│   ├── n01443537/
│   └── ...
└── val/
    ├── n01440764/
    ├── n01443537/
    └── ...
```

#### 2. Configure Data Paths

Update the data paths in `scripts/prepare_data.sh`:

```bash
IMAGENET_ROOT="YOUR_IMGNET_ROOT"
OUTPUT_DIR="YOUR_OUTPUT_DIR"
LOG_DIR="YOUR_LOG_DIR"
```

#### 3. Launch Data Preparation

Run the data preparation script to compute latent representations:

```bash
IMAGE_SIZE=256 COMPUTE_LATENT=True bash ./scripts/prepare_data.sh
```

**Parameters:**
- `IMAGE_SIZE`: Image size for processing (256, 512, or 1024). Latent sizes will be 32x32, 64x64, or 128x128 respectively.
- `COMPUTE_LATENT`: Whether to compute and save the latent dataset (True/False)
- `COMPUTE_FID`: Whether to compute FID statistics (True/False)

The script will:
- Encode ImageNet images to latent representations using a VAE model
- Save the latent dataset to `OUTPUT_DIR/`
- Compute FID statistics and save to `OUTPUT_DIR/imagenet_{IMAGE_SIZE}_fid_stats.npz`
- Log progress to `LOG_DIR/$USER/`

### Configuration Setup

After data preparation, you need to configure your FID cache reference in the config files:

#### 1. Update Config Files

Edit your config file (e.g., `configs/train_config.yml` and `configs/eval_config.yml`) and replace the placeholder values:

```yaml
dataset:
    root: YOUR_DATA_ROOT  # Path to your prepared latent dataset, only for training config

fid:
    cache_ref: YOUR_FID_CACHE_REF  # Path to your FID statistics file
```

#### 2. Available Config Files

- `configs/train_config.yml` - Configuration for iMF-B/2 model training (recommended)
- `configs/eval_config.yml` - Configuration for evaluation
- `configs/default.py` - Default configuration (Python format, used as base)

**Configuration Hierarchy:**
The system uses a hierarchical approach where `train_config.yml` and `eval_config.yml` override specific parameters from `default.py`. This allows you to customize only the parameters you need while keeping sensible defaults.
Make sure to update both the dataset root path and the FID cache reference path according to your data preparation output.

## Training

Run the following commands to launch training:
```bash
bash scripts/launch.sh JOB_NAME
```

**Note:** Update the environment variables in `scripts/train.sh` before running:
- `DATA_ROOT`: Path to your prepared data directory
- `LOG_DIR`: Path where to save training logs

#### Config System

The training system uses two config files:

- **`configs/default.py`** - Base configuration with all default hyperparameters
- **`configs/train_config.yml`** - Model-specific overrides for iMF-B/2 training

The system merges these files, allowing you to customize only the parameters you need.

#### Customizing Training

To create a custom experiment:

1. **Create a new config file** (e.g., `configs/my_exp_config.yml`)
2. **Update the launch script** to use your config:
   ```bash
   # In launch.sh, change the config line to:
   --config=configs/load_config.py:my_exp
   ```

**Example custom config:**
```yaml
training:
    num_epochs: 80                  # Train for fewer epochs

method:
    model_str: MiT_B_2               # Use iMF-B/2 model
    cfg_beta: 1.0                     # Set cfg distribution
```

for more details on configuration options, refer to `configs/default.py` and `configs/train_config.yml`.

## License

This repo is under the MIT license. See [LICENSE](./LICENSE) for details.

## Citation

If you find this work useful in your research, please consider citing our paper :)

```bib
@article{improvedmeanflows,
  title={Improved Mean Flows: On the Challenges of Fastforward Generative Models},
  author={Geng, Zhengyang and Lu, Yiyang and Wu, Zongze and Shechtman, Eli and Kolter, J Zico and He, Kaiming},
  journal={arXiv preprint arXiv:2512.02012},
  year={2025}
}
```

## Contributors

This repository is a collaborative effort by Kaiming He, Hanhong Zhao, Yiyang Lu, and Zhengyang Geng, developed in support of several research projects. We sincerely thank Qiao Sun, Zhicheng Jiang, Xianbang Wang for their help in building the codebase and infrastructure.

## Acknowledgement

We gratefully acknowledge the Google TPU Research Cloud (TRC) for granting TPU access.
We hope this work will serve as a useful resource for the open-source community.
