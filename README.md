  # PyTorch Re-implementation of Improved Mean Flows

<p align="center">
  <img src="assets/teaser.png" width="60%">
</p>

This is the official PyTorch re-implementation for the paper [Improved Mean Flows: On the Challenges of Fastforward Generative Models](https://arxiv.org/abs/2512.02012), which is originally implemented with JAX+TPUs. This code is written and tested on H100 GPUs.

## Initialization

Use `requirements.txt` to install the dependencies (Torch+GPUs).

## Inference

You can quickly verify your setup with our provided checkpoint by reproducing the numbers below:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<td valign="bottom"></td>
<td valign="bottom" align="center">iMF-B/2</td>
<td valign="bottom" align="center">iMF-M/2</td>
<td valign="bottom" align="center">iMF-L/2</td>
<td valign="bottom" align="center" colspan="2">iMF-XL/2</td>
<!-- TABLE BODY -->
<tr><td align="left">pre-trained checkpoint</td>
<td align="center"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/MiT_B_2.pth">download</a></td>
<td align="center"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/MiT_M_2.pth">download</a></td>
<td align="center"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/MiT_L_2.pth">download</a></td>
<td align="center" colspan="2"><a href="https://huggingface.co/he-vision-group/iMF/blob/main/MiT_XL_2.pth">download</a></td>
</tr>
<tr><td align="left">NFE</td>
<td align="center">1</td>
<td align="center">1</td>
<td align="center">1</td>
<td align="center">1</td>
<td align="center">2</td>
</tr>
<tr><td align="left">FID (this repo / original paper)</td>
<td align="center">3.32/3.39</td>
<td align="center">2.26/2.27</td>
<td align="center">1.83/1.86</td>
<td align="center">1.72/1.72</td>
<td align="center">1.54/1.54</td>
</tr>
<tr><td align="left">IS (this repo / original paper)</td>
<td align="center">255.7/255.3</td>
<td align="center">258.3/257.7</td>
<td align="center">275.7/276.6</td>
<td align="center">279.9/282.0</td>
<td align="center">288.9/-</td>
</tr>
</tbody></table>

Note that slight differences may arise due to minor differences in JAX/Torch implementations and hardware.

To generate a batch of images using the iMF-XL/2 model (for visualization only), run:

```bash
python evaluate.py sample \
--ckpt-path /path/to/XL/2/checkpoint.pth \
--workdir ./visualize \
--model MiT_XL_2 \
```

To evaluate the FID of the iMF-B/2 model, run:

```bash
torchrun --nproc-per-node=8 evaluate.py evaluate \
--ckpt-path /path/to/B/2/checkpoint.pth \
--workdir ./b2_fid_output \
--gen-bsz 64
```

To evaluate the FID of the iMF-M/2 model, run:

```bash
torchrun --nproc-per-node=8 evaluate.py evaluate \
--ckpt-path /path/to/M/2/checkpoint.pth \
--workdir ./m2_fid_output \
--model MiT_M_2 \
--cfg-omega 10.5 \
--interval-min 0.4 \
--interval-max 0.6 \
--gen-bsz 64
```

To evaluate the FID of the iMF-L/2 model, run:

```bash
torchrun --nproc-per-node=8 evaluate.py evaluate \
--ckpt-path /path/to/L/2/checkpoint.pth \
--workdir ./l2_fid_output \
--model MiT_L_2 \
--cfg-omega 10.5 \
--interval-min 0.4 \
--interval-max 0.6 \
--gen-bsz 32
```

To evaluate the FID of the iMF-XL/2 model, run:

```bash
torchrun --nproc-per-node=8 evaluate.py evaluate \
--ckpt-path /path/to/XL/2/checkpoint.pth \
--workdir ./xl2_fid_output \
--model MiT_XL_2 \
--cfg-omega 8.0 \
--interval-min 0.42 \
--interval-max 0.62 \
--gen-bsz 32
```

To evaluate the FID of the iMF-XL/2 model with 2 NFE, run:

```bash
torchrun --nproc-per-node=8 evaluate.py evaluate \
--ckpt-path /path/to/XL/2/checkpoint.pth \
--workdir ./xl2_2nfe_fid_output \
--model MiT_XL_2 \
--num-sampling-steps 2 \
--cfg-omega 4.0 \
--interval-min 0.36 \
--interval-max 0.64 \
--gen-bsz 32
```

You may use the `--save-samples` option to keep the sampled images after FID evaluation is finished. Otherwise, they will automatically be removed.

For FID evaluation, we use the pre-computed reference file from [JiT](https://github.com/LTH14/JiT).

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

We gratefully acknowledge the Google TPU Research Cloud (TRC) for granting TPU access, and Mingyang Deng for supporting GPU resources.
We hope this work will serve as a useful resource for the open-source community.

A portion of codes in this repo is based on [JiT](https://github.com/LTH14/JiT).
