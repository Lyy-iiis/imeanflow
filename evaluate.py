import argparse
import numpy as np
import os
import time
import shutil
from pathlib import Path

import torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

import cv2

import utils.torch_dist_util as dist
import utils.torch_util as tu
from utils.vae_util import VAEWrapper
from utils.fidelity_wrapper import calculate_metrics
from imf import iMeanFlow

def print0(*args, **kwargs):
    if dist.process_index() == 0:
        print(*args, **kwargs)

def run_evaluate(
    model,
    workdir,
    vae_wrapper,
    fid_ref,
    num_samples: int = 50000,
    device_batch_size: int = 64,
    initial_seed: int = 0,
    keep_samples: bool = False,
    **inference_kwargs,
):
    model.eval()
    world_size = dist.process_count()
    local_rank = dist.process_index()
    num_steps = (
        num_samples // (device_batch_size * world_size) + 1
    )
    device_bs = device_batch_size

    # Construct the folder name for saving generated images.
    save_folder = os.path.join(
        workdir, 'fid_outputs'
    )
    if local_rank == 0:
        if os.path.exists(save_folder):
            shutil.rmtree(save_folder)
        os.makedirs(save_folder)

    print0(f"Save to: {save_folder}")
    if local_rank == 0 and not os.path.exists(save_folder):
        os.makedirs(save_folder)

    num_classes = model.num_classes
    assert (
        num_samples % num_classes == 0
    ), "Number of FID samples must be divisible by number of classes"
    class_label_gen_world = np.arange(0, num_classes).repeat(
        num_samples // num_classes
    )
    class_label_gen_world = np.hstack([class_label_gen_world, np.zeros(50000)])

    for i in range(num_steps):
        print0("Generation step {}/{}".format(i, num_steps))

        start_idx = world_size * device_bs * i + local_rank * device_bs
        end_idx = start_idx + device_bs
        labels_gen = class_label_gen_world[start_idx:end_idx]
        labels_gen = torch.from_numpy(labels_gen).long().cuda()
        sample_idx = world_size * device_bs * i + local_rank * device_bs + torch.arange(device_bs)

        sampled_latents = model.generate(
            n_sample=sample_idx.shape[0],
            rng=tu.BatchGenerator(device=dist.local_device(), seeds=sample_idx ^ initial_seed),
            labels=labels_gen,
            **inference_kwargs,
        )
        sampled_images = vae_wrapper.decode(sampled_latents)
        sampled_images = tu.device_get(sampled_images) # np array
        sampled_images = sampled_images.transpose(0, 2, 3, 1)  # b h w c

        # denormalize images
        sampled_images = (sampled_images + 1) / 2

        # distributed save images
        for b_id in range(device_bs):
            img_id = sample_idx[b_id].item()
            if img_id >= num_samples:
                break
            gen_img = np.round(np.clip(sampled_images[b_id] * 255, 0, 255))
            gen_img = gen_img.astype(np.uint8)[:, :, ::-1]
            cv2.imwrite(
                os.path.join(save_folder, "{}.png".format(str(img_id).zfill(5))),
                gen_img,
            )

    dist.barrier()

    fid, inception_score = None, None

    print0("Calculating FID...")
    if dist.process_index() == 0:
        metrics_dict = calculate_metrics(
            input1=save_folder,
            input2=fid_ref,
            cuda=True,
            isc=True,
            fid=True,
            kid=False,
            prc=False,
            verbose=False,
        )
        fid = metrics_dict["frechet_inception_distance"]
        inception_score = metrics_dict["inception_score_mean"]
        if not keep_samples:
            shutil.rmtree(save_folder)

    dist.barrier()
    print0("FID: {}, Inception Score: {}".format(fid, inception_score))
    return fid, inception_score

def get_args_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('mode', type=str, choices=['sample', 'evaluate'], help='Mode: sample (only sample a batch and visualize) or evaluate')
    
    # working directory
    parser.add_argument('--workdir', type=str, required=True)
    
    # checkpoint
    parser.add_argument('--ckpt-path', default='', type=str, metavar='PATH', 
                        help='Path to checkpoint to resume training', required=True)

    # architecture
    parser.add_argument('--model', default='imfDiT_B_2', type=str, metavar='MODEL',
                        help='Name of the model to train',
                        choices=['imfDiT_B_2', 'imfDiT_M_2', 'imfDiT_L_2', 'imfDiT_XL_2'])
    
    # sampling
    parser.add_argument('--sample-seed', default=42, type=int,
                        help='Random seed for sampling')
    parser.add_argument('--num-sampling-steps', default=1, type=int,
                        help='Sampling steps')
    parser.add_argument('--cfg-omega', default=8.0, type=float,
                        help='Classifier-free guidance factor')
    parser.add_argument('--interval-min', default=0.4, type=float,
                        help='CFG interval min')
    parser.add_argument('--interval-max', default=0.65, type=float,
                        help='CFG interval max')
    parser.add_argument('--num-images', default=50000, type=int,
                        help='Number of images to generate')
    parser.add_argument('--gen-bsz', type=int, default=64,
                        help='Generation batch size')
    parser.add_argument('--save-samples', action='store_true',
                        help='Whether to keep generated samples for FID calculation')
    
    # fid reference
    parser.add_argument('--fid-ref', type=str, default='https://raw.githubusercontent.com/LTH14/JiT/refs/heads/main/fid_stats/jit_in256_stats.npz',
                        help='Path or URL to FID reference statistics file')

    return parser


def main(args):
    dist.initialize()
    print0('Working directory:', args.workdir)
    print0("Arguments:\n{}".format(args).replace(', ', ',\n'))
    
    if dist.process_index() == 0:
        os.makedirs(args.workdir, exist_ok=True)

    # Set seeds for reproducibility
    tu.seed(0)
    
    # Create model
    model = iMeanFlow(args.model)
    
    # Create VAE
    vae_wrapper = VAEWrapper(decode_batch_size=args.gen_bsz)
    
    # Load checkpoint
    if args.ckpt_path and os.path.isfile(args.ckpt_path):
        print(f"Loading checkpoint from {args.ckpt_path}...")
        checkpoint = torch.load(args.ckpt_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print("Checkpoint loaded.")
    else:
        print(f"No checkpoint found at {args.ckpt_path}, exiting.")
        return
    
    model = tu.device_put(model)
    
    # Evaluate
    if args.mode == 'evaluate':
        run_evaluate(
            model,
            args.workdir,
            vae_wrapper,
            fid_ref=args.fid_ref,
            num_samples=args.num_images,
            device_batch_size=args.gen_bsz,
            initial_seed=args.sample_seed,
            keep_samples=args.save_samples,
            num_steps=args.num_sampling_steps,
            omega=args.cfg_omega,
            t_min=args.interval_min,
            t_max=args.interval_max,
        )
    elif args.mode == 'sample':
        labels = torch.tensor([207, 360, 387, 974, 88, 979, 417, 279], dtype=torch.int32)
        num_rows = 2
        num_cols = len(labels) // num_rows
        indices = torch.arange(labels.shape[0])
        sampled_latents = model.generate(
            n_sample=labels.shape[0],
            rng=tu.BatchGenerator(device=dist.local_device(), seeds=indices),
            num_steps=args.num_sampling_steps,
            omega=args.cfg_omega,
            t_min=args.interval_min,
            t_max=args.interval_max,
            labels=labels,
        )
        sampled_images = vae_wrapper.decode(sampled_latents)
        sampled_images = tu.device_get(sampled_images)  # np array

        # denormalize images
        sampled_images = (sampled_images + 1) / 2
        gen_img = np.round(np.clip(sampled_images * 255, 0, 255)).transpose((0, 2, 3, 1))
        
        # make a grid
        img_shape = gen_img.shape[1:]
        gen_img = np.einsum('rnhwc->rhnwc', gen_img.reshape(num_rows, num_cols, *img_shape)).reshape(
            num_rows * img_shape[0], num_cols * img_shape[1], img_shape[2]
        )
        assert gen_img.shape[-1] == 3, gen_img.shape
        
        gen_img = gen_img.astype(np.uint8)[:, :, ::-1] # rgb to bgr
        save_path = os.path.join(args.workdir, 'sampled_image.png')
        cv2.imwrite(save_path, gen_img)
        print(f"Sampled image saved to {save_path}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)