#!/usr/bin/env python3
"""
Script for computing latent datasets from ImageNet and FID statistics.

This script:
1. Loads ImageNet data from the specified folder
2. Encodes images to latents using a VAE model
3. Saves the latent dataset to disk
4. Computes FID statistics and saves them

Usage:
    python compute_latent_dataset.py --config configs/default.py --imagenet_root /path/to/imagenet --output_dir /path/to/output
"""

import logging
import os

import jax
from absl import app, flags

# Initialize JAX distributed processing
jax.distributed.initialize()

from utils.data_util import compute_latent_dataset
from utils.fid_util import compute_fid_stats
from utils.logging_util import log_for_0

FLAGS = flags.FLAGS
flags.DEFINE_string("config", "configs/default.py", "Path to config file")
flags.DEFINE_string(
    "imagenet_root", "/path/to/imagenet", "Path to ImageNet dataset root"
)
flags.DEFINE_string(
    "output_dir", "/path/to/output", "Output directory for latent dataset and FID stats"
)
flags.DEFINE_integer("batch_size", 32, "Batch size for processing")
flags.DEFINE_string("vae_type", "mse", "VAE type (mse, ema)")
flags.DEFINE_integer(
    "image_size",
    256,
    "Image size for processing (common: 256->32x32, 512->64x64, 1024->128x128 latents)",
)
flags.DEFINE_boolean(
    "compute_latent", True, "Whether to compute and save latent dataset"
)
flags.DEFINE_boolean("compute_fid", True, "Whether to compute FID statistics")
flags.DEFINE_boolean("overwrite", False, "Whether to overwrite existing files")


def main(argv):
    """Main function."""
    del argv  # Unused

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Validate paths
    if not os.path.exists(FLAGS.imagenet_root):
        raise ValueError(f"ImageNet root path does not exist: {FLAGS.imagenet_root}")

    # Create output directory
    os.makedirs(FLAGS.output_dir, exist_ok=True)
    log_for_0(f"Output directory: {FLAGS.output_dir}")

    # Validate that at least one computation is requested
    if not FLAGS.compute_latent and not FLAGS.compute_fid:
        raise ValueError(
            "At least one of --compute_latent or --compute_fid must be True"
        )

    # Validate batch size compatibility with JAX distributed setup
    local_device_count = jax.local_device_count()
    if FLAGS.batch_size % local_device_count != 0:
        log_for_0(
            f"WARNING: Batch size {FLAGS.batch_size} is not divisible by local device count {local_device_count}"
        )
        log_for_0(
            "This will be handled by padding, but consider using a divisible batch size for optimal performance"
        )

    log_for_0(
        f"JAX distributed setup: process {jax.process_index()}/{jax.process_count()}, "
        f"local devices: {local_device_count}, total devices: {jax.device_count()}"
    )

    # Compute latent dataset
    if FLAGS.compute_latent:
        log_for_0("=" * 50)
        log_for_0("COMPUTING LATENT DATASET")
        log_for_0("=" * 50)

        compute_latent_dataset(
            imagenet_root=FLAGS.imagenet_root,
            output_dir=FLAGS.output_dir,
            vae_type=FLAGS.vae_type,
            batch_size=FLAGS.batch_size,
            image_size=FLAGS.image_size,
            overwrite=FLAGS.overwrite,
        )
    else:
        log_for_0("Skipping latent dataset computation")

    # Compute FID statistics
    if FLAGS.compute_fid:
        log_for_0("=" * 50)
        log_for_0("COMPUTING FID STATISTICS")
        log_for_0("=" * 50)

        fid_stats_path = compute_fid_stats(
            imagenet_root=FLAGS.imagenet_root,
            output_dir=FLAGS.output_dir,
            image_size=FLAGS.image_size,
            overwrite=FLAGS.overwrite,
        )

        log_for_0(f"FID statistics computed and saved to: {fid_stats_path}")
    else:
        log_for_0("Skipping FID statistics computation")

    log_for_0("=" * 50)
    log_for_0("COMPUTATION COMPLETED SUCCESSFULLY")
    log_for_0("=" * 50)


if __name__ == "__main__":
    app.run(main)
