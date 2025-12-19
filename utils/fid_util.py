import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax.experimental import multihost_utils
from tqdm import tqdm
from absl import logging

from .jax_fid import inception, resize
from .logging_util import log_for_0


def compute_fid(mu1, mu2, sigma1, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_1d(sigma1).astype(np.float64)
    sigma2 = np.atleast_1d(sigma2).astype(np.float64)

    assert mu1.shape == mu2.shape
    assert sigma1.shape == sigma2.shape

    diff = mu1 - mu2
    tr_covmean = np.sum(
        np.sqrt(np.linalg.eigvals(sigma1.dot(sigma2)).astype("complex128")).real
    )
    fid = float(diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean)
    return fid


def build_jax_inception(batch_size=200):
    """
    Build InceptionV3 model that always returns all features.

    Args:
        batch_size: Batch size for compilation

    Returns:
        Dictionary with model parameters and compiled function
    """
    logging.info("Initializing Extended InceptionV3")
    model = inception.InceptionV3(
        pretrained=True,
        include_head=True,  # Need head for logits
        transform_input=False,  # Already normalized in resize.forward
    )

    # Initialize with dummy input
    dummy_input = jnp.ones((1, 299, 299, 3))
    rng = jax.random.PRNGKey(0)
    inception_params = model.init(rng, dummy_input, train=False)

    logging.info("Initialized Extended InceptionV3")

    # Create a single function that always returns all features
    def inception_apply(params, x):
        return model.apply(params, x, train=False)

    # JIT compile the function
    inception_fn = jax.jit(inception_apply)

    # Compile for the expected batch size
    fake_x = jnp.zeros((batch_size, 299, 299, 3), dtype=jnp.float32)
    logging.info("Start compiling inception function...")
    t_start = time.time()

    # Trigger compilation
    _ = inception_fn(inception_params, fake_x)

    logging.info(f"End compiling: {(time.time() - t_start):.4f} seconds.")

    inception_net = {"params": inception_params, "fn": inception_fn, "model": model}
    return inception_net


def get_reference(cache_path):
    # Load ref_mu and ref_sigma from npz file
    assert os.path.exists(cache_path), f"Cache file must exist: {cache_path}"

    log_for_0(f"Loading ref_mu and ref_sigma from {cache_path}")
    if jax.process_index() == 0:
        os.system("md5sum " + cache_path)

    ref = {}

    with np.load(cache_path) as data:
        if "ref_mu" in data:
            ref_mu, ref_sigma = data["ref_mu"], data["ref_sigma"]
        else:
            raise NotImplementedError

    ref = {"mu": ref_mu, "sigma": ref_sigma}
    return ref


LDC = jax.local_device_count()


def revert_pmap_shape(x):
    return x.reshape((-1, *x.shape[2:]))


def compute_stats(
    samples,
    inception_net,
    batch_size=200,
    fid_samples=50000,
):
    inception_fn = inception_net["fn"]
    inception_params = inception_net["params"]

    num_samples = len(samples)
    full_batch_size = batch_size * LDC
    pad = int(np.ceil(num_samples / full_batch_size)) * full_batch_size - num_samples
    samples = np.concatenate(
        [samples, np.zeros((pad, *samples.shape[1:]), dtype=np.uint8)]
    )
    assert len(samples) % full_batch_size == 0

    l_feats = []
    l_logits = []

    for i in range(0, len(samples), full_batch_size):
        x = samples[i : i + full_batch_size].astype(np.float32).transpose(0, 3, 1, 2)
        x = torch.from_numpy(x)  # uint8 to float32
        log_for_0(f"Evaluating {i} / {len(samples)}: {list(x.shape)}")
        x = resize.forward(x)  # match the Pytorch version
        x = x.numpy().transpose(0, 2, 3, 1)

        pooled_features, spatial_features, logits = inception_fn(
            inception_params, jax.lax.stop_gradient(x)
        )

        # Pooled features are already in the right shape [B, 2048]
        l_feats.append(pooled_features)
        l_logits.append(logits)

    # Process pooled features
    np_feats = jnp.concatenate(l_feats)
    np_feats = np_feats[:num_samples]

    all_feats = multihost_utils.process_allgather(np_feats)
    all_feats = all_feats.reshape(-1, np_feats.shape[-1])
    all_feats = jax.device_get(all_feats)

    log_for_0(
        f"FID final samples: {all_feats.shape[0]} samples -> {fid_samples} samples"
    )
    all_feats = all_feats[:fid_samples]
    # Convert to float64 for higher precision FID computation
    all_feats_64 = all_feats.astype(np.float64)
    mu = np.mean(all_feats_64, axis=0)
    sigma = np.cov(all_feats_64, rowvar=False)

    result = {"mu": mu, "sigma": sigma}

    np_logits = jnp.concatenate(l_logits)
    np_logits = np_logits[:num_samples]

    all_logits = multihost_utils.process_allgather(np_logits)
    all_logits = all_logits.reshape(-1, np_logits.shape[-1])
    all_logits = jax.device_get(all_logits)
    all_logits = all_logits[:fid_samples]

    result["logits"] = all_logits

    return result


def compute_inception_score(logits, splits=10):
    """
    Compute Inception Score from logits.

    Args:
        logits: Raw logits from InceptionV3 model, shape [N, num_classes]
        splits: Number of splits for computing IS (default: 10)

    Returns:
        is_mean: Mean inception score
        is_std: Standard deviation of inception score
    """
    rng = np.random.RandomState(2020)
    logits = logits[rng.permutation(logits.shape[0]), :]

    # Convert logits to probabilities
    probs = jax.nn.softmax(logits, axis=-1)
    probs_64 = np.array(probs, dtype=np.float64)

    # Split the probabilities
    N = probs_64.shape[0]
    split_size = N // splits

    scores = []
    for i in range(splits):
        part = probs_64[i * split_size : (i + 1) * split_size]

        # Compute p(y|x) - conditional distribution
        py_x = part

        # Compute p(y) - marginal distribution
        py = np.mean(part, axis=0, keepdims=True)

        # Compute KL divergence
        kl_div = py_x * (np.log(py_x + 1e-10) - np.log(py + 1e-10))
        kl_div = np.sum(kl_div, axis=1)
        kl_div = np.mean(kl_div)

        scores.append(np.exp(kl_div))

    scores = np.array(scores, dtype=np.float64)
    is_mean = np.mean(scores)
    is_std = np.std(scores)

    return is_mean, is_std


def compute_fid_stats(
    imagenet_root, output_dir, image_size, batch_size=200, overwrite=False
):
    """Compute and save FID statistics for ImageNet using distributed loading and chunked gathering."""
    from utils.data_util import create_imagenet_dataloader

    log_for_0("Starting FID statistics computation...")

    # Output path for FID stats
    fid_stats_path = os.path.join(output_dir, f"imagenet_{image_size}_fid_stats.npz")

    # Check if already exists
    if not overwrite and os.path.exists(fid_stats_path):
        log_for_0(f"FID stats already exist at {fid_stats_path}, skipping...")
        return fid_stats_path

    # Build Inception model
    inception_net = build_jax_inception(batch_size=batch_size)

    # Create dataloader for training set (for FID reference)
    # Use num_workers=0 to avoid fork() incompatibility with JAX multithreading
    dataloader, dataset_size, true_total_samples = create_imagenet_dataloader(
        imagenet_root, "train", batch_size, image_size, num_workers=0, for_fid=True
    )

    log_for_0(f"Computing FID features for {dataset_size} samples per worker...")
    log_for_0(f"Expected batches per worker: {len(dataloader)}")

    # Process data batch by batch and accumulate features
    log_for_0("Processing batches and computing features...")
    all_features_list = []

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        images, labels = batch

        # Convert images to numpy array format
        if isinstance(images, list):
            images_np = np.stack(images, axis=0)
        else:
            images_np = np.array(images)

        # Compute features for this batch directly
        batch_features = compute_batch_features(images_np, inception_net, batch_size)

        # Move to CPU and accumulate
        batch_features_cpu = jax.device_get(batch_features)
        all_features_list.append(batch_features_cpu)

        if batch_idx % 100 == 0:
            log_for_0(
                f"Worker {jax.process_index()}: Processed {batch_idx}/{len(dataloader)} batches"
            )

    # Concatenate all local features from this worker
    local_features = np.concatenate(all_features_list, axis=0)
    log_for_0(
        f"Worker {jax.process_index()}: Local features shape: {local_features.shape}"
    )

    # Clear feature list to free memory
    del all_features_list

    # Gather features across all workers using chunked approach to avoid OOM
    log_for_0("Gathering features across workers using chunked approach...")

    # Use smaller chunk size to avoid OOM (10K samples per chunk)
    chunk_size = 10000
    all_gathered_features = []

    for chunk_start in range(0, local_features.shape[0], chunk_size):
        chunk_end = min(chunk_start + chunk_size, local_features.shape[0])
        local_chunk = local_features[chunk_start:chunk_end]

        log_for_0(
            f"Worker {jax.process_index()}: Gathering chunk {chunk_start//chunk_size + 1}, "
            f"samples {chunk_start}:{chunk_end} ({local_chunk.shape[0]} samples)"
        )

        # Convert to JAX array and gather this chunk across all processes
        local_chunk_jax = jnp.array(local_chunk)

        # Gather this chunk from all workers
        gathered_chunk = multihost_utils.process_allgather(local_chunk_jax)
        gathered_chunk = gathered_chunk.reshape(-1, gathered_chunk.shape[-1])

        # Move to CPU to free memory
        gathered_chunk_cpu = jax.device_get(gathered_chunk)
        all_gathered_features.append(gathered_chunk_cpu)

        log_for_0(
            f"Worker {jax.process_index()}: Successfully gathered chunk {chunk_start//chunk_size + 1}, "
            f"total shape: {gathered_chunk_cpu.shape}"
        )

    # Concatenate all gathered chunks
    all_features_gathered = np.concatenate(all_gathered_features, axis=0)
    log_for_0(f"Total features shape before truncation: {all_features_gathered.shape}")

    # Truncate the padding by gathering
    if all_features_gathered.shape[0] != true_total_samples:
        log_for_0("Truncating to expected number of samples to fix padding...")
        all_features_gathered = all_features_gathered[:true_total_samples]

    log_for_0(f"Final features shape after truncation: {all_features_gathered.shape}")

    # Clear local features to free memory
    del local_features

    # Compute statistics
    log_for_0("Computing final statistics...")
    mu = np.mean(all_features_gathered, axis=0)
    sigma = np.cov(all_features_gathered, rowvar=False)

    # Save statistics
    os.makedirs(os.path.dirname(fid_stats_path), exist_ok=True)
    np.savez(fid_stats_path, ref_mu=mu, ref_sigma=sigma)
    log_for_0(f"FID statistics saved to {fid_stats_path}")

    return fid_stats_path


def compute_batch_features(batch_images, inception_net, batch_size):
    """Compute Inception features for a batch of images."""
    actual_batch_size = batch_images.shape[0]
    inception_params = inception_net["params"]
    inception_fn = inception_net["fn"]

    # Convert uint8 [0,255] numpy to float32 [0,255] tensor
    x = torch.tensor(batch_images, dtype=torch.float32)
    x = x.permute(0, 3, 1, 2)  # BHWC → BCHW for PyTorch

    # Apply resize and normalization, then convert to JAX format
    x = resize.forward(x)  # Resize to 299x299 and normalize to [-1,1]
    x = x.numpy().transpose(0, 2, 3, 1)  # BCHW → BHWC for JAX

    # Pad batch to expected size if needed (for JAX compilation compatibility)
    if actual_batch_size < batch_size:
        # Pad with zeros to reach expected batch size
        padding_size = batch_size - actual_batch_size
        padding_shape = (padding_size,) + x.shape[1:]
        padding = np.zeros(padding_shape, dtype=x.dtype)
        x_padded = np.concatenate([x, padding], axis=0)
    else:
        x_padded = x

    # Extract Inception features
    pred, _, _ = inception_fn(inception_params, jax.lax.stop_gradient(x_padded))
    pred = pred.squeeze(axis=1).squeeze(axis=1)

    # Return only the features for actual samples (remove padding)
    pred = pred[:actual_batch_size]

    return jax.device_get(pred)
