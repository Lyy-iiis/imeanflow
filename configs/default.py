"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""
    config = ml_collections.ConfigDict()

    # ------------------------------------------------------------
    # Dataset
    config.dataset = dataset = ml_collections.ConfigDict()

    dataset.name = "imgnet_latent"
    dataset.root = "DATA_ROOT"

    dataset.num_workers = 4
    dataset.prefetch_factor = 2
    dataset.pin_memory = False
    dataset.cache = False

    dataset.image_size = 32
    dataset.image_channels = 4
    dataset.num_classes = 1000
    dataset.vae = "mse"

    # ------------------------------------------------------------
    # Training
    config.training = training = ml_collections.ConfigDict()

    training.learning_rate = 0.0001
    training.batch_size = 256

    training.num_epochs = 1000

    training.log_per_step = 100
    training.sample_per_epoch = 10
    training.checkpoint_per_epoch = 10
    training.fid_per_epoch = 10
    training.half_precision = False

    training.seed = 42

    training.adam_b2 = 0.95
    training.ema_val = 0.9999

    training.lr_schedule = "warmup_const"
    training.warmup_epochs = 0

    # ------------------------------------------------------------
    # MeanFlow
    config.model = model = ml_collections.ConfigDict()
    model.num_classes = dataset.num_classes

    # Noise Distribution
    model.P_mean = -0.4
    model.P_std = 1.0

    # Loss
    model.data_proportion = 0.5
    model.cfg_beta = 1.0
    model.class_dropout_prob = 0.1

    # Training Dynamics
    model.norm_p = 1.0
    model.norm_eps = 0.01

    # ------------------------------------------------------------
    # Sampling
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.num_steps = 1
    sampling.num_classes = dataset.num_classes

    # ------------------------------------------------------------
    # FID
    config.fid = fid = ml_collections.ConfigDict()
    fid.num_samples = 50000
    fid.device_batch_size = 128
    fid.cache_ref = "FID_CACHE_REF"

    # ------------------------------------------------------------
    # Logging
    config.logging = logging = ml_collections.ConfigDict()
    logging.use_wandb = False
    logging.wandb_project = ""
    logging.wandb_entity = ""
    logging.wandb_notes = ""
    logging.wandb_tags = []

    # others
    config.load_from = ""
    config.eval_only = False

    return config
