from utils import registry


DATASETS = registry.Registry('dataset')


def build_dataset_from_cfg(cfg, default_args = None):
    """Build the registered dataset selected by cfg.NAME."""
    return DATASETS.build(cfg, default_args = default_args)
