from utils import registry


MODELS = registry.Registry('models')


def build_model_from_cfg(cfg, **kwargs):
    """Build the registered model selected by cfg.NAME."""

    return MODELS.build(cfg, **kwargs)
