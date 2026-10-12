"""Load training modules on demand so inference does not import datasets."""

from importlib import import_module

__all__ = ["config", "data_loader", "weight_loaders"]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
