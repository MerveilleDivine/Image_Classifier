"""Backward-compatible dataset module.

Prefer importing from src.data_loaders in new code.
"""

from .data_loaders import (
    build_dip_test_transform,
    build_test_transform,
    build_train_transform,
    get_cifar10_datasets,
    get_cifar10_loaders,
)

__all__ = [
    "build_dip_test_transform",
    "build_test_transform",
    "build_train_transform",
    "get_cifar10_datasets",
    "get_cifar10_loaders",
]
