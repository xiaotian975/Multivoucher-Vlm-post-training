"""Stronger visual augmentations that keep the same image dimensions and bbox coordinates."""

from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def apply_robust_augment(image: Image.Image, *, seed: int) -> Image.Image:
    """Apply blur/noise/jpeg-like perturbations without affine transforms."""

    rng = random.Random(seed)
    output = ImageEnhance.Brightness(image.copy()).enhance(rng.uniform(0.86, 1.10))
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.90, 1.18))
    if rng.random() < 0.7:
        output = output.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.35, 0.9)))

    array = np.asarray(output).astype(np.int16)
    noise = np.random.default_rng(seed).normal(loc=0.0, scale=rng.uniform(2.0, 6.0), size=array.shape)
    array = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB")
