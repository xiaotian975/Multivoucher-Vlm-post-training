"""Light visual augmentations that preserve image geometry."""

from __future__ import annotations

import random

from PIL import Image, ImageEnhance


def apply_light_augment(image: Image.Image, *, seed: int) -> Image.Image:
    """Apply mild brightness/contrast changes without changing coordinates."""

    rng = random.Random(seed)
    output = image.copy()
    output = ImageEnhance.Brightness(output).enhance(rng.uniform(0.94, 1.06))
    output = ImageEnhance.Contrast(output).enhance(rng.uniform(0.96, 1.08))
    return output
