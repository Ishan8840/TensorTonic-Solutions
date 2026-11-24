import numpy as np

def patch_embed(image: np.ndarray, patch_size: int,
                W_proj: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """
    Returns float64 patch embeddings with shape (B, N, D).
    """
    b, h, w, c = image.shape
    grid_height = h // patch_size
    grid_width = w // patch_size

    usable = image[:, :grid_height * patch_size, :grid_width * patch_size, :]

    patches = usable.reshape(b, grid_height, patch_size, grid_width, patch_size, c)

    patches = patches.transpose(0, 1, 3, 2, 4, 5)
    patches = patches.reshape(b, grid_height * grid_width, -1)
    return patches @ W_proj + bias