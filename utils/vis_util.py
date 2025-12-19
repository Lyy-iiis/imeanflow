import jax.numpy as jnp


def make_grid_visualization(vis, grid=8, max_bz=8):
    assert vis.ndim == 4
    n, h, w, c = vis.shape

    col = grid
    row = min(grid, n // col)
    if n % (col * row) != 0:
        n = col * row * max_bz
        vis = vis[:n]
        n, h, w, c = vis.shape
    assert n % (col * row) == 0

    vis = vis.reshape((-1, col, row * h, w, c))
    vis = jnp.einsum("mlhwc->mhlwc", vis)
    vis = vis.reshape((-1, row * h, col * w, c))

    bz = min(vis.shape[0], max_bz)
    vis = vis[:bz]
    return vis
