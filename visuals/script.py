import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

blocksPerGrid = (2, 3, 4)
threadsPerBlock = (4, 4, 4)


def get_block_id(bx, by, bz, grid_x, grid_y):
    return bx + by * grid_x + bz * grid_x * grid_y


def get_thread_id(tx, ty, tz, block_dim):
    return tx + ty * block_dim[0] + tz * block_dim[0] * block_dim[1]


fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection="3d")
ax.set_title("CUDA 3D Grid of 3D Blocks", fontsize=16)

grid_x, grid_y, grid_z = blocksPerGrid
block_dim = threadsPerBlock
thread_count_per_block = block_dim[0] * block_dim[1] * block_dim[2]
colors = ["red", "green", "blue", "orange", "purple", "brown", "cyan", "magenta"]

for bz in range(grid_z):
    for by in range(grid_y):
        for bx in range(grid_x):
            block_id = get_block_id(bx, by, bz, grid_x, grid_y)
            block_origin = np.array(
                [
                    bx * (block_dim[0] + 2),
                    by * (block_dim[1] + 2),
                    bz * (block_dim[2] + 2),
                ]
            )
            ax.text(
                block_origin[0],
                block_origin[1],
                block_origin[2] + 6,
                f"Block ID {block_id}\n({bx},{by},{bz})",
                color=colors[block_id % len(colors)],
                fontsize=8,
            )

            for tz in range(block_dim[2]):
                for ty in range(block_dim[1]):
                    for tx in range(block_dim[0]):
                        local_tid = get_thread_id(tx, ty, tz, block_dim)
                        global_tid = block_id * thread_count_per_block + local_tid

                        x = block_origin[0] + tx
                        y = block_origin[1] + ty
                        z = block_origin[2] + tz

                        ax.scatter(x, y, z, color=colors[block_id % len(colors)], s=20)
                        if local_tid == 0:
                            ax.text(x, y, z, f"{global_tid}", fontsize=6)

ax.set_xlabel("X (block/thread)")
ax.set_ylabel("Y (block/thread)")
ax.set_zlabel("Z (block/thread)")
plt.savefig("cuda_grid_labeled.png", dpi=300)
