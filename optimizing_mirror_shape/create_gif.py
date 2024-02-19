#!/usr/bin/env python
# coding: utf-8

import imageio.v2 as imageio
import os
import sys


def count_files(directory):
    try:
        # List all files in the directory
        files = os.listdir(directory)

        # Count the number of files
        num_files = len(files)

        return num_files
    except FileNotFoundError:
        print(f"The directory '{directory}' does not exist.")
        return 0


def main():
    if len(sys.argv) != 3:
        print("Usage: python create_gif.py <name of run directory> <fps>")
        sys.exit(1)

    path = sys.argv[1]
    fps = int(sys.argv[2])
    path_to_raytrace = path + "/raytrace"
    steps = count_files(path_to_raytrace)

    rays = imageio.get_writer(path + "/rays.mp4", fps=fps)
    rays_zoomed = imageio.get_writer(path + "/rays_zoomed.mp4", fps=fps)

    for i in range(0, steps, 1):
        rays.append_data(imageio.imread(path_to_raytrace + "/step_%d.png" % i))
        rays_zoomed.append_data(
            imageio.imread(path_to_raytrace + "/step_%d_zoomed.png" % i)
        )

    rays.close()
    rays_zoomed.close()


if __name__ == "__main__":
    main()
