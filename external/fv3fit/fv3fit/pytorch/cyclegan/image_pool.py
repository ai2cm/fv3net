# flake8: noqa
# Taken from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/eb6ae80412e23c09b4317b04d889f1af27526d2d/util/image_pool.py
# Copyright (c) 2017, Jun-Yan Zhu and Taesung Park under a BSD license

import random
from typing import Tuple
import torch


class ImagePool:
    """This class implements an image buffer that stores previously generated images.
    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class
        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []
            self.times = []

    def query(
        self, images: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return an image from the pool.
        Parameters:
            images: the latest generated images from the generator
        Returns images from the buffer.
        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_times = []
        return_images = []
        for time, image in zip(*images):
            time = torch.unsqueeze(time.data, 0)
            image = torch.unsqueeze(image.data, 0)
            if (
                self.num_imgs < self.pool_size
            ):  # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.times.append(time)
                self.images.append(image)
                return_times.append(time)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if (
                    p > 0.5
                ):  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(
                        0, self.pool_size - 1
                    )  # randint is inclusive
                    tmp_time = self.times[random_id].clone()
                    tmp_img = self.images[random_id].clone()
                    self.times[random_id] = time
                    self.images[random_id] = image
                    return_times.append(tmp_time)
                    return_images.append(tmp_img)
                else:  # by another 50% chance, the buffer will return the current image
                    return_times.append(time)
                    return_images.append(image)
        return_times = torch.cat(return_times, 0)  # collect all the times and return
        return_images = torch.cat(return_images, 0)  # collect all the images and return
        return (return_times, return_images)
