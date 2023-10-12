import torch


class Normalizer:
    def __init__(self, dataloader):
        self.dataloader = dataloader

    def get_mean_std(self):
        num_pixels = 0
        mean = 0.0
        std = 0.0
        for images, _ in self.dataloader:
            images = images.to(torch.float)
            batch_size, num_channels, height, width = images.shape
            num_pixels += batch_size * height * width
            mean += images.mean(axis=(0, 2, 3)).sum()
            std += images.std(axis=(0, 2, 3)).sum()

        mean /= num_pixels
        std /= num_pixels

        return mean, std