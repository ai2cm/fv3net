from fv3fit.pytorch.cyclegan.image_pool import ImagePool
import numpy as np
import torch


def test_image_times_are_paired():
    pool = ImagePool(20)
    times = torch.as_tensor(np.arange(10))
    images = torch.as_tensor(np.arange(10)[:, None, None])
    pool.query((times, images))
    pool.query((times, images))
    result = pool.query((times, images))
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert not np.all(result[0].cpu().numpy() == times.cpu().numpy())
    for time, image in zip(result[0], result[1]):
        assert np.all(image.cpu().numpy() == time.cpu().numpy())
