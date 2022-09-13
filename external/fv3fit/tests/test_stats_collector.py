from fv3fit.pytorch.cyclegan.cyclegan_trainer import StatsCollector
import numpy as np


def test_stats_collector():
    np.random.seed(0)
    values = np.random.uniform(low=5.0, high=1.0, size=(100, 10))
    stats_collector = StatsCollector(n_dims_keep=1)
    for i in range(values.shape[0]):
        stats_collector.observe(values[i, :])
    assert np.allclose(stats_collector.mean, np.mean(values, axis=0))
    assert np.allclose(stats_collector.std, np.std(values, axis=0))
