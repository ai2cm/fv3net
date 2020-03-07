from dask_kubernetes import KubeCluster

cluster = KubeCluster.from_yaml('/home/jovyan/worker.yaml')
cluster.scale_up(5)  # specify number of nodes explicitly

# Example usage
import distributed
import dask.array as da

# Connect dask to the cluster
client = distributed.Client(cluster)

# Create an array and calculate the mean
array = da.ones((1000, 1000, 1000), chunks=(100, 100, 10))
print(array.mean().compute())  # Should print 1.0

