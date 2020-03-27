from distributed import Client
from dask_kubernetes import KubeCluster

cluster = KubeCluster.from_yaml('worker-spec.yml')
cluster.scale_up(1)  # specify number of nodes explicitly

client = Client(cluster)

res = client.submit(lambda : "test job succeeded")
print(res.result())
