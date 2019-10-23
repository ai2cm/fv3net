gcloud container clusters create snakemake-cluster \
   --num-nodes=4 --scopes storage-rw \
   --machine-type n1-standard-1 \
   --zone us-central1-a \
   --disk-type pd-ssd

# node pool for the model running step
gcloud container node-pools create big-mem --cluster snakemake-cluster \
--machine-type n1-standard-2 --num-nodes 1 --zone us-central1-a

# gcloud container node-pools delete big-mem --cluster snakemake-cluster \
# --machine-type n1-standard-2 --num-nodes 1 --zone us-central1-a
