gcloud container clusters create snakemake-cluster \
   --num-nodes=4 --scopes storage-rw \
   --machine-type n1-standard-1 \
   --zone us-central1-a \
   --disk-type pd-ssd
