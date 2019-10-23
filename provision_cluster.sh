gcloud container clusters create snakemake-cluster \
   --num-nodes=1 --scopes storage-rw \
   --machine-type n1-standard-4 \
   --zone us-central1-a \
   --disk-type pd-ssd
