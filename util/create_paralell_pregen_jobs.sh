#! /bin/bash

mkdir ../kube/pregen_jobs
NUM_FILES=20
for i in $(seq 1 $NUM_FILES)
do
  cat ../kube/wa-hls4ml-search-job-template.yml | sed "s/\$CONFIG/$i/" > ../kube/pregen_jobs/job-$i.yaml
done
