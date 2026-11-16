#!/usr/bin/env bash
# Download and decrypt the Turner et al. (2025) EM training datasets into data/raw/.
# They are shipped encrypted (easy-dataset-share) to keep them out of web scrapes;
# the decrypted copy is gitignored for the same reason.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
curl -sfL -o training_datasets.zip.enc \
  https://github.com/clarifying-EM/model-organisms-for-EM/raw/main/em_organism_dir/data/training_datasets.zip.enc
uvx easy-dataset-share unprotect-dir training_datasets.zip.enc \
  -p model-organisms-em-datasets --remove-canaries
rm training_datasets.zip.enc
