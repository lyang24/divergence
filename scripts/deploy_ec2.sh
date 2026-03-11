#!/bin/bash
# Deploy divergence to EC2, preserving generated data (SAQ, DiskANN exports).
#
# Usage: ./scripts/deploy_ec2.sh [EC2_IP]
# Default IP from DIVERGENCE_EC2_IP env var or argument.

set -euo pipefail

EC2_IP="${1:-${DIVERGENCE_EC2_IP:-}}"
PEM="${DIVERGENCE_EC2_PEM:-$HOME/Downloads/ubuntu.pem}"

if [ -z "$EC2_IP" ]; then
    echo "Usage: $0 <EC2_IP>  or set DIVERGENCE_EC2_IP"
    exit 1
fi

echo "Deploying to ubuntu@$EC2_IP:/mnt/nvme/divergence/"

rsync -az \
    --delete \
    --exclude target \
    --exclude .git \
    --exclude 'data/cohere_100k/saq_*' \
    --exclude 'data/cohere_100k/diskann/' \
    --exclude 'scripts/saq_eval' \
    -e "ssh -i $PEM" \
    ~/repos/divergence/ \
    ubuntu@"$EC2_IP":/mnt/nvme/divergence/

echo "Done. SAQ data and DiskANN exports preserved."
