#!/bin/bash
set -e
sudo docker compose up -d
sudo docker exec -it cuda-dev bash
sudo docker compose down
