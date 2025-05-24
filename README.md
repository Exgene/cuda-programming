# Learning CUDA Programming

I wanna learn how CUDA Programming works. I will probably try to apply this to existing ML models and learn lot of stuff.I am geniunely excited to code this.

## Set up

You can run this if you have a NVIDIA GPU. Basically docker pull and run.

But you also need the bindings for docker to access the GPU. There is a package download that after that.

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  }
}
EOF

sudo systemctl restart docker
```

```bash
sudo docker compose up -d
sudo docker exec -it cuda-dev bash
```

