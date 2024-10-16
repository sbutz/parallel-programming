# parallel-programming
Shared Workspace for parallel programming workshop

## Setup
- Rent compute engine g2-standard-4 (os: ubuntu 24.04)
- Install required tools
```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install build-essential git tmux vim libjpeg-dev cuda-toolkit-12-6 cuda-runtime-12-6
```
- Add exports to ~/.bashrc
```bash
export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}% 
```
- Generate ssh key and upload to Github
```bash
ssh-keygen
```
- Clone repo
```bash
git clone git@github.com:sbutz/parallel-programming.git
```
- Connect via vs-code remote extension
