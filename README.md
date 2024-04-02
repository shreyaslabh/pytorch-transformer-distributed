# pytorch-transformer-distributed

Distributed training of an attention model. Forked from: [hkproj/pytorch-transformer](https://github.com/hkproj/pytorch-transformer)

# DDP
## Instructions for Paperspace

### Machines

Make sure to create everything in the same region. I used `West Coast (CA1)`.

1. Create 1x Private network. Assign both computers to the private network when creating the machines.
2. Create 2x nodes of `P4000x2` (multi-GPU) with `ML-in-a-Box` as operating system
3. Create 1 Network drive (250 GB)

### Setup

Login on each machine and perform the following operations:

1. `sudo apt-get update`
2. `sudo apt-get install net-tools smbclient cifs-utils`
3. Mount the network drive
   1. `sudo mkdir /mnt/training-data`
   2. `sudo mount -t cifs //NETWORD_DRIVE_IP/NETWORK_SHARE_NAME /mnt/training-data -o uid=1000,gid=1000,rw,user,username=NETWORK_DRIVE_USERNAME`
4. `git clone https://github.com/shreyaslabh/pytorch-transformer-distributed`
5. `cd pytorch-transformer-distributed`
6. `pip install -r requirements.txt`
7. Login on Weights & Biases: `wandb login`   
8. Add ip & hostname in each other's /etc/hosts file
    1. ifconfig
    2. hostname
    3. sudo nano /etc/hosts
9. Run the training command from below

### Local training

Run the following command on any machine. Make sure to not run it on both, otherwise they will end up overwriting each other's checkpoints.

`torchrun --nproc_per_node=2 --nnodes=1 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=127.0.0.1:48123 train.py --batch_size 8 --model_folder "/mnt/training-data/weights"`

### Distributed training

Run the following command on each machine (replace `IP_ADDR_MASTER_NODE` with the IP address of the master node):

`torchrun --nproc_per_node=2 --nnodes=2 --rdzv_id=456 --rdzv_backend=c10d --rdzv_endpoint=IP_ADDR_MASTER_NODE:48123 train.py --batch_size 8 --model_folder "/mnt/training-data/weights"`

### Monitoring

Login to Weights & Biases to monitor the training progress: https://app.wandb.ai/


# FSDP

sudo apt-get update

sudo apt-get install net-tools smbclient cifs-utils

ifconfig

hostname

nano /etc/hosts

git clone https://github.com/shreyaslabh/pytorch-transformer-distributed

cd pytorch-transformer-distributed/FSDP

pip install -r requirements.txt

sh download_dataset.sh

torchrun --nnodes=1 --nproc_per_node=2  T5_training.py
