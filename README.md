# ima206_project

## Installation

```
conda create -n ima206 python=3.10 -y
source activate ima206
pip install -r requirements.txt
```



## Usage

### simclr : open the notebook

### barlow twins

```shell
cd src/trainer

### Train Baseline (ResNet 18)
CUDA_VISIBLE_DEVICES=0 python resnet18_baseline.py \
--project barlow_twins \
--img_size 64 -p 1 \
--batch_size 2048  --lr 1e-2 --log_step 1 \
--max_epochs 1000 --warmup_epochs 50 \
--num_workers 16 \
--ckpt_dir ../../ckpt/ --log_dir ../../log/ \
--from_epoch 0 \
 --ps 100_lr1e-2 
 
 ### Pretrain Barlow Twins
 CUDA_VISIBLE_DIVICES=0 python barlow_twins_pretrain.py \
 --project barlow_twins \
 --batch_size 1024 --max_epoch 200 \
 --warmup_epochs 5 --num_workers 16 \
 --ckpt_dir ../../ckpt/ --log_dir ../../log/ 
 --lr 1e-4 --img_size 28 \
 -p 1 \
 --ps pretrain_100  <- this is some description for wandblogger saving run name

 ### Fine-tuning Barlow Twins
CUDA_VISIBLE_DEVICES=0 python barlow_twins_finetune.py \
--project barlow_twins --batch_size 2048 --accumulate_grad_batches 8  \
--log_step 5 --max_epochs 200 --warmup_epochs 20 --num_workers 8 \
--ckpt_dir ../../ckpt/ --log_dir ../../log/ \
--lr 1e-3 --img_size 64 -p 1 \
--ckpt ../../ckpt/20240618-101523/epoch=972-step=41839.ckpt  
--ps lr001  --from_epoch 0 \
--frozen # (add --frozen means linear probing, remove it means fine tuning)
```


## Github Usage


1. Clone this [repository](https://github.com/Lupin2019/ima206_project.git)


2. create a new branch base on `main` branch

   ```shell
   git checkout main 
   # Output: Already on 'main'
   git pull 
   # Output: Already up to date.
   git checkout -b abc123 # (new branch name)
   # Output: Switched to a new branch 'abc123'
   
   # make some updates
   echo "Hello" > a.txt
   
   git add a.txt
   git commit -m "Add a new file ./a.txt"
   
   # Only the for first push:
   # add a new remote branch, typically keep the same name with the local one.
   git push --set-upstream origin abc123
   
   # Next push:
   git push 
   ```

## Links
- [Barlow Twins: Self-Supervised Learning via Redundancy Reduction](https://arxiv.org/abs/2103.03230)
- [MedMNIST](https://medmnist.com/)
