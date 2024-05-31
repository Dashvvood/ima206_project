python resnet18_end2end.py \
--project resnet18_pathmnist \
--batch_size 4 \
--warmup_steps 1000 \
--train_steps 10000 \
--device_num 1 \
--img_size 28 \
--ckpt_dir ../../ckpt/ \
--log_dir ../../log/ \
--lr 1e-4 \
--max_epochs 100 \
--ps test

