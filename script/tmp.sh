CUDA_VISIBLE_DEVICES=2 python barlow_twins_pretrain.py --batch_size 1024 --num_workers 32 --device_num 1 --img_size 28 -
-lr 1e-4 --ckpt_dir ../../ckpt --log_dir ../../log/ --max_epochs 100 --ps test

--batch_size 4 --num_workers 4 --device_num 1 --img_size 28 --lr 1e-4 --ckpt_dir ../../ckpt --log_dir ../../log/ --max_epochs 100 --ps test