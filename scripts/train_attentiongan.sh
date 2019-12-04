set -ex
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_attentiongan --model attention_gan --dataset_mode unaligned --pool_size 50 --no_dropout --norm instance --lambda_A 10 --lambda_B 10 --lambda_identity 0.5 --load_size 286 --crop_size 256 --batch_size 4 --niter 60 --niter_decay 0 --gpu_ids 0 --display_id 0 --display_freq 100 --print_freq 100
