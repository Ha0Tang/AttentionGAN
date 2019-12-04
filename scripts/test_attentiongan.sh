set -ex
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_attentiongan --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 1 --gpu_ids 0 --num_test 1000000000 --saveDisk
