#Vanilla:
#python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_siz 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10 --vanilla

#Ours:
#python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10 --dist_coef 0.1 --lr_ratio 10.0 --aug_type single_crop

#python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10 --aug_type single_crop --aug_crop aug_asymmAugs
python3 -W ignore train.py --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10 --aug_type single_crop --aug_crop aug_asymmAugs
python3 -W ignore train.py --dataset air --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10 --aug_type single_crop --aug_crop aug_asymmAugs

python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 30 --aug_type single_crop --aug_crop aug_asymmAugs
python3 -W ignore train.py --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 30 --aug_type single_crop --aug_crop aug_asymmAugs
python3 -W ignore train.py --dataset air --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 30 --aug_type single_crop --aug_crop aug_asymmAugs

#python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10
python3 -W ignore train.py --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10
python3 -W ignore train.py --dataset air --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10

python3 -W ignore train.py --dataset CUB --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 30
python3 -W ignore train.py --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 30
python3 -W ignore train.py --dataset air --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 30