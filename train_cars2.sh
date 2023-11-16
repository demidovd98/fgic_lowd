#Vanilla:
#python3 -W ignore train.py --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 40000 --fp16 --eval_every 200 --dataload_workers 12 --split 10 --vanilla

#Ours:
python3 -W ignore train.py --name autoScheduler --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 20000 --fp16 --eval_every 500 --dataload_workers 12 --split 10 --dist_coef 1.0 --aug_type double_crop --auto_scheduler
python3 -W ignore train.py --name autoScheduler --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 20000 --fp16 --eval_every 500 --dataload_workers 12 --split 30 --dist_coef 1.0 --aug_type double_crop --auto_scheduler
python3 -W ignore train.py --name autoScheduler --dataset cars --model_type cnn --model_name resnet50 --img_size 224 --resize_size 256 --train_batch_size 24 --eval_batch_size 24 --learning_rate 0.03 --num_steps 20000 --fp16 --eval_every 500 --dataload_workers 12 --split 50 --dist_coef 1.0 --aug_type double_crop --auto_scheduler
