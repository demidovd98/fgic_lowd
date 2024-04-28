python inference.py \
--dataset CUB \
--model_type cnn \
--model_name resnet50 \
--img_size 224 \
--resize_size 256 \
--train_batch_size 24 \
--eval_batch_size 24 \
--learning_rate 0.03 \
--num_steps 40000 \
--fp16 \
--eval_every 200 \
--split 10 \
--aug_type double_crop \
--montecarlo_dropout 0.1