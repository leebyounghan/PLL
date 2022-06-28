python ../main.py --data_path '../../data' --dataset_name 'clinc' \
    --cluster_type 'SCCL' --cluster_num 150 --normal_class_index_list 0\
    --cluster_model_train_epochs 0 --classifier_epochs 1\
    --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
    --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
    --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0\
    --cluster_model_batch_size 128 --max_length 32 --iter 200  --language_model 'roberta-base'

python ../main.py --data_path '../../data' --dataset_name 'banking' \
    --cluster_type 'SCCL' --cluster_num 50 --normal_class_index_list 1\
    --cluster_model_train_epochs 0 --classifier_epochs 1\
    --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
    --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
    --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0\
    --cluster_model_batch_size 128 --max_length 32 --iter 200  --language_model 'roberta-base'

python ../main.py --data_path '../../data' --dataset_name 'hwu' \
    --cluster_type 'SCCL' --cluster_num 48 --normal_class_index_list 0\
    --cluster_model_train_epochs 0 --classifier_epochs 1\
    --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
    --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
    --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0\
    --cluster_model_batch_size 128 --max_length 32 --iter 200  --language_model 'roberta-base'

python ../main.py --data_path '../../data' --dataset_name 'stackoverflow' \
    --cluster_type 'SCCL' --cluster_num 15 --normal_class_index_list 0\
    --cluster_model_train_epochs 0 --classifier_epochs 1\
    --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
    --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
    --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0\
    --cluster_model_batch_size 128 --max_length 32 --iter 200  --language_model 'roberta-base'
#
