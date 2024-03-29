# for cluster in $(seq 10 14) #10까진 해놈
# do
  # python ../main.py --data_path '../../data' --dataset_name 'rostd' \
      # --cluster_type 'SCCL' --cluster_num $cluster --normal_class_index_list 0\
      # --cluster_model_train_epochs 1 --classifier_epochs 0\
      # --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
      # --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
      # --cluster_loss_on True --gpu_device 1 --GT False --pooling 'mean' --alpha 1.0\
      # --cluster_model_batch_size 32 --max_length 128 --iter 500  --language_model 'roberta-base'
# done
#
# for cluster in $(seq 10 14)
# do
  # python ../main.py --data_path '../../data' --dataset_name 'rostd' \
      # --cluster_type 'SCCL' --cluster_num $cluster --normal_class_index_list 0\
      # --cluster_model_train_epochs 1 --classifier_epochs 1\
      # --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
      # --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
      # --cluster_loss_on True --gpu_device 1 --GT False --pooling 'mean' --alpha 1.0\
      # --cluster_model_batch_size 32 --max_length 128 --iter 500  --language_model 'roberta-base'
# done
#

python ../main.py --data_path '../../data' --dataset_name 'clinc' \
    --cluster_type 'SCCL' --cluster_num 150 --normal_class_index_list 0\
    --cluster_model_train_epochs 1 --classifier_epochs 1\
    --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
    --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
    --cluster_loss_on True --gpu_device 1 --GT False --pooling 'mean' --alpha 1.0\
    --cluster_model_batch_size 32 --max_length 128 --iter 500  --language_model 'roberta-base'
