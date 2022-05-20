for cluster in 7 12 20
  do
    python ../main.py --data_path '../../data' --dataset_name 'news20' \
        --cluster_type 'DEC_bert' --cluster_num $cluster --normal_class_index_list  0 1 2 3 4\
        --cluster_model_train_epochs 10 --classifier_epochs 3\
        --cluster_model_train_lr 3e-6 --classifier_lr 3e-6 \
        --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
        --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0\
        --cluster_model_batch_size 8
  done
#
for cluster in 6 10 17
  do
    python ../main.py --data_path '../../data' --dataset_name 'news20' \
        --cluster_type 'DEC_bert' --cluster_num $cluster --normal_class_index_list  13\
        --cluster_model_train_epochs 10 --classifier_epochs 3\
        --cluster_model_train_lr 3e-6 --classifier_lr 3e-6 \
        --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001 \
        --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0 \
        --cluster_model_batch_size 8
   done

# for cluster in $(seq 2 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'agnews' \
        # --cluster_type 'SCCL' --cluster_num $cluster --normal_class_index_list  4\
        # --cluster_model_train_epochs 0 --classifier_epochs 3\
        # --cluster_model_train_lr 3e-6 --classifier_lr 3e-6 \
        # --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
        # --cluster_loss_on True --gpu_device 2 --GT False --pooling 'mean' --alpha 1.0 \
        # --cluster_model_batch_size 32 --max_length 128 --iter 500
  # done
#
# for cluster in $(seq 1 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'DEC_bert' --cluster_num $cluster --normal_class_index_list  13\
        # --cluster_model_train_epochs 0 --classifier_epochs 3\
        # --cluster_model_train_lr 3e-6 --classifier_lr 3e-6 \
        # --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
        # --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0
  # done

# for cluster in $(seq 1 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'DEC_bert' --cluster_num $cluster --normal_class_index_list  14 15 16\
        # --cluster_model_train_epochs 0 --classifier_epochs 3\
        # --cluster_model_train_lr 3e-6 --classifier_lr 3e-6 \
        # --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001 \
        # --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0
  # done
#
#
# for cluster in $(seq 1 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'DEC_bert' --cluster_num $cluster --normal_class_index_list  17 18 19\
        # --cluster_model_train_epochs 0 --classifier_epochs 3\
        # --cluster_model_train_lr 3e-6 --classifier_lr 3e-6 \
        # --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
        # --cluster_loss_on True --gpu_device 2 --GT True --pooling 'mean' --alpha 1.0
  # done
