for cluster in $(seq 10 10 300)
do
    for seed in $(seq 0 0)
    do
      python ../main.py --data_path '../../data' --dataset_name 'hwu' \
          --cluster_type 'SCCL' --cluster_num $cluster --normal_class_index_list $seed\
          --cluster_model_train_epochs 3 --classifier_epochs 1\
          --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
          --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
          --cluster_loss_on True --gpu_device 1 --GT False --pooling 'mean' --alpha 1.0\
          --cluster_model_batch_size 128 --max_length 32 --iter 200  --language_model 'roberta-base'
    done

    # for seed in $(seq 0 4)
    # do
      # python ../main.py --data_path '../../data' --dataset_name 'stackoverflow' \
          # --cluster_type 'DEC_bert' --cluster_num $cluster --normal_class_index_list $seed\
          # --cluster_model_train_epochs 3 --classifier_epochs 1\
          # --cluster_model_train_lr 3e-6 --classifier_lr 3e-5\
          # --classifier_type 'DEC_bert' --temperature 1000 --perturbation 0.001\
          # --cluster_loss_on True --gpu_device 1 --GT False --pooling 'mean' --alpha 1.0\
          # --cluster_model_batch_size 32 --max_length 128 --iter 500   --language_model 'roberta-base'
      # done
done
#

