for cluster in $(seq 2 20)
  do
    python ../main.py --data_path '../../data' --dataset_name 'news20' \
        --n_hidden_features 10 \
        --cluster_type 'linear_word' --cluster_num $cluster --normal_class_index_list  0 1 2 3 4\
        --cluster_model_pretrain_epochs 400 --classifier_epochs 1\
        --classifier_type 'bert' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
  done
#
# for cluster in $(seq 2 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'linear_word' --cluster_num $cluster --normal_class_index_list  5 6 7 8\
        # --cluster_model_pretrain_epochs 400 --classifier_epochs 1\
        # --classifier_type 'bert' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
  # done
#
# for cluster in $(seq 2 7)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'linear_word' --cluster_num $cluster --normal_class_index_list 14 15 16\
        # --cluster_model_pretrain_epochs 400 --classifier_epochs 1\
        # --classifier_type 'bert' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
  # done

# for cluster in $(seq 2 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'linear_word' --cluster_num $cluster --normal_class_index_list  9 10 11 12\
        # --cluster_model_pretrain_epochs 400 --classifier_epochs 1\
        # --classifier_type 'bert' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
  # done

# for cluster in $(seq 2 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'linear_word' --cluster_num $cluster --normal_class_index_list  13\
        # --cluster_model_pretrain_epochs 400 --classifier_epochs 1\
        # --classifier_type 'bert' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
  # done

# for cluster in $(seq 2 20)
  # do
    # python ../main.py --data_path '../../data' --dataset_name 'news20' \
        # --n_hidden_features 10 \
        # --cluster_type 'linear_word' --cluster_num $cluster --normal_class_index_list  17 18 19\
        # --cluster_model_pretrain_epochs 400 --classifier_epochs 1\
        # --classifier_type 'bert' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
  # done

