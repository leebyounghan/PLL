for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 0 1 2 3 4 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 5 6 7 9 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
    done


for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 9 10 11 12 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 13 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 14 15 16 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
    done
    
for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 17 18 19 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 's_bert'
    done


for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 0 1 2 3 4 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_glove'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 5 6 7 9 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_glove'
    done


for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 9 10 11 12 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_glove'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 13 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_glove'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 14 15 16 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_glove'
    done
    
for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 17 18 19 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_glove'
    done
for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 0 1 2 3 4 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_fasttext'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 5 6 7 9 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_fasttext'
    done


for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 9 10 11 12 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_fasttext'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 13 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_fasttext'
    done

for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 14 15 16 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_fasttext'
    done
    
for ((i = 2; i<11; i++))
    do
        python ../main.py --data_path '../../data' --dataset_name 'news20' \
    --n_hidden_features 10 \
    --cluster_type 'linear_word' --cluster_num $i --normal_class_index_list 17 18 19 \
    --cluster_model_pretrain_epochs 400 --classifier_epochs 300\
    --classifier_type 'dec' --temperature 1000 --perturbation 0.001 --sentence_embedding 'avg_fasttext'
    done