import os

# for wordLSTM target
# command = 'python attack_classification.py --dataset_path data/yelp ' \
#           '--target_model wordLSTM --batch_size 128 ' \
#           '--target_model_path /scratch/jindi/adversary/BERT/results/yelp ' \
#           '--word_embeddings_path /data/medg/misc/jindi/nlp/embeddings/glove.6B/glove.6B.200d.txt ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path ./cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache'

# for BERT target
# command = 'python attack_classification.py --dataset_path data/yelp ' \
#           '--target_model bert ' \
#           '--target_model_path /scratch/jindi/adversary/BERT/results/yelp ' \
#           '--max_seq_length 256 --batch_size 32 ' \
#           '--counter_fitting_embeddings_path /data/medg/misc/jindi/nlp/embeddings/counter-fitted-vectors.txt ' \
#           '--counter_fitting_cos_sim_path /scratch/jindi/adversary/cos_sim_counter_fitting.npy ' \
#           '--USE_cache_path /scratch/jindi/tf_cache'

# '--counter_fitting_cos_sim_path cos_sim_counter_fitting.npy ' \
# python textfooler_attack.py --dataset_path outputs/ag/train_tok.csv --config_path model/base/config.json --nclasses 4 --target_model deberta --target_model_path model/base/pytorch_model.bin --counter_fitting_embeddings_path TextFooler/counter-fitted-vectors.txt --USE_cache_path /scratch/jindi/tf_cache
# python textfooler_attack.py --dataset_path outputs/imdb/train_tok.csv --config_path model/base/config.json --nclasses 4 --target_model deberta --target_model_path model/base/pytorch_model.bin --counter_fitting_embeddings_path TextFooler/counter-fitted-vectors.txt --counter_fitting_cos_sim_path TextFooler/cos_sim_counter_fitting.npy --USE_cache_path /tf_cache

# for DeBERTa target
command = 'python textfooler_attack.py --dataset_path TextFooler/data/imdb ' \
          '--target_model deberta ' \
          '--target_model_path model/base/pytorch_model.bin ' \
          '--config_path model/base/config.json' \
          '--nclasses 4 ' \
          '--max_seq_length 256 --batch_size 32 ' \
          '--counter_fitting_embeddings_path counter-fitted-vectors.txt ' \
          '--USE_cache_path /tf_cache'


os.system(command)
