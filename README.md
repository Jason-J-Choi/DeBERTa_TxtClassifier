# DeBERTa Text Classifier

This repository is a purely academic implementation in determining the robustness of [ **DeBERTa**](https://arxiv.org/abs/2006.03654) in its ability to perform text classification from the [ **TextFooler**](https://arxiv.org/abs/1907.11932) adversarial generator.

Additionally modified TextFooler adversarial generator to consider for DeBERTa's disentangled attention by switching location of pairs of words and applying pair-wise synonym substition with adherence to original TextFooler's semantics requirements.

Credits of original development of DeBERTa and TextFooler goes to their respective authors

## Attributes
- DeBERTa model location: [here](https://huggingface.co/models?search=microsoft%2Fdeberta)
- Datasets to be used from TextFooler [here](https://bit.ly/nlp_adv_data).

## Introduction to DeBERTa 
DeBERTa (Decoding-enhanced BERT with disentangled attention) improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency of model pre-training and performance of downstream tasks.
Additional documentation [here](https://deberta.readthedocs.io/en/latest/)
Specifics of installing DeBERTa can be found directly at the [Github page](https://github.com/Jason-J-Choi/DeBERTa)

## Introducing TextFooler
A Model for Natural Language Attack on Text Classification and Inference

## Setup
* Due to large files, have git lfs setup.
```
git lfs install
```
If errors still prevail due to large file size, try with git bash terminal - https://github.com/git-lfs/git-lfs/issues/3216


* Install the necessary requirement files from the requirements.txt. The requirements.txt was compiled from all the supporting repositories. Alternatively, install DeBERTa via Docker and separately install TextFooler.
```
pip install requirements.txt
```

*Install the TextFooler supporting system [ESIM system](https://github.com/coetaur0/ESIM).
```
cd ESIM
python setup.py install
cd ..
```

#### Run DeBERTa experiments from command line
For glue tasks, 
1. Get the data
``` bash
cache_dir=/tmp/DeBERTa/
cd experiments/glue
./download_data.sh  $cache_dir/glue_tasks
```

2. Download models
``` bash
python download_model.py
# it will download the base model
```

2.1 Train if needed
``` bash
python train.py \
    --dataset_path #location of dataset to train with \\
    --target_model #location of model downloaded \\
```
Additional parameters of nclasses, target_model_path, learning_rate, and num_epochs.
If target_model_path is not specified, will automatically download from Hugging Face.

3. Run task
Look at original TextFooler code for more details
``` bash
python textfooler_attack.py \
    --dataset_path #location of dataset to validate \\
    --config_path #location of config.json file downloaded from 2. \\
    --target_model #location of model \\
    --target_model_type #base or xxlarge-v2. default base \\
```

## Additional Input for TextFooler
* (Optional) Run the following code to pre-compute the cosine similarity scores between word pairs based on the [counter-fitting word embeddings](https://drive.google.com/open?id=1bayGomljWb6HeYDMTDKXrh0HackKtSlx).

```
python comp_cos_sim_mat.py [PATH_TO_COUNTER_FITTING_WORD_EMBEDDINGS]
```

* Run the following code to generate the adversaries for text classification:

```
python attack_classification.py
```

For Natural langauge inference:

```
python attack_nli.py
```

Examples of run code for these two files are in [run_attack_classification.py](https://github.com/jind11/TextFooler/blob/master/run_attack_classification.py) and [run_attack_nli.py](https://github.com/jind11/TextFooler/blob/master/run_attack_nli.py). Here we explain each required argument in details:

  * --dataset_path: The path to the dataset. We put the 1000 examples for each dataset we used in the paper in the folder [data](https://github.com/jind11/TextFooler/tree/master/data).
  * --target_model: Name of the target model such as ''bert''.
  * --target_model_path: The path to the trained parameters of the target model. For ease of replication, we shared the [trained BERT model parameters](https://drive.google.com/drive/folders/1wKjelHFcqsT3GgA7LzWmoaAHcUkP4c7B?usp=sharing), the [trained LSTM model parameters](https://drive.google.com/drive/folders/108myH_HHtBJX8MvhBQuvTGb-kGOce5M2?usp=sharing), and the [trained CNN model parameters](https://drive.google.com/drive/folders/1Ifowzfers0m1Aw2vE8O7SMifHUhkTEjh?usp=sharing) on each dataset we used.
  * --counter_fitting_embeddings_path: The path to the counter-fitting word embeddings.
  * --counter_fitting_cos_sim_path: This is optional. If given, then the pre-computed cosine similarity scores based on the counter-fitting word embeddings will be loaded to save time. If not, it will be calculated.
  * --USE_cache_path: The path to save the USE model file (Downloading is automatic if this path is empty).
  
Two more things to share with you:

1. In case someone wants to replicate our experiments for training the target models, we shared the used [seven datasets](https://drive.google.com/open?id=1N-FYUa5XN8qDs4SgttQQnrkeTXXAXjTv) we have processed for you!

2. In case someone may want to use our generated adversary results towards the benchmark data directly, [here it is](https://drive.google.com/drive/folders/12yeqcqZiEWuncC5zhSUmKBC3GLFiCEaN?usp=sharing).

# Citation
DeBERTa and TextFooler respectively
```
@misc{he2020deberta,
    title={DeBERTa: Decoding-enhanced BERT with Disentangled Attention},
    author={Pengcheng He and Xiaodong Liu and Jianfeng Gao and Weizhu Chen},
    year={2020},
    eprint={2006.03654},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}

@article{jin2019bert,
  title={Is BERT Really Robust? Natural Language Attack on Text Classification and Entailment},
  author={Jin, Di and Jin, Zhijing and Zhou, Joey Tianyi and Szolovits, Peter},
  journal={arXiv preprint arXiv:1907.11932},
  year={2019}
}
```


