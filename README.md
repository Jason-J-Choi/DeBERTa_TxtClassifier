# DeBERTa Text Classifier

This repository is a purely academic implementation in determining the robustness of [ **DeBERTa**](https://arxiv.org/abs/2006.03654) in its ability to perform text classification from the [ **TextFooler**](https://arxiv.org/abs/1907.11932) adversarial generator.

## Attributes
- DeBERTa model location: [here](https://huggingface.co/models?search=microsoft%2Fdeberta)
- Datasets to be used from TextFooler [here](https://bit.ly/nlp_adv_data).

## Introduction to DeBERTa 
DeBERTa (Decoding-enhanced BERT with disentangled attention) improves the BERT and RoBERTa models using two novel techniques. The first is the disentangled attention mechanism, where each word is represented using two vectors that encode its content and position, respectively, and the attention weights among words are computed using disentangled matrices on their contents and relative positions. Second, an enhanced mask decoder is used to replace the output softmax layer to predict the masked tokens for model pretraining. We show that these two techniques significantly improve the efficiency of model pre-training and performance of downstream tasks.
Additional documentation [here](https://deberta.readthedocs.io/en/latest/)
Specifics of installing DeBERTa can be found directly at the [Github page](https://github.com/Jason-J-Choi/DeBERTa)

## Introducing TextFooler**
A Model for Natural Language Attack on Text Classification and Inference

## Setup
*Install the necessary requirement files from the requirements.txt. The requirements.txt was compiled from all the supporting repositories. Alternatively, install DeBERTa via Docker and separately install TextFooler.
```
pip install requirements.txt
```

*Install the TextFooler supporting system [ESIM system](https://github.com/coetaur0/ESIM).
```
cd ESIM
python setup.py install
cd ..
```

#### Use DeBERTa in existing code
``` Python

# To apply DeBERTa into your existing code, you need to make two changes on your code,
# 1. change your model to consume DeBERTa as the encoder
from DeBERTa import deberta
import torch
class MyModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    # Your existing model code
    self.deberta = deberta.DeBERTa(pre_trained='base') # Or 'large' 'base-mnli' 'large-mnli' 'xlarge' 'xlarge-mnli' 'xlarge-v2' 'xxlarge-v2'
    # Your existing model code
    # do inilization as before
    # 
    self.deberta.apply_state() # Apply the pre-trained model of DeBERTa at the end of the constructor
    #
  def forward(self, input_ids):
    # The inputs to DeBERTa forward are
    # `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length] with the word token indices in the vocabulary
    # `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token types indices selected in [0, 1]. 
    #    Type 0 corresponds to a `sentence A` and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
    # `attention_mask`: an optional parameter for input mask or attention mask. 
    #   - If it's an input mask, then it will be torch.LongTensor of shape [batch_size, sequence_length] with indices selected in [0, 1]. 
    #      It's a mask to be used if the input sequence length is smaller than the max input sequence length in the current batch. 
    #      It's the mask that we typically use for attention when a batch has varying length sentences.
    #   - If it's an attention mask then if will be torch.LongTensor of shape [batch_size, sequence_length, sequence_length]. 
    #      In this case, it's a mask indicate which tokens in the sequence should be attended by other tokens in the sequence. 
    # `output_all_encoded_layers`: whether to output results of all encoder layers, default, True
    encoding = deberta.bert(input_ids)[-1]

# 2. Change your tokenizer with the the tokenizer built in DeBERta
from DeBERTa import deberta
vocab_path, vocab_type = deberta.load_vocab(pretrained_id='base')
tokenizer = deberta.tokenizers[vocab_type](vocab_path)
# We apply the same schema of special tokens as BERT, e.g. [CLS], [SEP], [MASK]
max_seq_len = 512
tokens = tokenizer.tokenize('Examples input text of DeBERTa')
# Truncate long sequence
tokens = tokens[:max_seq_len -2]
# Add special tokens to the `tokens`
tokens = ['[CLS]'] + tokens + ['[SEP]']
input_ids = tokenizer.convert_tokens_to_ids(tokens)
input_mask = [1]*len(input_ids)
# padding
paddings = max_seq_len-len(input_ids)
input_ids = input_ids + [0]*paddings
input_mask = input_mask + [0]*paddings
features = {
'input_ids': torch.tensor(input_ids, dtype=torch.int),
'input_mask': torch.tensor(input_mask, dtype=torch.int)
}

```

#### Run DeBERTa experiments from command line
For glue tasks, 
1. Get the data
``` bash
cache_dir=/tmp/DeBERTa/
cd experiments/glue
./download_data.sh  $cache_dir/glue_tasks
```

2. Run task

``` bash
task=STS-B 
OUTPUT=/tmp/DeBERTa/exps/$task
export OMP_NUM_THREADS=1
python3 -m DeBERTa.apps.run --task_name $task --do_train  \
  --data_dir $cache_dir/glue_tasks/$task \
  --eval_batch_size 128 \
  --predict_batch_size 128 \
  --output_dir $OUTPUT \
  --scale_steps 250 \
  --loss_scale 16384 \
  --accumulative_update 1 \  
  --num_train_epochs 6 \
  --warmup 100 \
  --learning_rate 2e-5 \
  --train_batch_size 32 \
  --max_seq_len 128
```

## Notes
- 1. By default we will cache the pre-trained model and tokenizer at `$HOME/.~DeBERTa`, you may need to clean it if the downloading failed unexpectedly.
- 2. You can also try our models with [HF Transformers](https://github.com/huggingface/transformers). But when you try XXLarge model you need to specify --sharded_ddp argument. Please check our [XXLarge model card](https://huggingface.co/microsoft/deberta-xxlarge-v2) for more details.


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

## Experiments
Our fine-tuning experiments are carried on half a DGX-2 node with 8x32 V100 GPU cards, the results may vary due to different GPU models, drivers, CUDA SDK versions, using FP16 or FP32, and random seeds. 
We report our numbers based on multple runs with different random seeds here. Here are the results from the Large model:

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|**MNLI xxlarge v2**|	`experiments/glue/mnli.sh xxlarge-v2`|	**91.7/91.9** +/-0.1|	4h|
|MNLI xlarge v2|	`experiments/glue/mnli.sh xlarge-v2`|	91.7/91.6 +/-0.1|	2.5h|
|MNLI xlarge|	`experiments/glue/mnli.sh xlarge`|	91.5/91.2 +/-0.1|	2.5h|
|MNLI large|	`experiments/glue/mnli.sh large`|	91.3/91.1 +/-0.1|	2.5h|
|QQP large|	`experiments/glue/qqp.sh large`|	92.3 +/-0.1|		6h|
|QNLI large|	`experiments/glue/qnli.sh large`|	95.3 +/-0.2|		2h|
|MRPC large|	`experiments/glue/mrpc.sh large`|	91.9 +/-0.5|		0.5h|
|RTE large|	`experiments/glue/rte.sh large`|	86.6 +/-1.0|		0.5h|
|SST-2 large|	`experiments/glue/sst2.sh large`|	96.7 +/-0.3|		1h|
|STS-b large|	`experiments/glue/Stsb.sh large`|	92.5 +/-0.3|		0.5h|
|CoLA large|	`experiments/glue/cola.sh`|	70.5 +/-1.0|		0.5h|

And here are the results from the Base model

|Task	 |Command	|Results	|Running Time(8x32G V100 GPUs)|
|--------|---------------|---------------|-------------------------|
|MNLI base|	`experiments/glue/mnli.sh base`|	88.8/88.5 +/-0.2|	1.5h|


#### Fine-tuning on NLU tasks

We present the dev results on SQuAD 1.1/2.0 and several GLUE benchmark tasks.

| Model                     | SQuAD 1.1 | SQuAD 2.0 | MNLI-m/mm   | SST-2 | QNLI | CoLA | RTE    | MRPC  | QQP   |STS-B |
|---------------------------|-----------|-----------|-------------|-------|------|------|--------|-------|-------|------|
|                           | F1/EM     | F1/EM     | Acc         | Acc   | Acc  | MCC  | Acc    |Acc/F1 |Acc/F1 |P/S   |
| BERT-Large                | 90.9/84.1 | 81.8/79.0 | 86.6/-      | 93.2  | 92.3 | 60.6 | 70.4   | 88.0/-       | 91.3/- |90.0/- |
| RoBERTa-Large             | 94.6/88.9 | 89.4/86.5 | 90.2/-      | 96.4  | 93.9 | 68.0 | 86.6   | 90.9/-       | 92.2/- |92.4/- |
| XLNet-Large               | 95.1/89.7 | 90.6/87.9 | 90.8/-      | 97.0  | 94.9 | 69.0 | 85.9   | 90.8/-       | 92.3/- |92.5/- |
| [DeBERTa-Large](https://huggingface.co/microsoft/deberta-large)<sup>1</sup> | 95.5/90.1 | 90.7/88.0 | 91.3/91.1| 96.5|95.3| 69.5| 91.0| 92.6/94.6| 92.3/- |92.8/92.5 |
| [DeBERTa-XLarge](https://huggingface.co/microsoft/deberta-xlarge)<sup>1</sup> | -/-  | -/-  | 91.5/91.2| 97.0 | - | -    | 93.1   | 92.1/94.3    | -    |92.9/92.7|
| [DeBERTa-V2-XLarge](https://huggingface.co/microsoft/deberta-v2-xlarge)<sup>1</sup>|95.8/90.8| 91.4/88.9|91.7/91.6| **97.5**| 95.8|71.1|**93.9**|92.0/94.2|92.3/89.8|92.9/92.9|
|**[DeBERTa-V2-XXLarge](https://huggingface.co/microsoft/deberta-v2-xxlarge)<sup>1,2</sup>**|**96.1/91.4**|**92.2/89.7**|**91.7/91.9**|97.2|**96.0**|**72.0**| 93.5| **93.1/94.9**|**92.7/90.3** |**93.2/93.1** |
--------

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


