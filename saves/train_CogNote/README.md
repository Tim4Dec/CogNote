---
base_model: /share/home/tj13070/ctt/Meta-Llama-3-8B-Instruct
library_name: peft
license: other
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: train_2024-12-17-20-22-sum
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# train_2024-12-17-20-22-sum

This model is a fine-tuned version of [/share/home/tj13070/ctt/Meta-Llama-3-8B-Instruct](https://huggingface.co//share/home/tj13070/ctt/Meta-Llama-3-8B-Instruct) on the ctind_train dataset.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 2
- eval_batch_size: 8
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 8
- total_train_batch_size: 32
- total_eval_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results



### Framework versions

- PEFT 0.12.0
- Transformers 4.41.2
- Pytorch 2.4.0+cu121
- Datasets 2.20.0
- Tokenizers 0.19.1