from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer, AutoTokenizer, BertModel, TrainingArguments, Trainer
import torch
import evaluate
import numpy as np

dataset = load_dataset("yangwang825/rte")

print(dataset)

tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

# Data for sentence classification
sentences = [" sent1 ", " sent2 "]
# Data for pair classification
sentence_pairs_text1 = [" sent3 ", " sent4 "]
sentence_pairs_text2 = [" sent5 ", " sent6 "]

# # Processed sentences for sentence classification
# tokenized_inp = tokenizer( sentences , truncation =True , padding =
# 	True , return_tensors ='pt', max_length=512)
# # AND ,
# # Processed sentences for sentence classification
# tokenized_inp = tokenizer( sentence_pairs_text1 ,
# 	sentence_pairs_text2 ,
# 	truncation =True , padding =
# 	True , return_tensors ='pt', max_length=512)

# Processed sentences for sentence classification
# tokenized_inp = tokenizer( sentences , truncation =True , padding =
# 	True , return_tensors ='pt', max_length=512)
# AND ,
# Processed sentences for sentence classification
tokenized_inp = tokenizer( dataset["train"]["text1"] ,
	dataset["train"]["text2"] ,
	truncation =True , padding =
	True , return_tensors ='pt', max_length=512)

def encode(examples):
    return tokenizer(examples["text1"], examples["text2"], truncation=True, padding="max_length" , return_tensors ='pt', max_length=512)

tokenized_datasets = dataset.map(encode, batched=True)

# tokenized_datasets = dataset.map(lambda examples: {"labels": examples["label"]}, batched=True)

# tokenized_datasets.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask", "label"])
# tokenized_datasets.set_format(type="torch", columns=["input_ids", "token_type_ids", "attention_mask"])

print(tokenized_datasets)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)#.select(range(1000))
eval_dataset = tokenized_datasets["test"].shuffle(seed=42)#.select(range(1000))

# print(tokenized_inp)

model = BertModel.from_pretrained("prajjwal1/bert-tiny") # Initialization
# out = model ( ** tokenized_inp )

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

training_args = TrainingArguments(output_dir="test_trainer")
metric = evaluate.load("accuracy")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()