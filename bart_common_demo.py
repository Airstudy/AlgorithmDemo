#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import datasets
from datasets import load_dataset, dataset_dict
from transformers.utils import logging
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoTokenizer,
    PreTrainedModel,
    BertConfig,
)
import numpy as np
import lawrouge
import torch


def flatten(example):
    return {
        "document": example["content"],
        "summary": example["title"],
        "id": "0"
    }


dataset = load_dataset('json', data_files='./data/nlpcc_data.json', field='data')
dataset = dataset["train"].map(flatten, remove_columns=["title", "content"])  # , remove_columns=["title", "content"]


TokenModel = "bert-base-chinese"
tokenizer = AutoTokenizer.from_pretrained(TokenModel)
config = BertConfig.from_pretrained(TokenModel)
model_checkpoint = "fnlp/bart-large-chinese"
print(model_checkpoint)

max_input_length = 512  # input, source text 注意长度，复旦BART中文预训练模型使用的bert tokenizer
max_target_length = 128  # summary, target text


if model_checkpoint in ["t5-small", "t5-base", "t5-larg", "t5-3b", "t5-11b"]:
    prefix = "summarize: "
else:
    prefix = "" # BART-12-3


def preprocess_function(examples):

    inputs = [prefix + doc for doc in examples["document"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["summary"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


raw_datasets = dataset
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

train_data_txt, validation_data_txt = dataset.train_test_split(test_size=0.1, shuffle=True, seed=42).values()
train_data_txt, test_data_tex = train_data_txt.train_test_split(test_size=0.1, shuffle=True, seed=42).values()
# 装载数据
dd = datasets.DatasetDict({"train": train_data_txt, "validation": validation_data_txt, "test": test_data_tex})

raw_datasets = dd
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

logger = logging.get_logger(__name__)

batch_size = 4
args = Seq2SeqTrainingArguments(
    output_dir="results",
    num_train_epochs=2,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,  # demo
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-04,
    warmup_steps=500,
    weight_decay=0.001,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=500,
    evaluation_strategy="epoch",
    save_total_limit=3,

    # generation_max_length最大生成长度，系统默认20 generation_num_beams=1表示贪心解码，大于1为树搜索
    generation_max_length=64,
    generation_num_beams=1,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


# 这里用的是中文lawrouge 至于字符级还是词级计算看自己调整 这里是字符级
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = ["".join(pred.replace(" ", "")) for pred in decoded_preds]
    decoded_labels = ["".join(label.replace(" ", "")) for label in decoded_labels]
    # Rouge with jieba cut
    # decoded_preds = [" ".join(jieba.cut(pred.replace(" ", ""))) for pred in decoded_preds]
    # decoded_labels = [" ".join(jieba.cut(label.replace(" ", ""))) for label in decoded_labels]

    labels_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in labels]
    # length = len(prediction_lens)

    # print(decoded_preds)
    # print(decoded_labels)
    rouge = lawrouge.Rouge()

    result = rouge.get_scores(decoded_preds, decoded_labels, avg=True)
    # print(result)
    print(result)
    result = {'rouge-1': result['rouge-1']['f'], 'rouge-2': result['rouge-2']['f'], 'rouge-l': result['rouge-l']['f']}

    result = {key: value * 100 for key, value in result.items()}
    return result;


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

train_result = trainer.train()
print(train_result)

trainer.save_model()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()



model.load_state_dict(torch.load('../mbart-large-cc25/pytorch_model.bin'))


def generate_summary(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        max_length=max_input_length,
        return_tensors="pt",
    )
    input_ids = inputs.input_ids.to(model.device)

    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
    print(outputs)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


_, x = generate_summary(
    "20日凌晨,寒风刺骨,两名年纪相仿的婴儿相继被狠心的父母遗弃在翔安的两个角落,一个在莲花总院厕所里,一个在东园社区一榕树下。两名婴儿被发现时间相距不过10分钟,莲河边防派出所民警连夜走访,未寻得婴儿家属。目前,一名婴儿已被送往福利院,另一名暂时安置在村民家中。据悉,经医生初步检查,两名婴儿均身体健康,无残疾、无疾病。记者陈佩珊通讯员蔡美娟林才龙",
    model)
print(x)
print(len(x[0]))

'''
tensor([[ 102,  101, 1336, 7305, 5425, 2128,  697, 1399, 2399, 5279, 4685,  820,
         4638, 2048, 1036, 4685, 5326, 6158, 6890, 2461, 1762,  697,  702, 6235,
         5862,  117,  671,  782, 1762, 5813, 5709, 2600, 7368, 1329, 2792, 7027,
          117,  671,  702, 1762,  691, 1736, 4852, 1277,  671, 3525, 3409,  678,
          511,  102]], device='cuda:0')
['厦 门 翔 安 两 名 年 纪 相 仿 的 婴 儿 相 继 被 遗 弃 在 两 个 角 落, 一 人 在 莲 花 总 院 厕 所 里, 一 个 在 东 园 社 区 一 榕 树 下 。']
91
'''

eval_results = trainer.evaluate()
print(eval_results)
