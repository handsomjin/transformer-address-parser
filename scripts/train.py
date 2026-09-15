import os
import json
import sys

import paddle
import pandas as pd
from paddle import nn
from paddle.io import DataLoader
from paddlenlp.transformers import BertModel, BertTokenizer, LinearDecayWithWarmup
from src.transformer_address_parser.data import AddressDataset
from src.transformer_address_parser.models.address_model import AddressModel
from src.transformer_address_parser.training.trainer import Trainer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
script_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(script_dir, "../data/areas_code.json")
json_path = os.path.normpath(json_path)


# 从 JSON 文件加载
def load_labels(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        label_list = json.load(f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label_list, label2id, id2label


def load_training_data(csv_file, tokenizer, label2id, max_src_len=128, max_tgt_len=3):
    def get_code_segment(code, level):
        if code is None or code == "":
            return "<unk>"
        return f"{level}_{code}"

    df = pd.read_csv(csv_file, sep=",", header=0, dtype=str, nrows=3)
    data = []
    for _, row in df.iterrows():
        data.append({
            "text": row["address"],
            "lv1": get_code_segment(row["aa"], 1),
            "lv2": get_code_segment(row["bb"], 2),
            "lv3": get_code_segment(row["cc"], 3)
        })

    # 数据集与数据加载器
    train_dataset = AddressDataset(
        data,
        tokenizer,
        label2id,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    return train_dataset


def build_model(bert_name, vocab_size=500, hidden_size=768):
    model = AddressModel(
        bert_name=bert_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )

    return model


def build_optimizer(model, total_steps=None):
    # lr_scheduler = LinearDecayWithWarmup(
    #     learning_rate=5e-5,
    #     total_steps=total_steps,
    #     warmup=0.1
    # )

    optimizer = paddle.optimizer.AdamW(
        learning_rate=5e-5,
        parameters=model.parameters(),
        weight_decay=0.01
    )

    return optimizer


def main():
    epochs = 10
    batch_size = 32
    bert_name = "bert-base-chinese"

    tokenizer = BertTokenizer.from_pretrained(bert_name)

    label_list, label2id, id2label = load_labels("./data/labels.json")

    model = build_model(bert_name, vocab_size=len(label2id), hidden_size=768)

    optimizer = build_optimizer(model)

    train_dataset = load_training_data("./data/areas_full.csv", tokenizer, label2id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=label2id["<pad>"])

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        loss_fn=loss_fn,
        epochs=epochs,
        save_dir="./output/address_model"
    )
    trainer.train()


if __name__ == "__main__":
    main()
