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


def extract_labels_from_json(json_path):
    """
    从行政区划 JSON 中提取标签集合。

    参数:
        json_path (str): JSON 文件路径
        use_code_segment (bool): 是否使用 code 的片段作为标签，
                                 True 时使用 code 的 2n-(2n+1) 位，
                                 False 时使用节点 name（默认）
        levels (tuple): 需要提取的级别，例如 (1,2,3) 表示省、市、区三级
    返回:
        label_set (set): 所有出现过的标签集合（不包含根节点和无效名称）
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    label_map = {}
    label_set = set()

    def get_code_segment(code, level):
        """
        根据 level 截取 code 的对应片段。
        规则：2n ~ 2n+1 位，其中 n = level - 1。
        例如：
            level=1 -> code[0:2]   (省级)
            level=2 -> code[2:4]   (市级)
            level=3 -> code[4:6]   (区级)
        """
        start = 2 * (level - 1)
        end = start + 2
        return f"{level}_{code[start:end]}" if len(code) >= end else None

    def traverse(node, path_labels):
        level = node.get('level', 0)
        code = node.get('code', '')

        if level > 0:
            label = get_code_segment(code, level)
            # 过滤无效标签（如根节点 name 为 "None"）
            if label and label != 'None':
                label_map[node.get('name', '')] = label
                label_set.add(label)

        # 递归处理子节点（注意复制路径列表，避免相互影响）
        children = node.get('children', [])
        for child in children:
            traverse(child, path_labels.copy())

    # 从根节点开始遍历，根节点 level=0 不会被加入
    traverse(data, [])
    return label_set, label_map


def build_label_vocab(json_path, special_tokens=("<pad>", "<sos>", "<eos>", "<unk>")):
    label_set, label_map = extract_labels_from_json(json_path)

    # 添加特殊 token
    for token in special_tokens:
        label_set.add(token)

    label_list = sorted(label_set)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label, label_map


def build_train_dataset(data_path, label2id, tokenizer, label_map, max_src_len=128, max_tgt_len=3):
    # 从文件数据加载
    def load_data_from_file(file_path):
        df = pd.read_csv(file_path, sep=",", header=0)
        data = []

        def get_label(val):
            if pd.isna(val):
                return '<unk>'
            s = str(val).strip()
            return label_map.get(s, '<unk>')

        for _, row in df.iterrows():
            text = row["原始文本"].strip()
            labels = f"{get_label(row['省'])},{get_label(row['市'])},{get_label(row['区'])}"
            data.append({
                "text": text,  # 喂 tokenizer 用「无分隔符」的整串
                "labels": labels,  # 标签仍是按字拆好的 list
                "省": row['省'],
                "市": row['市'],
                "区": row['区']
            })

        return data

    # 数据集与数据加载器
    train_data = load_data_from_file(data_path)
    train_dataset = AddressDataset(
        label_map,
        train_data,
        tokenizer,
        label2id,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )

    # 数据集与数据加载器
    eval_data = load_data_from_file(data_path)
    eval_dataset = AddressDataset(
        label_map,
        eval_data,
        tokenizer,
        label2id,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len
    )
    return train_dataset, eval_dataset


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

    label2id, id2label, label_map = build_label_vocab(json_path)

    model = build_model(bert_name, vocab_size=len(label2id), hidden_size=768)

    optimizer = build_optimizer(model)

    train_dataset, eval_dataset = build_train_dataset("./data/train.csv", label2id, tokenizer, label_map)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    eval_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    loss_fn = nn.CrossEntropyLoss(ignore_index=label2id["<pad>"])

    epochs = 10
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        eval_loader=eval_loader,
        loss_fn=loss_fn,
        epochs=epochs,
        save_dir="./output/address_model"
    )
    trainer.train()


if __name__ == "__main__":
    main()
