import os
import json
import sys

import paddle
import numpy as np
from paddlenlp.transformers import BertTokenizer
from src.transformer_address_parser.models.address_model import AddressModel

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)
script_dir = os.path.dirname(os.path.abspath(__file__))


def load_labels(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        label_list = json.load(f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label_list, label2id, id2label


def load_model(save_dir, vocab_size, hidden_size=768, bert_name="bert-base-chinese"):
    """加载模型参数"""
    model = AddressModel(
        bert_name=bert_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    state_dict = paddle.load(os.path.join(save_dir, "address_model.pdparams"))
    model.set_state_dict(state_dict)
    model.eval()
    return model


def predict(text, model, tokenizer, label2id, id2label,
            max_src_len=128, max_tgt_len=3, device="cpu"):
    """
    对单条地址文本进行预测，返回省、市、区的中文名称。

    假设模型采用自回归生成方式（输入源文本，输出目标标签序列）。
    解码策略：贪心搜索，从 <sos> 开始，逐时间步取概率最大的 token，
    遇到 <eos> 停止或达到 max_tgt_len。
    """
    # 编码源文本
    encoded = tokenizer(
        text,
        max_length=max_src_len,
        padding="max_length",
        truncation=True,
        return_tensors="np",
        return_attention_mask=True,
    )
    input_ids = paddle.to_tensor(encoded["input_ids"], dtype="int64")
    attention_mask = paddle.to_tensor(encoded["attention_mask"], dtype="int64")
    token_type_ids = paddle.to_tensor(encoded["token_type_ids"], dtype="int64")

    # 初始化解码器输入：<sos>
    sos_id = label2id.get("<sos>", None)
    eos_id = label2id.get("<eos>", None)
    pad_id = label2id.get("<pad>", 0)
    if sos_id is None or eos_id is None:
        raise ValueError("label2id 中必须包含 <sos> 和 <eos> 特殊标记")

    # 批量维度为1，构造 decoder_input_ids: [1, 1] 起始为 <sos>
    decoder_input_ids = paddle.full([1, 1], sos_id, dtype="int64")

    # 逐步生成
    for step in range(max_tgt_len):  # 最多生成 max_tgt_len+2 (含 sos 和 eos)
        logits = model(input_ids, token_type_ids, attention_mask, decoder_input_ids)  # [1, seq_len, vocab_size]
        next_token_logits = logits[:, -1, :]  # 最后一个时间步的 logits
        next_token_id = paddle.argmax(next_token_logits, axis=-1).unsqueeze(-1)  # [1, 1]

        # 拼接新 token
        decoder_input_ids = paddle.concat([decoder_input_ids, next_token_id], axis=1)

        # 如果生成了 <eos> 则停止
        if next_token_id.item() == eos_id:
            break

    # 去掉开头的 <sos> 和结尾的 <eos>，得到标签 ID 序列
    pred_ids = decoder_input_ids[0].numpy().tolist()
    if sos_id in pred_ids:
        pred_ids = pred_ids[pred_ids.index(sos_id) + 1:]  # 去掉 <sos>
    if eos_id in pred_ids:
        pred_ids = pred_ids[:pred_ids.index(eos_id)]  # 去掉 <eos>

    # 将 ID 转为标签 code（如 "1_11"），再通过 code2name 转为中文名
    result = []
    for idx in pred_ids:
        label_code = id2label.get(idx, "<unk>")
        if '_' in label_code:
            label_code = label_code.split('_', 1)[1]
        result.append(label_code)

    # 确保输出三个元素（省、市、区），不足补空字符串
    while len(result) < 3:
        result.append("")
    return result[:3]


def main():
    # 配置路径和设备
    save_dir = "./outputs/"
    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)

    # 加载词汇映射和配置
    label_list, label2id, id2label = load_labels("./data/labels.json")

    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 加载模型
    model = load_model(save_dir, vocab_size=len(label2id))

    # 交互式预测
    print("地址解析预测启动，输入 'quit' 退出。")
    while True:
        addr = input("\n请输入地址文本：").strip()
        if addr.lower() in ("quit", "exit", "q"):
            break
        if not addr:
            continue
        province, city, district = predict(
            addr, model, tokenizer, label2id, id2label
        )
        print(f"code：{province}{city}{district}")


if __name__ == "__main__":
    main()
