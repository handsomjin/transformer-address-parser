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


def load_vocab_and_config(json_path, special_tokens=("<pad>", "<sos>", "<eos>", "<unk>")):
    label_set, label_map = extract_labels_from_json(json_path)
    code2name = {v: k for k, v in label_map.items()}

    # 添加特殊 token
    for token in special_tokens:
        label_set.add(token)

    label_list = sorted(label_set)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label, code2name


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


def predict(text, model, tokenizer, label2id, id2label, code2name,
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
    label2id, id2label, code2name = load_vocab_and_config(json_path)

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
            addr, model, tokenizer, label2id, id2label, code2name
        )
        print(f"code：{province}{city}{district}")


if __name__ == "__main__":
    main()
