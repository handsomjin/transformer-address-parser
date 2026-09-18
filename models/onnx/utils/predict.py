import os
import json
import sys
import onnxruntime as ort
import paddle
import numpy as np
from paddlenlp.transformers import BertTokenizer
import io

sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

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


def predict(text, tokenizer, label2id, id2label, encoder_session, decoder_session, max_tgt_len=3):
    # 初始化解码器输入：<sos>
    sos_id = label2id.get("<sos>", None)
    eos_id = label2id.get("<eos>", None)
    if sos_id is None or eos_id is None:
        raise ValueError("label2id 中必须包含 <sos> 和 <eos> 特殊标记")

    encoded = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="np",  # 直接返回 numpy 数组
        return_attention_mask=True,
    )

    input_ids_np = encoded["input_ids"]  # shape: [1, 128]
    token_type_ids_np = encoded["token_type_ids"]  # shape: [1, 128]
    attention_mask_np = encoded["attention_mask"]  # shape: [1, 128]

    # 编码
    enc_inputs = {
        'input_ids': input_ids_np,
        'token_type_ids': token_type_ids_np,
        'attention_mask': attention_mask_np,
    }
    memory = encoder_session.run(None, enc_inputs)[0]  # [1, src_len, 768]

    # 解码
    decoder_input_ids = np.array([[sos_id]], dtype=np.int64)
    for step in range(max_tgt_len):
        dec_inputs = {
            'decoder_input_ids': decoder_input_ids,
            'memory': memory,
        }
        logits = decoder_session.run(None, dec_inputs)[0]  # [1, cur_len, vocab_size]
        next_token_id = np.argmax(logits[0, -1, :])
        next_token_array = np.array([[next_token_id]], dtype=np.int64)
        decoder_input_ids = np.concatenate([decoder_input_ids, next_token_array], axis=1)

    # 去掉开头的 <sos> 和结尾的 <eos>，得到标签 ID 序列
    pred_ids = decoder_input_ids[0].tolist()
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
    device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
    paddle.set_device(device)

    # 加载词汇映射和配置
    label_list, label2id, id2label = load_labels("../../data/labels.json")

    # 加载 tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 创建 session
    encoder_session = ort.InferenceSession('../../models/onnx/model/v1/encoder.onnx')
    decoder_session = ort.InferenceSession('../../models/onnx/model/v1/decoder.onnx')

    # 交互式预测
    print("地址解析预测启动，输入 'quit' 退出。")
    while True:
        addr = input("\n请输入地址文本：").strip()
        if addr.lower() in ("quit", "exit", "q"):
            break
        if not addr:
            continue
        province, city, district = predict(
            addr, tokenizer, label2id, id2label, encoder_session, decoder_session
        )
        print(f"code：{province}{city}{district}")


if __name__ == "__main__":
    main()
