import paddle
import paddle.nn as nn
import json
from src.transformer_address_parser.runtime.address_model import AddressModel


def load_model(filepath, vocab_size, hidden_size=768, bert_name="bert-base-chinese"):
    """加载模型参数"""
    model = AddressModel(
        bert_name=bert_name,
        vocab_size=vocab_size,
        hidden_size=hidden_size,
    )
    state_dict = paddle.load(filepath)
    model.set_state_dict(state_dict)
    model.eval()
    return model


def load_labels(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        label_list = json.load(f)
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    return label_list, label2id, id2label


# 1. 定义编码器的包装类
class EncoderWrapper(nn.Layer):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model

    def forward(self, input_ids, token_type_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return outputs[0]  # [B, S, H]


def main():

    # 加载词汇映射和配置
    label_list, label2id, id2label = load_labels("../../../data/labels.json")

    address_model = load_model("../../../outputs/address_model.pdparams", vocab_size=len(label2id))
    encoder_wrapper = EncoderWrapper(address_model.bert)

    # 2. 设置输入样例（用于确定动态轴）
    batch_size = 1
    seq_len = 128
    hidden_size = 768

    input_ids = paddle.randint(0, 100, shape=[batch_size, seq_len], dtype='int64')
    token_type_ids = paddle.zeros([batch_size, seq_len], dtype='int64')
    attention_mask = paddle.ones([batch_size, seq_len], dtype='int64')

    # 3. 导出 ONNX
    paddle.onnx.export(
        encoder_wrapper,
        './onnx_models/encoder',
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype='int64', name='input_ids'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64', name='token_type_ids'),
            paddle.static.InputSpec(shape=[None, None], dtype='int64', name='attention_mask'),
        ],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'seq_len'},
            'token_type_ids': {0: 'batch_size', 1: 'seq_len'},
            'attention_mask': {0: 'batch_size', 1: 'seq_len'},
            # 输出 memory 的 batch 和 seq_len 也是动态的
        },
        opset_version=12,
        enable_onnx_checker=True,
    )
    print("编码器 ONNX 导出成功")


if __name__ == '__main__':
    main()
