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


class DecoderWrapper(nn.Layer):
    def __init__(self, decoder, decoder_embedding, output_layer):
        super().__init__()
        self.decoder = decoder
        self.decoder_embedding = decoder_embedding
        self.output = output_layer

    def forward(self, decoder_input_ids, memory):
        # memory shape: [B, src_len, H]
        decoder_input = self.decoder_embedding(decoder_input_ids)  # [B, tgt_len, H]
        decoder_output = self.decoder(decoder_input, memory, tgt_mask=None)
        logits = self.output(decoder_output)  # [B, tgt_len, vocab_size]
        return logits


def main():
    batch_size = 1
    seq_len = 128
    hidden_size = 768
    # 加载词汇映射和配置
    label_list, label2id, id2label = load_labels("../../../data/labels.json")

    address_model = load_model("../../../outputs/address_model.pdparams", vocab_size=len(label2id))
    decoder_wrapper = DecoderWrapper(address_model.decoder, address_model.decoder_embedding, address_model.output)

    # 设置样例
    vocab_size = len(label2id)  # 从你的 label2id 获取
    tgt_len = 3  # 解码最大步数（可动态）

    decoder_input_ids_sample = paddle.full([batch_size, 1], fill_value=label2id['<sos>'], dtype='int64')
    memory_sample = paddle.randn([batch_size, seq_len, hidden_size], dtype='float32')

    paddle.onnx.export(
        decoder_wrapper,
        './onnx_models/decoder',
        input_spec=[
            paddle.static.InputSpec(shape=[None, None], dtype='int64', name='decoder_input_ids'),
            paddle.static.InputSpec(shape=[None, None, hidden_size], dtype='float32', name='memory'),
        ],
        dynamic_axes={
            'decoder_input_ids': {0: 'batch_size', 1: 'tgt_len'},
            'memory': {0: 'batch_size', 1: 'src_len'},
            # 输出 logits 的 batch 和 tgt_len 也是动态
        },
        opset_version=12,
        enable_onnx_checker=True,
    )
    print("解码器 ONNX 导出成功")


if __name__ == '__main__':
    main()
