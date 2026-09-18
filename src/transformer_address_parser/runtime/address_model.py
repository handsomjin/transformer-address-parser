import paddle.nn as nn
from paddlenlp.transformers import BertModel


class AddressModel(nn.Layer):

    def __init__(
            self,
            bert_name="bert-base-chinese",
            vocab_size=500,
            hidden_size=768,
    ):
        super().__init__()

        # 1. Encoder
        self.bert = BertModel.from_pretrained(bert_name)

        # 2. Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=2,
        )

        # 3. Decoder token embedding
        self.decoder_embedding = nn.Embedding(
            vocab_size,
            hidden_size
        )

        # 4. 输出层
        self.output = nn.Linear(
            hidden_size,
            vocab_size
        )

    def encode(self, input_ids, token_type_ids, attention_mask):
        """仅执行编码器部分，返回 memory"""
        bert_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        return bert_output[0]  # [B, src_len, 768]

    def forward(
            self,
            input_ids=None,
            token_type_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            tgt_mask=None,
            memory=None  # 新增参数，允许外部传入编码结果
    ):
        # =========================
        # Encoder（如果未提供 memory 则计算）
        # =========================
        if memory is None:
            memory = self.encode(input_ids, token_type_ids, attention_mask)

        # =========================
        # Decoder input
        # =========================
        decoder_input = self.decoder_embedding(decoder_input_ids)

        # =========================
        # Decoder
        # =========================
        decoder_output = self.decoder(decoder_input, memory, tgt_mask=tgt_mask)

        logits = self.output(decoder_output)
        return logits
