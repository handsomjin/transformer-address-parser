import paddle
from paddle.io import Dataset
from .negative_sample_generator import generate_negative_samples


class AddressDataset(Dataset):
    def __init__(self, label_map, raw_data, tokenizer, label2id, max_src_len=128, max_tgt_len=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.sos_id = label2id["<sos>"]
        self.eos_id = label2id["<eos>"]
        self.pad_id = label2id["<pad>"]
        negatives = generate_negative_samples(label_map, raw_data, seed=42, rule2_prob=0.7)
        self.data = raw_data + negatives
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        labels = sample["labels"].split(",")  # list of str, length=5

        # 编码源文本
        encoded = self.tokenizer(
            text,
            max_seq_len=self.max_src_len,
            pad_to_max_seq_len=True,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        input_ids = encoded["input_ids"]
        token_type_ids = encoded["token_type_ids"]
        attention_mask = encoded["attention_mask"]

        # 将标签转为 ID 序列
        label_ids = [self.label2id.get(l, self.label2id["<unk>"]) for l in labels]
        # 构造 decoder 输入：<sos> + 前四个真实标签（teacher forcing）
        decoder_input = [self.sos_id] + label_ids[:-1]
        # 目标输出：五个标签 + <eos>（可选）
        target = label_ids  # 长度 6，若不需要 eos 可只保留 label_ids

        # 填充到固定长度
        if len(decoder_input) < self.max_tgt_len:
            decoder_input += [self.pad_id] * (self.max_tgt_len - len(decoder_input))
        if len(target) < self.max_tgt_len:
            target += [self.pad_id] * (self.max_tgt_len - len(target))

        return {
            "input_ids": paddle.to_tensor(input_ids, dtype="int64"),
            "token_type_ids": paddle.to_tensor(token_type_ids, dtype="int64"),
            "attention_mask": paddle.to_tensor(attention_mask, dtype="int64"),
            "decoder_input_ids": paddle.to_tensor(decoder_input, dtype="int64"),
            "labels": paddle.to_tensor(target, dtype="int64"),
        }
