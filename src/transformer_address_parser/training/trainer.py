import paddle
from tqdm import tqdm


def generate_square_subsequent_mask(tgt_len):
    """生成上三角掩码，防止 decoder 看到未来位置"""
    mask = paddle.triu(paddle.ones([tgt_len, tgt_len]) * float('-inf'), diagonal=1)
    return mask


class Trainer:

    def __init__(
            self,
            model,
            optimizer,
            train_loader,  # ← 改：原来是 train_loader
            loss_fn,
            epochs=10,
            save_dir="./ckpt",
            save_by_epoch=True,  # 是否仍每个 epoch 存一份（可选关掉）
    ):
        self.model = model
        self.optimizer = optimizer
        self.epochs = epochs
        self.save_dir = save_dir
        self.save_by_epoch = save_by_epoch

        # 内部包 DataLoader
        self.train_loader = train_loader

        # best 追踪
        self.best_eval_loss = float("inf")

        # 损失函数
        self.loss_fn = loss_fn

    def _compute_loss(self, logits, labels):
        """
        logits: [B, L, V]
        labels: [B, L]
        """
        logits = logits.reshape([-1, logits.shape[-1]])
        labels = labels.reshape([-1])
        return self.loss_fn(logits, labels)

    def train(self):

        # 训练
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch + 1}"):
                input_ids = batch["input_ids"]
                token_type_ids = batch["token_type_ids"]
                attention_mask = batch["attention_mask"]
                decoder_input_ids = batch["decoder_input_ids"]
                labels = batch["labels"]

                # 生成 decoder 因果掩码
                tgt_len = decoder_input_ids.shape[1]
                tgt_mask = generate_square_subsequent_mask(tgt_len)

                # 前向传播
                logits = self.model(input_ids, token_type_ids, attention_mask, decoder_input_ids, tgt_mask=tgt_mask)

                # 计算损失：将 logits 和 labels 展平
                logits = logits.reshape([-1, logits.shape[-1]])
                labels = labels.reshape([-1])
                loss = self.loss_fn(logits, labels)

                # 反向传播
                loss.backward()
                self.optimizer.step()
                self.optimizer.clear_grad()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch + 1} average loss: {avg_loss:.4f}")

        # 保存模型
        paddle.save(self.model.state_dict(), "./outputs/address_model.pdparams")
