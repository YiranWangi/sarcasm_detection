import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class T5SarcasmClassifier:
    def __init__(self, model_name="google/mt5-base", device="cpu"):
        """
        model_name:
          - 训练时如果用的就是 "google/mt5-base"，保存下来后（例如 best_t5_model/），
            预测时这里就要传相同的目录或名称。
        device:
          - "mps"（Apple M1/M2 上） 或 "cpu" / "cuda"。
        """
        # 1) 自动下载/加载一个 MT5 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

        # 2) 加载模型
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

        # 3) 设备（MPS / CPU / CUDA）选择
        self.device = device
        if self.device:
            self.model.to(self.device)

        # 4) 定义一个“从 <extra_id_0> 到 中文标签” 的映射
        #    训练时你让 T5 生成 “反讽”/“非反讽”，其实在 tokenizer.decode 之前
        #    它会先生成 <extra_id_0> 这个占位符，所以我们需要把占位符替换成人工定义的标签。
        self.id2label = {
            "<extra_id_0>": "反讽",
            "<extra_id_1>": "非反讽"
        }

    def predict(self, sentence: str, max_length: int = 64, num_beams: int = 4):
        """
        对单句 sentence 做推理，返回 "反讽" 或 "非反讽"。
        之所以在这里需要“搬到 CPU 上做 generate”，是因为 MPS 后端下 generate() 有时会报错。
        """
        # 1. 构造 prompt
        prompt = f"classify sarcasm: {sentence}"

        # 2. Tokenize 并先 send 到 self.device (比如 MPS)
        encoding = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=max_length
        ).to(self.device)

        self.model.eval()
        with torch.no_grad():
            # —— 把模型搬到 CPU，以保证 generate 不会在 MPS 下崩掉 ——
            self.model.to("cpu")
            input_ids_cpu = encoding["input_ids"].to("cpu")
            attention_mask_cpu = encoding["attention_mask"].to("cpu")

            # —— 在 CPU 上生成 (generate) ——
            #    这里直接让它只生成 1 个 token (因为你在训练时 labels 也只有 "<extra_id_0>" 或 "<extra_id_1>" 两种情况)
            generated_ids_cpu = self.model.generate(
                input_ids=input_ids_cpu,
                attention_mask=attention_mask_cpu,
                max_length=2,  # 只让它生成 "<extra_id_x>" + 可能的 "</s>"
                num_beams=num_beams,
                early_stopping=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

            # —— 把模型再搬回 self.device (MPS)，以便下次继续使用 ——
            if self.device:
                self.model.to(self.device)

        # 3. 生成的 generated_ids_cpu[0] 可能是类似 [extra_id_0_id, eos_id] 这样的序列
        pred_token = self.tokenizer.decode(
            generated_ids_cpu[0],
            skip_special_tokens=False,  # 先保留 "<extra_id_0>"
            clean_up_tokenization_spaces=True
        ).strip()

        # pred_token 可能是 "<extra_id_0>" 或 "<extra_id_1>" 或者 "<extra_id_0></s>" 之类
        # 我们做一次替换，把 "<extra_id_0>" 变成 "反讽"，把 "<extra_id_1>" 变成 "非反讽"。
        for key, val in self.id2label.items():
            if key in pred_token:
                # 一旦找到占位符，就立刻返回相应的标签
                return val

        # 如果连 "<extra_id_0>" / "<extra_id_1>" 都没生成，说明有点奇怪，
        # 那就 fallback 返回“非反讽”
        return "非反讽"

    def save(self, path: str):
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def load(self, path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = MT5ForConditionalGeneration.from_pretrained(path)
        if self.device:
            self.model.to(self.device)
