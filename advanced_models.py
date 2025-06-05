# advanced_models.py
import torch
import torch.nn as nn
from transformers import BertModel

try:
    from torchcrf import CRF
except ImportError:
    CRF = None
    # 如果需要使用 CRF，请先 pip install torchcrf

class BertCRFSarcasmTagger(nn.Module):
    """
    用于 token‐level 序列标注的 BERT+CRF（输出每个 token 是否为“反讽”指标词）。
    如果你只关心一句话层面的“反讽/非反讽”分类，不需要 CRF，可直接用 BertForSequenceClassification。
    """
    def __init__(self, pretrain_model_name="bert-base-chinese", num_labels=2, dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        if CRF is None:
            raise ImportError("需要先安装 torchcrf：pip install torchcrf")
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        如果传入 labels（shape: [B, L]），则返回 CRF 损失值；
        否则，返回最优标签序列（List[List[int]]]）。
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # (B, L, hidden_size)
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)  # (B, L, num_labels)

        mask = attention_mask.bool()
        if labels is not None:
            # CRF 的 loss 是取负号，因为 torchcrf 返回的是 log‐likelihood
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            return loss
        else:
            best_paths = self.crf.decode(emissions, mask=mask)
            return best_paths


class BertCNNClassifier(nn.Module):
    """
    用于句子分类的 BERT+多通道 CNN。
    """
    def __init__(self, pretrain_model_name="bert-base-chinese", num_classes=2,
                 cnn_out_channels=64, kernel_sizes=(2,3,4), dropout_prob=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrain_model_name)
        hidden_size = self.bert.config.hidden_size  # 通常 768
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=hidden_size, out_channels=cnn_out_channels, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout_prob)
        self.fc = nn.Linear(cnn_out_channels * len(kernel_sizes), num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # last_hidden_state: (B, L, hidden_size)
        x = outputs.last_hidden_state.permute(0, 2, 1)  # 转为 (B, hidden_size, L)
        conv_outs = []
        for conv in self.convs:
            y = torch.relu(conv(x))      # (B, out_ch, L - k + 1)
            y = torch.max(y, dim=2)[0]   # (B, out_ch)
            conv_outs.append(y)
        cat = torch.cat(conv_outs, dim=1)  # (B, out_ch * num_kernels)
        cat = self.dropout(cat)
        logits = self.fc(cat)              # (B, num_classes)
        return logits
