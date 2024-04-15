import torch
import torch.nn as nn
from transformers import AutoModel


class Attention(nn.Module):
    def __init__(self, hidden_size, dropout=0.5) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, inputs):
        attention_scores = torch.matmul(inputs, self.weight).squeeze(-1)
        attention_weights = torch.softmax(attention_scores, dim=-1)
        dropout_output = self.dropout(attention_weights)
        weighted_sum = torch.matmul(
            inputs.transpose(1, 2), dropout_output.unsqueeze(-1)
        ).squeeze(-1)
        return weighted_sum


class PhoBertBiLSTMAttentionModel(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1) -> None:
        super().__init__()
        self.phobert = AutoModel.from_pretrained("vinai/phobert-base-v2")
        self.bilstm = nn.LSTM(
            bidirectional=True,
            input_size=hidden_size,
            hidden_size=hidden_size // 2,
            batch_first=True,
        )
        self.attention = Attention(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        phobert_output = self.phobert(
            input_ids=input_ids, attention_mask=attention_mask
        )[0]
        lstm_output, _ = self.bilstm(phobert_output)
        attention_output = self.attention(lstm_output)
        dropout_output = self.dropout(attention_output)
        logits = self.fc(dropout_output)
        return logits
