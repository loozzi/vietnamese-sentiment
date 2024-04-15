import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore
from torch.nn.functional import softmax
from transformers import AutoTokenizer

from phobert_bilstm_model import PhoBertBiLSTMAttentionModel


def load_model(path, hidden_size, num_classes, dropout, device):
    model = PhoBertBiLSTMAttentionModel(hidden_size, num_classes, dropout).to(device)
    model.load_state_dict(torch.load(path, map_location=device.type))
    model.eval()
    return model


def predict(tokenizer, sequences, max_len, model, device):
    seq_tok = tokenizer(sequences)
    train_ids = torch.tensor(
        pad_sequences(
            seq_tok["input_ids"],
            maxlen=max_len,
            dtype="long",
            value=0,
            truncating="post",
            padding="post",
        )
    )
    train_mask = torch.tensor(
        pad_sequences(
            seq_tok["attention_mask"],
            maxlen=max_len,
            dtype="long",
            value=0,
            truncating="post",
            padding="post",
        )
    )
    with torch.no_grad():
        output = model(train_ids.to(device), train_mask.to(device))
        if device.type == "cuda":
            output = output.to("cpu")
        print(output)
        softmax_output = softmax(output, dim=1)
        predicted_class = torch.argmax(softmax_output, dim=1)

    return predicted_class, softmax_output[:, 0], softmax_output[:, 1]


if __name__ == "__main__":
    hidden_size = 768
    num_classes = 2
    learning_rate = 1e-5
    batch_size = 32
    num_epochs = 5
    dropout = 0.3
    MAX_LEN = 125

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(
        "./models/Bert_samodel-[0.3-125-5].pth",
        hidden_size,
        num_classes,
        dropout,
        device,
    )
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")

    predicted, prob_0, prob_1 = predict(
        tokenizer, ["Tôi yêu bạn"], MAX_LEN, model, device
    )

    # Positive: 0
    # Negative: 1
    for i, p in enumerate(predicted):
        print(
            f"Predicted: {p.item()}, prob_0: {prob_0[i].item()}, prob_1: {prob_1[i].item()}"
        )
