import os
import re

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformers import AdamW, AutoTokenizer

from phobert_bilstm_model import PhoBertBiLSTMAttentionModel
from vncorenlp import VnCoreNLP


def load_data(base_dir, path, test=False):
    rdrsegmenter = VnCoreNLP(
        base_dir + "/vncorenlp/VnCoreNLP-1.1.1.jar",
        annotators="wseg",
        max_heap_size="-Xmx500m",
    )
    ids, sentences, labels = [], [], []
    with open(base_dir + path, "r") as f:
        data = f.read().strip()
        if test:
            data = re.findall('test_[\s\S]+?"\n[01]\n\n', data)
        else:
            data = re.findall('train_[\s\S]+?"\n[01]\n\n', data)
        for sample in data:
            splits = sample.strip().split("\n")

            id = splits[0]
            label = int(splits[-1])
            text = " ".join(splits[1:-1])[1:-1]
            text = rdrsegmenter.tokenize(text)
            text = " ".join([" ".join(x) for x in text])

            ids.append(id)
            sentences.append(text)
            if not test:
                labels.append(label)
    if test:
        return ids, sentences
    else:
        return ids, sentences, labels


def preprocess_data(sentences, labels, tokenizer, max_len):
    input_ids, attention_masks = [], []
    encoding = tokenizer(sentences, truncation=True, padding=True, return_tensors="pt")

    input_ids = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    def _pad_sequences(data):
        return pad_sequences(
            data,
            maxlen=max_len,
            dtype="long",
            value=0,
            truncating="post",
            padding="post",
        )

    input_ids = torch.tensor(_pad_sequences(input_ids))
    attention_masks = torch.tensor(_pad_sequences(attention_mask))
    labels = torch.tensor(labels)

    return input_ids, attention_masks, labels


def train_epoch(data_loader, model, optimizer, criterion, device):
    running_loss = 0
    total_preds = 0
    correct_preds = 0
    pbar = tqdm(data_loader)
    for batch in pbar:
        input_ids, attention_mask, labels = batch
        input_ids, attention_mask, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            labels.to(device),
        )
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        pred = torch.argmax(logits, dim=1)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        correct_preds += (pred == labels).sum().item()
        total_preds += len(labels)
        pbar.set_description(f"acc {correct_preds / total_preds:.4f}")

    return running_loss


def test_epoch(data_loader, model, optimizer, criterion, device):
    val_loss = 0.0
    correct_preds = 0
    total_preds = 0
    # Positive = 0
    # Negative = 1
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    with torch.no_grad():
        pbar = tqdm(data_loader)
        for batch in pbar:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                labels.to(device),
            )
            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            # test lấy các chỉ số
            for i in range(len(predicted)):
                if predicted[i].item() == 0:
                    if labels[i] == 0:
                        correct_preds += 1
                        TP += 1
                    else:
                        FP += 1
                if predicted[i].item() == 1:
                    if labels[i] == 1:
                        correct_preds += 1
                        TN += 1
                    else:
                        FN += 1
            #
            # correct_preds += (predicted == labels).sum().item() test lấy các chỉ số
            total_preds += labels.size(0)
            pbar.set_description(f"test acc {correct_preds / total_preds:.4f}")

    return val_loss, correct_preds, total_preds, TP, TN, FP, FN


def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=5):
    # train model
    for epoch in range(num_epochs):
        model.train()
        running_loss = train_epoch(train_loader, model, optimizer, criterion, device)

        model.eval()
        # test lấy các chỉ số
        val_loss, correct_preds, total_preds, TP, TN, FP, FN = test_epoch(
            val_loader, model, optimizer, criterion, device
        )
        recall_p = TP / (FN + TP)
        precision_p = TP / (TP + FP)
        F1_p = (2 * recall_p * precision_p) / (recall_p + precision_p)
        recall_n = TN / (FP + TN)
        precision_n = TN / (TN + FN)
        F1_n = (2 * recall_n * precision_n) / (recall_n + precision_n)
        F1 = ((TN + FP) * F1_n + (TP + FN) * F1_p) / (total_preds)

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct_preds / total_preds * 100

        print(
            f"Epoch {epoch + 1}/{num_epochs}:",
            f"Train Loss: {avg_train_loss:.4f},",
            f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%,",
            f"F1_score: {F1:.2f}",
        )


def download_vncorenlp():
    print("Downloading VnCoreNLP...")
    os.system("mkdir -p vncorenlp/models/wordsegmenter")
    os.system(
        "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.1.1.jar"
    )
    os.system(
        "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab"
    )
    os.system(
        "wget https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr"
    )
    os.system("mv VnCoreNLP-1.1.1.jar vncorenlp/")
    os.system("mv vi-vocab vncorenlp/models/wordsegmenter/")
    os.system("mv wordsegmenter.rdr vncorenlp/models/wordsegmenter/")


if __name__ == "__main__":
    # define constants
    base_dir = "."
    train_path = "/data/train.crash"
    hidden_size = 768
    num_classes = 2
    learning_rate = 1e-5
    batch_size = 32
    num_epochs = 3
    dropout = 0.15
    MAX_LEN = 256

    download_vncorenlp()

    # load train data
    ids, sentences, labels = load_data(base_dir, train_path)
    train_sentences, val_sentences, train_labels, val_labels = train_test_split(
        sentences, labels, test_size=0.2, random_state=42
    )

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
    train_ids, train_attention_masks, train_labels = preprocess_data(
        train_sentences, train_labels, tokenizer, MAX_LEN
    )
    val_ids, val_attention_masks, val_labels = preprocess_data(
        val_sentences, val_labels, tokenizer, MAX_LEN
    )

    train_dataset = TensorDataset(train_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_ids, val_attention_masks, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # define model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PhoBertBiLSTMAttentionModel(hidden_size, num_classes, dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    # train model
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)

    # save model
    output_path = f"/model-{dropout}-{MAX_LEN}-{num_epochs}.pth"
    torch.save(model.state_dict(), base_dir + output_path)
    print(f"Model saved at {base_dir + output_path}")
