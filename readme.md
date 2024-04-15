<div align="center">
	<h1>Phân loại cảm xúc tiếng Việt</h1>
	<!-- Badges -->
	<p>
	<a href="https://github.com/loozzi/vietnamese-sentiment/graphs/contributors">
		<img src="https://img.shields.io/github/contributors/loozzi/vietnamese-sentiment" alt="contributors" />
	</a>
	<a href="">
		<img src="https://img.shields.io/github/last-commit/loozzi/vietnamese-sentiment" alt="last update" />
	</a>
	<a href="https://github.com/loozzi/vietnamese-sentiment/network/members">
		<img src="https://img.shields.io/github/forks/loozzi/vietnamese-sentiment" alt="forks" />
	</a>
	<a href="https://github.com/loozzi/vietnamese-sentiment/stargazers">
		<img src="https://img.shields.io/github/stars/loozzi/vietnamese-sentiment" alt="stars" />
	</a>
	<a href="https://github.com/loozzi/vietnamese-sentiment/issues/">
		<img src="https://img.shields.io/github/issues/loozzi/vietnamese-sentiment" alt="open issues" />
	</a>
	</p>
</div>

# Mô tả

Sử dụng mô hình học máy: phoBERT, BiLSTM, Attention để thực hiện phân loại cảm xúc văn bản tiếng Việt (tích cực/tiêu cực)

# Cài đặt & sử dụng

### Cài đặt (Ubuntu 20.04 - Python 3.x)

```
git clone https://github.com/loozzi/vietnamese-sentiment.git

cd vietnamese-sentiment

pip install -r requirements.txt
```

### Train model

```
python3 train.py
```

#### Input:

```
dropout = <0 -> 1> (0.2 ~ 0.4)
max_len = <0 -> n> (128 | 256)
epoch = <0 -> n> (3~10)
```

### Predict

Theo bộ dữ liệu đi kèm thì: `0-Positive` và `1-Negative`

```
python3 predict.py
```

#### Input

```
Model path: <model-name>-<dropout>-<max_len>-<epoch>.pth
Sentences path: <filename>.txt (1 sentence per line)
```
