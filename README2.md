# CS336 Spring 2025 Assignment 1: Basics

2026-2-10 ~ 2026-2-11 实现 CausalMultiHeadSelfAttention 和 transformerblock 和 transformerlm 类

2026-2-11 实现 5 training loop 和 transformerblock 的generate

2026-2-12 获取数据集（必须在语句中加 -Outfile 明确文件命名），成功训练bpe，存储在 data/tokenizer_tiny 中

2026-2-13 【数据准备】阶段 调用训练好的tokenizer，将 TinyStoriesV2-GPT4-train.txt 和 TinyStoriesV2-GPT4-valid.txt 都转化成bin

## 代码理论基础
在jupyterLab运行时，要
```
pip install -e .
```

### tokenization

#### 1 训练 bpe
获取数据
```
mkdir -p data
cd data
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt -OutFile TinyStoriesV2-GPT4-train.txt
wget https://hf-mirror.com/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt -OutFile TinyStoriesV2-GPT4-valid.txt
cd ..
```

需要运行 `scripts/train_tokenizer.py`

```
python scripts/train_tokenizer.py --input data/TinyStoriesV2-GPT4-train.txt --vocab_size 10000 --out_dir data/tokenizer
```

执行 `2.5 train_bpe_tinystories` 这个任务，输出 `vocab`, `merges`。

#### 2 编码
需要运行 `scripts/prepare_data.py`, encode these sampled documents into integerID

```
python scripts/prepare_data.py --input_file data/TinyStoriesV2-GPT4-train.txt --vocab_file data/tokenizer/vocab.json --merges_file data/tokenizer/merges.txt --out_bin data/preprocessed/TinyStoriesV2-GPT4-train.bin
```

```
python scripts/prepare_data.py --input_file data/TinyStoriesV2-GPT4-valid.txt --vocab_file data/tokenizer/vocab.json --merges_file data/tokenizer/merges.txt --out_bin data/preprocessed/TinyStoriesV2-GPT4-valid.bin
```

执行 `2.7 tokenizer_experiments` 输出.bin文件

关键代码：`best_pair = max(pair_frequency, key=lambda x: (pair_frequency[x], x))`

先比第一个元素（频率）如果相同，再比第二个元素（pair 本身）

### train

```
python scripts/train.py --train_data data/preprocessed/TinyStoriesV2-GPT4-train.bin --val_data data/preprocessed/TinyStoriesV2-GPT4-valid.bin --out_dir out/tiny_stories_v2_final --vocab_size 10000 --context_length 256 --d_model 512 --num_layers 4 --num_heads 16 --d_ff 1344 --batch_size 128 --max_iters 10000 --learning_rate 5e-4 --warmup_iters 1000 --cosine_cycle_iters 10000 --weight_decay 0.1 --betas 0.9 0.95 --log_interval 50 --eval_interval 500 --save_interval 2500
```

### generate

```
python scripts/generate.py --vocab_file /root/autodl-tmp/data/tokenizer/vocab.json --merges_file /root/autodl-tmp/data/tokenizer/merges.txt
```