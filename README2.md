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

## 手撕的函数
```
def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
	
class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
	
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model, num_heads, positional_encoder):
	
	def forward(self, x, token_positions=None):

class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff):

class TransformerBlock(nn.Module):

    def __init__(self, d_model, num_heads, d_ff, positional_encoder):
	
	def forward(self, in_features, token_positions):

class TransformerLM(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 context_length: int,
                 d_model: int,
                 num_layers: int,
                 num_heads: int,
                 d_ff: int,
                 rope_theta: float):
	def forward(self, x):
	def generate(self, idx, max_new_tokens, temperature=1.0, top_p=None):

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
	def forward(self, x: torch.Tensor) -> torch.Tensor:

	def cross_entropy(inputs: Float[Tensor, " batch_size vocab_size"], targets: Int[Tensor, " batch_size"]) -> Float[
    Tensor, ""]:

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:

def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):

def scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... values d_v"],
        mask: Float[Tensor, " ... queries keys"] | None = None,
) -> Float[Tensor, " ... queries d_v"]:

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
	def step(self, closure: Optional[Callable] = None):

def get_lr_cosine_schedule(
        it: int,
        max_learning_rate: float,
        min_learning_rate: float,
        warmup_iters: int,
        cosine_cycle_iters: int,
):

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):

	def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
	def _bpe_merge(self, word_bytes):
	def encode(self, text: str) -> list[int]:
	def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
	def decode(self, ids: list[int]) -> str:

def train_bpe(
        input_path: str | os.PathLike,
        vocab_size: int,
        special_tokens: list[str],
        **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
```
### generate

```
python scripts/generate.py --vocab_file /root/autodl-tmp/data/tokenizer/vocab.json --merges_file /root/autodl-tmp/data/tokenizer/merges.txt
```
