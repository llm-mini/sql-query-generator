# LLM Fine-Tuning — Personal Learning Guide

> Use this document to continue learning across sessions. Each section is self-contained.

---

## 1. What is Fine-Tuning?

Taking a pre-trained model and continuing to train it on a smaller, task-specific dataset so it becomes better at a particular job — without training from scratch.

**Why not train from scratch?**
Pre-trained models already understand language. Fine-tuning just redirects that knowledge toward your specific task. Far cheaper, faster, and requires much less data.

---

## 2. Types of Fine-Tuning

### 2.1 Full Fine-Tuning
All model weights are updated during training.
- Most powerful but requires significant GPU memory
- Practical only for smaller models (GPT-2, DistilBERT) or large infrastructure
- **Project 1 uses this method**

### 2.2 Parameter-Efficient Fine-Tuning (PEFT)
Only a small subset of weights are trained. The rest are frozen.

#### Adapters (2019 — Houlsby et al.)
Small bottleneck modules inserted *inside* each transformer layer.
```
Input → Attention → [Adapter] → FFN → [Adapter] → Output
```
Each adapter: Linear DOWN (compress) → Activation → Linear UP (expand) → Residual
- Slight inference overhead because extra layers stay permanently
- Mostly superseded by LoRA today

#### LoRA — Low-Rank Adaptation (2021 — Microsoft)
Does not insert new layers. Instead decomposes weight updates into two small matrices A and B running *parallel* to frozen weights.
```
New W = W (frozen) + A × B
A: 768×8    B: 8×768    rank r=8
Params: 12,288 vs original 589,824
```
**Why two matrices and not one?**
A single matrix C added to W must be the same shape as W — so you save nothing. A×B gives a correctly shaped result through two small matrices. During training gradients flow through small A and B (cheap). After training A×B can be merged back into W — zero inference overhead.

- **Project 2 uses this method**

#### QLoRA (2023 — University of Washington)
LoRA + 4-bit quantization of the base model.
- Base model weights: 4-bit (frozen, quantized) → saves memory
- LoRA A, B matrices: 16-bit (trainable) → preserves quality
- Uses NF4 quantization + double quantization
- **Project 3 uses this method**

### 2.3 Instruction Fine-Tuning
Not a separate technique — describes the *type of data* used. Training on prompt→response pairs to teach the model to follow instructions.
- Uses LoRA or QLoRA as the engine under the hood
- Data format: Alpaca-style (instruction, input, output)
- **Project 4 uses this method**

### 2.4 RLHF — Reinforcement Learning from Human Feedback
Uses human preference rankings to shape model behavior via a reward model.
- Complex — involves training a separate reward model
- How ChatGPT and Claude were trained
- **Project 5 touches this indirectly via DPO**

### 2.5 DPO — Direct Preference Optimization
Simpler alternative to RLHF. Give the model (good response, bad response) pairs and it learns to prefer the good one.
- No separate reward model needed
- **Project 5 uses this method** — chains onto Project 4's output

---

## 3. Other PEFT Methods (Reference)

### Prompt Tuning
Prepend a few learnable virtual token vectors to the input. Model weights completely frozen.
```
Normal:        "Classify this review: Great product!"
Prompt tuning: [T1][T2][T3] "Classify this review: Great product!"
```
- T1, T2, T3 are not real words — learned embedding vectors
- Works well only on very large models (10B+)

### Prefix Tuning
Like prompt tuning but prepends learnable vectors at *every* transformer layer, not just the input.
- More expressive than prompt tuning
- Still fully frozen base model

### BitFit — Bias-only Fine-Tuning
Freeze everything except bias vectors across the whole model.
```
output = W · input + b
              ↑ only this trains
```
- ~0.09% of parameters trained
- Surprisingly effective for tasks similar to pretraining
- Works best for classification on text

---

## 4. Key Terms Glossary

| Term | Explanation |
|---|---|
| **Pre-trained model** | Starting point already trained on billions of tokens (GPT, LLaMA, Mistral) |
| **Base model** | Predicts next tokens — no instruction following |
| **Instruct model** | Base model already fine-tuned to follow instructions |
| **Parameters** | Learned weights inside the neural network. 7B = 7 billion weights |
| **Epochs** | Number of times the model sees the full training dataset |
| **Overfitting** | Model memorizes training data instead of generalizing. Train loss falls, val loss rises |
| **Loss** | Number measuring how wrong the model is. Training minimizes this |
| **Learning rate** | Size of each weight update step. Too high = unstable, too low = no learning |
| **Warmup steps** | Learning rate starts near zero and rises gradually for N steps. Prevents unstable early updates |
| **Weight decay** | Regularization that penalizes large weights. Reduces overfitting |
| **fp16** | 16-bit training instead of 32-bit. Halves VRAM usage, faster on T4 |
| **Batch size** | Number of examples processed together before a weight update |
| **Tokenization** | Converting text into integer IDs the model understands |
| **pad_token** | Special token used to make all sequences the same length in a batch |
| **eos_token** | End-of-sequence token. Tells the model where a sequence ends |
| **labels** | Copy of input_ids in causal LM training. Model predicts next token at every position |
| **Quantization** | Storing weights in fewer bits (e.g. 4-bit) to save memory with acceptable quality loss |
| **NF4 quantization** | 4-bit quantization with non-uniform buckets denser near zero, matching the normal distribution of neural network weights |
| **Double quantization** | Quantizing the quantization constants themselves to save additional memory |
| **Rank (r) in LoRA** | Inner dimension of A and B matrices. Lower = fewer params, less expressive. Higher = more params, more expressive |
| **VRAM** | GPU memory. The main bottleneck for fine-tuning large models |
| **Checkpoint** | Saved model state at a specific training step. Allows resuming or loading best epoch |
| **Exact match** | Strict evaluation metric — output must match gold answer character-for-character |
| **Alpaca-style dataset** | Standard instruction fine-tuning format with three fields: instruction, input (optional), output |
| **Schema (text-to-SQL)** | Table and column definitions given to the model so it knows what exists in the database |
| **load_best_model_at_end** | HF Trainer flag that loads the checkpoint with lowest val loss after all epochs finish |
| **remove_columns** | Drops original string columns after tokenization so only tensors remain for the Trainer |
| **mlm=False** | Tells DataCollator this is causal LM (predict next token), not masked LM like BERT |
| **resize_token_embeddings** | Expands model's embedding matrix when new tokens are added to the tokenizer |
| **greedy decoding** | Model always picks the highest probability next token. Deterministic, no randomness |
| **repetition_penalty** | Penalizes the model for repeating tokens it already generated |

---

## 5. Model Reference

| Model | Params | Best for |
|---|---|---|
| GPT-2 | 117M | Learning — full fine-tuning on CPU/free GPU |
| GPT-2 Medium | 354M | Project 1 — full fine-tuning |
| Phi-3-mini | 3.8B | Project 4 — instruction fine-tuning, very capable for size |
| Mistral-7B | 7B | Projects 2 & 3 — LoRA / QLoRA |
| LLaMA-3-8B | 8B | Projects 2 & 3 — LoRA / QLoRA |

---

## 6. GPU Requirements

| Method | Minimum VRAM | Recommended |
|---|---|---|
| Full fine-tune (GPT-2) | 4GB | Free Colab T4 (15GB) |
| LoRA (7B model) | 8GB | T4 |
| QLoRA (7B model) | 6GB | T4 |
| QLoRA (70B model) | 24GB | A100 |

---

## 7. Project Roadmap

### Project 1 — SQL Query Generator ← IN PROGRESS
- **Method:** Full fine-tuning
- **Model:** GPT-2 Medium (354M)
- **Dataset:** xlangai/spider (HF)
- **Task:** Natural language question → SQL query
- **Metric:** Exact match accuracy
- **Key learning:** Full fine-tuning pipeline, overfitting, schema importance

**Progress so far:**
- [x] Environment setup (Cells 1–6)
- [x] Data loading and exploration (Cells 7–11)
- [x] Tokenization and preprocessing (Cells 12–15)
- [x] Training config and model init (Cells 16–17)
- [x] Training run — observed overfitting (Cell 18)
- [x] Loaded best checkpoint (checkpoint-875, val loss 1.410)
- [x] Inference and evaluation (Cells 21–23)
- [ ] Schema-aware retraining (Cells 26–28) ← **next step**
- [ ] Push to Hugging Face Hub

**Key lessons learned:**
1. Without schema in prompt, model hallucinates table/column names
2. GPT-2 Medium overfits on Spider (7K examples too small for 354M params)
3. `load_best_model_at_end=True` auto-selects best checkpoint
4. Exact match is very strict — even casing differences = wrong
5. Re-runs overwrite checkpoints if `output_dir` is the same

**Pending fix — schema-aware training:**
```
Format: Schema: table(col1, col2) | table2(col1)\nQuestion: ...\nSQL: ...
MAX_LENGTH: 256 (increased from 128)
output_dir: "./sql-gpt2-v2"
```

---

### Project 2 — Sentiment-aware News Summarizer
- **Method:** Full fine-tuning
- **Model:** GPT-2 (117M)
- **Dataset:** financial_phrasebank (HF)
- **Task:** Financial headline → sentiment label + summary
- **Metric:** Sentiment accuracy + BLEU
- **Status:** Not started

---

### Project 3 — LoRA on Mistral-7B
- **Method:** LoRA (PEFT)
- **Extra libs:** `peft`
- **Status:** Not started

---

### Project 4 — Instruction Fine-Tuning with QLoRA
- **Method:** QLoRA + instruction data
- **Model:** Phi-3-mini
- **Dataset:** Alpaca-style
- **Extra libs:** `peft`, `bitsandbytes`, `trl`
- **Status:** Not started

---

### Project 5 — DPO Alignment
- **Method:** DPO
- **Model:** Output of Project 4
- **Extra libs:** `trl` (DPOTrainer)
- **Status:** Not started — chains onto Project 4

---

## 8. Standard Setup (All Projects)

```python
# Base install
!pip install transformers datasets accelerate huggingface_hub

# PEFT projects (3, 4, 5)
!pip install peft bitsandbytes trl

# Imports
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
from huggingface_hub import login

# Auth
login(token="hf_xxxx")

# Device check
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
```

---

## 9. Workflow Template (Reusable for Every Project)

```
1. Setup      → install libs, verify GPU, HF login
2. Data       → load dataset, inspect examples, check length distribution
3. Preprocess → format prompt, tokenize, set max_length, remove_columns
4. Configure  → TrainingArguments, learning rate, epochs, batch size
5. Train      → Trainer.train(), watch loss curve for overfitting
6. Evaluate   → load best checkpoint, run inference, measure metric
7. Publish    → push model + tokenizer + model card to HF Hub
```

---

*Last updated: Project 1 in progress — schema-aware retraining pending*
