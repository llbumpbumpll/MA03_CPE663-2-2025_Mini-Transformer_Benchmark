# CPE663 Mini Transformer Benchmark

## Course Information

**CPE 663: Special Topic III**  
**Multilingual NLP and Low-Resource NLP**

**Major Assignment 3**  
**Mini Transformer Benchmark**  
**A Small Transformer Encoder for Synthetic Sequence Classification**

**Member**  
**67070701603 Kantapat Suwannahong**

This project is submitted as part of Major Assignment 3: Mini Transformer Benchmark for the course CPE 663: Special Topic III: Multilingual NLP and Low-Resource NLP.  
Semester 2, Academic Year 2025

This project implements a mini Transformer encoder from scratch for the CPE663 major assignment. The task is binary sequence classification: predict whether the first non-padding token appears again in the second half of the valid sequence.

The implementation uses PyTorch tensors and common layers such as `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and `nn.Dropout`, but it does not use `torch.nn.Transformer`, `torch.nn.MultiheadAttention`, Hugging Face models, or pretrained Transformers.

## Project Structure

```text
MJA3/
|-- data/
|   |-- train.csv
|   |-- validation.csv
|   `-- test.csv
|-- NO_USE/
|   |-- CPE663-Major-Assignment-Transformer-Lab-Walkthrough.docx
|   `-- report backup files
|-- runs/
|   |-- benchmark_results.csv
|   |-- benchmark_summary.json
|   |-- training_curves.png
|   |-- training_loss_log_curves.png
|   `-- <variant folders with history/result/model files>
|-- benchmark.py
|-- data.py
|-- model.py
|-- README.md
|-- report.pdf
|-- requirements.txt
|-- train.py
`-- utils.py
```

## Task Description

Each input sequence contains tokens from the vocabulary `PAD`, `A`, `B`, `C`, and `D`. The valid sequence length is between 6 and 20 tokens, and shorter sequences are padded to length 20.

The label is defined as follows:

- `1` if the first non-padding token appears again in the second half of the valid sequence
- `0` otherwise

Example:

```text
Sequence: [A, C, B, D, A]
First token: A
Second half: [B, D, A]
Label: 1
```

## Files

- `data/`: dataset folder containing `train.csv`, `validation.csv`, and `test.csv`.
- `data.py`: CSV dataset loader and dataloader helpers.
- `model.py`: from-scratch mini Transformer encoder components.
- `train.py`: single-model training and evaluation script.
- `benchmark.py`: required benchmark across four model variants and figure generation.
- `utils.py`: reproducibility and metric helpers.
- `NO_USE/`: files kept for reference only and not used in the main workflow.
- `report.pdf`: report submitted for the assignment.

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train one model:

```powershell
python train.py --train-csv data/train.csv --validation-csv data/validation.csv --test-csv data/test.csv --epochs 10 --num-heads 4 --num-layers 1
```

Run the full benchmark:

```powershell
python benchmark.py --train-csv data/train.csv --validation-csv data/validation.csv --test-csv data/test.csv --epochs 10 --quiet
```

For a quicker smoke test:

```powershell
python benchmark.py --train-csv data/train.csv --validation-csv data/validation.csv --test-csv data/test.csv --epochs 1 --quiet
```

## Benchmark Variants

- `A_pos_1head_1layer`: positional encoding, 1 attention head, 1 encoder layer.
- `B_pos_4head_1layer`: positional encoding, 4 attention heads, 1 encoder layer.
- `C_no_pos_4head_1layer`: no positional encoding, 4 attention heads, 1 encoder layer.
- `D_pos_4head_2layer`: positional encoding, 4 attention heads, 2 encoder layers.

The benchmark writes outputs to `runs/`, including:

- `benchmark_results.csv`
- `benchmark_summary.json`
- `training_curves.png`
- `training_loss_log_curves.png`
- per-variant `history.csv`
- per-variant `result.json`
- per-variant `model.pt`

## Current Benchmark Results

These are the 10-epoch benchmark results currently stored in `runs/benchmark_results.csv`.

| Model | PE | Heads | Layers | Params | Time (s) | Val Acc | Test Acc | Test F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_pos_1head_1layer` | Yes | 1 | 1 | 39490 | 75.9 | 0.9350 | 0.9160 | 0.9509 |
| `B_pos_4head_1layer` | Yes | 4 | 1 | 39490 | 90.1 | 0.9880 | 0.9800 | 0.9879 |
| `C_no_pos_4head_1layer` | No | 4 | 1 | 38210 | 72.4 | 0.8550 | 0.8670 | 0.9236 |
| `D_pos_4head_2layer` | Yes | 4 | 2 | 72962 | 99.2 | 0.9830 | 0.9700 | 0.9819 |

## Notes

The default pooling is `first`, because the first token representation can attend to later valid tokens and the task rule is anchored on the first token. You can use `--pooling mean` to run the padding-aware mean-pooling version suggested in the assignment walkthrough.

The provided labels are imbalanced toward class `1`, so the scripts report precision, recall, and F1 in addition to accuracy. Use `--class-weight` if you want the loss function to compensate for the imbalance during training.

The repository includes `.gitignore` rules for local artifacts such as `.venv/`, `runs/`, temporary Word files, and review-only `.docx` files.
