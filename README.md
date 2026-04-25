# CPE663 Mini Transformer Benchmark

This project implements a small Transformer encoder from scratch for the CPE663 major assignment. The task is binary sequence classification: predict whether the first non-padding token appears again in the second half of the valid sequence.

The implementation uses PyTorch tensors and common layers such as `nn.Linear`, `nn.Embedding`, `nn.LayerNorm`, and `nn.Dropout`, but it does not use `torch.nn.Transformer`, `torch.nn.MultiheadAttention`, Hugging Face models, or pretrained Transformers.

## Files

- `data.py`: CSV dataset loader and dataloader helpers.
- `model.py`: from-scratch mini Transformer encoder components.
- `train.py`: single-model training and evaluation script.
- `benchmark.py`: required benchmark across four model variants and report generation.
- `utils.py`: reproducibility and metric helpers.
- `train.csv`, `validation.csv`, `test.csv`: provided dataset splits.

## Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Train one model:

```powershell
python train.py --epochs 10 --num-heads 4 --num-layers 1
```

Run the full benchmark:

```powershell
python benchmark.py --epochs 10
```

For a quicker smoke test:

```powershell
python benchmark.py --epochs 1 --quiet
```

## Benchmark Variants

- `A_pos_1head_1layer`: positional encoding, 1 attention head, 1 encoder layer.
- `B_pos_4head_1layer`: positional encoding, 4 attention heads, 1 encoder layer.
- `C_no_pos_4head_1layer`: no positional encoding, 4 attention heads, 1 encoder layer.
- `D_pos_4head_2layer`: positional encoding, 4 attention heads, 2 encoder layers.

The benchmark writes outputs to `runs/`, including:

- `benchmark_results.csv`
- `training_curves.png`
- per-variant `history.csv`
- per-variant `result.json`
- per-variant `model.pt`
- `report.pdf`

## Notes

The default pooling is `first`, because the first token representation can attend to later valid tokens and the task rule is anchored on the first token. You can use `--pooling mean` to run the padding-aware mean-pooling version suggested in the assignment walkthrough.

The provided labels are imbalanced toward class `1`, so the scripts report precision, recall, and F1 in addition to accuracy. Use `--class-weight` if you want the loss function to compensate for the imbalance during training.
