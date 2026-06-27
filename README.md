# PRENA

**PRENA** (Public Relations Engine Nusantara AI) — Indonesian Digital PR Dataset + nano language model, built for Telkom University thesis (Digital Public Relations).

## What is this?

First dataset specifically designed for Indonesian digital public relations instruction tuning. 748 instruction-response pairs across 7 PR task categories, combining real press releases scraped from Indonesian news sources with synthetically generated pairs.

## Dataset

| Split | Pairs |
|-------|-------|
| train | 513 |
| val | 64 |
| test | 64 |
| test_ood | 107 |

**Categories:** press_release, caption_sosmed, crisis_communication, media_monitoring, media_pitch, brand_storytelling, influencer_brief

HuggingFace: [ripkiiiii/prena](https://huggingface.co/datasets/ripkiiiii/prena)

## Model: PRENA Nano

10.5M-parameter LLaMA-style decoder-only model trained on this dataset.

| Model | In-domain PPL | OOD PPL |
|-------|--------------|---------|
| v1 (315 pairs, imbalanced) | 94.52 | 132.09 |
| v2 (513 pairs, balanced) | **17.55** | **24.75** |

5.4× improvement from category-balanced training.

## Structure

```
dataset/      — train/val/test splits (jsonl)
model/        — PRENA Nano checkpoints + tokenizer
notebooks/    — Kaggle training notebook
scripts/      — data collection, generation, training, eval
docs/         — documentation
```

## Citation

```bibtex
@dataset{sujana2026prena,
  title={PRENA: Indonesian Digital Public Relations Dataset},
  author={Sujana, Muhammad Rifky Firmansyah},
  year={2026},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/ripkiiiii/prena}
}
```

## License

CC BY 4.0 — Universitas Telkom, Fakultas Komunikasi dan Ilmu Sosial
