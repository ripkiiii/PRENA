---
language:
- id
license: cc-by-4.0
task_categories:
- text-generation
tags:
- public-relations
- indonesian
- instruction-tuning
- nlp
- low-resource
- press-release
pretty_name: PRENA — Indonesian Digital PR Dataset
size_categories:
- n<1K
configs:
- config_name: default
  data_files:
  - split: train
    path: train.jsonl
  - split: validation
    path: val.jsonl
  - split: test
    path: test.jsonl
  - split: test_ood
    path: test_ood.jsonl
---

# PRENA: Indonesian Digital Public Relations Dataset

**PRENA** (Public Relations Engine Nusantara AI) is, to the best of our knowledge, the first dataset specifically designed for Indonesian digital public relations instruction tuning tasks.

## Dataset Description

The dataset contains **748 instruction-response pairs** in ChatML format, covering seven core PR task categories in Bahasa Indonesia. It combines real-world press releases scraped from Indonesian news sources with synthetically generated pairs covering underrepresented PR categories.

## Dataset Structure

### Splits

| Split | Pairs | Description |
|-------|-------|-------------|
| `train` | 513 | Core training set (Synthetic v1+v2 + Antara) |
| `val` | 64 | Validation set |
| `test` | 64 | In-domain test set |
| `test_ood` | 107 | Out-of-domain test set (Bisnis.com + Liputan6) |

### Categories

| Category | Description |
|----------|-------------|
| `press_release` | Corporate or institutional press release writing |
| `caption_sosmed` | Social media caption creation (Instagram, Twitter/X, LinkedIn, TikTok) |
| `crisis_communication` | Crisis statements and reputation management |
| `media_monitoring` | Media coverage analysis and sentiment reporting |
| `media_pitch` | Email pitches and story angles for journalists |
| `brand_storytelling` | Brand narrative and manifesto writing |
| `influencer_brief` | Creative briefs for influencer/KOL collaborations |

**Train split category distribution:**

| Category | Count | % |
|----------|-------|---|
| press_release | 240 | 46.8% |
| caption_sosmed | 73 | 14.2% |
| media_monitoring | 53 | 10.3% |
| crisis_communication | 51 | 9.9% |
| media_pitch | 42 | 8.2% |
| brand_storytelling | 33 | 6.4% |
| influencer_brief | 21 | 4.1% |

### Data Format

Each entry follows the ChatML format with an additional `category` field:

```json
{
  "category": "press_release",
  "messages": [
    {"role": "user", "content": "[PR practitioner instruction]"},
    {"role": "assistant", "content": "[professional PR response]"}
  ]
}
```

## Data Sources

**Real data (350 pairs):**
- `antaranews.com` — 243 pairs (press releases, Antara news agency)
- `bisnis.com` — 34 pairs (OOD test set only)
- `liputan6.com` — 73 pairs (OOD test set only)

**Synthetic data (398 pairs):**
- Synthetic v1 — 200 pairs (all 7 categories)
- Synthetic v2 — 198 pairs (6 non-press_release categories, for category balancing)

## Experimental Results

We trained **PRENA Nano**, a 10.5M-parameter LLaMA-style decoder-only language model, on this dataset. Results demonstrate the impact of dataset quality and category balance:

| Model | In-domain PPL | OOD PPL | OOD Gap |
|-------|--------------|---------|---------|
| PRENA Nano v1 (315 pairs, imbalanced) | 94.52 | 132.09 | 37.57 |
| PRENA Nano v2 (513 pairs, balanced) | **17.55** | **24.75** | **7.20** |

Category-balanced training achieves a **5.4× improvement** in in-domain perplexity and reduces the OOD gap from 37.57 to 7.20.

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

This dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Real data sourced from publicly accessible Indonesian news websites. Synthetic data generated using large language models.

## Contact

Muhammad Rifky Firmansyah Sujana — Universitas Telkom, Fakultas Komunikasi dan Ilmu Sosial, Digital Public Relations
