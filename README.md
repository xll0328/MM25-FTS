# FTS: From Guesswork to Guarantee — Faithful Multimedia Web Forecasting with TimeSieve

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Project Page](https://img.shields.io/badge/Project%20Page-xll0328.github.io%2Ffts-2563eb)](https://xll0328.github.io/fts/)

**This paper was accepted at ACM MM 2025** (33rd ACM International Conference on Multimedia, October 27–31, 2025, Dublin, Ireland). CCF A, Core A\*. This repository is the official implementation of **Faithful TimeSieve (FTS)**.

| [**Project Page**](https://xll0328.github.io/fts/) | [**Code (GitHub)**](https://github.com/xll0328/MM25-FTS) |
|:---:|:---:|
| [xll0328.github.io/fts](https://xll0328.github.io/fts/) | [GitHub](https://github.com/xll0328/MM25-FTS) |

**Authors:** Songning Lai, Ninghui Feng, Jiechao Gao, Hao Wang, Haochen Sui, Xin Zou, Jiayu Yang, Wenshuo Chen, Lijie Hu, Hang Zhao, Xuming Hu, Yutao Yue

---

## Overview

**Faithful TimeSieve (FTS)** is an enhanced framework for improving the **reliability and robustness** of time series forecasting, especially in multimedia-rich web settings (e.g. video streaming workloads, ad click prediction). While [TimeSieve](https://github.com/ninghuifeng/TimeSieve) achieves strong accuracy, it is sensitive to random seeds, input noise, layer noise, and parameter perturbations. FTS systematically detects and mitigates these **unfaithfulness** issues and defines three core attributes:

- **Similarity in IB Space (Sib)** — Filtered representations stay close to the original in the information-bottleneck space under perturbations.
- **Consistency in Prediction Space (Cps)** — Predictions with original vs. fine-tuned weights remain consistent.
- **Stability in Noise Perturbations (Snp)** — Predictions are stable under bounded input perturbations.

FTS formulates a min-max optimization with PGD-based adversarial updates and adds auxiliary losses (L_sib, L_cps, L_snp) to TimeSieve training, achieving SOTA on multiple benchmarks while improving faithfulness.

---

## Table of Contents

- [Method](#method)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data & Usage](#data--usage)
- [Results Summary](#results-summary)
- [Citation](#citation)
- [License](#license)

---

## Method

- **Preliminaries:** TimeSieve uses wavelet decomposition and an Information Filtering and Compression Block (IFCB) for time series forecasting.
- **FTS definition:** A TimeSieve is **(α₁, α₂, β, δ, R₁, R₂)-Faithful** if it satisfies Sib, Cps, and Snp (see paper).
- **Framework:** Minimize a combined objective: L_reg + L_IB + λ₁·L_sib + λ₂·L_cps + λ₃·L_snp, with PGD iterations to find worst-case perturbations and batched gradient updates for model parameters.
- **Scalability:** The same idea extends to other time series models (e.g. PatchTST); see paper and appendix.

---

## Installation

```bash
git clone https://github.com/xll0328/MM25-FTS.git
cd MM25-FTS
pip install -r requirements.txt
```

---

## Project Structure

```
MM25-FTS/
├── README.md
├── LICENSE
├── requirements.txt
├── run_rob.py              # Main entry for robust (FTS) long-term forecasting
├── data_provider/          # Data loading
├── exp/                    # Experiment scripts (including exp_long_term_forecasting_rob)
├── layers/                 # Model layers
├── models/                 # Model definitions (TimeSieve, etc.)
├── utils/                  # Utilities
└── figures/                # Paper figures (heatmaps, etc.)
```

---

## Data & Usage

**Datasets:** ETT (ETTh1, etc.), Exchange, Wiki pageviews (see paper). Place data under `./data/` or set `--root_path` and `--data_path`.

**Example (long-term forecasting with FTS):**

```bash
python run_rob.py --task_name long_term_forecast --is_training 1 --model_id FTS_ETTh1 --model Timesieve \
  --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv \
  --seq_len 96 --pred_len 96 --features M --checkpoints ./checkpoints/
```

Adjust `--seq_len`, `--pred_len`, and other args as in the paper (e.g. T=2H for lookback). For full FTS training, use the robust experiment class and PGD/faithfulness losses as implemented in `exp/`.

---

## Results Summary

- **No perturbation:** FTS achieves best or second-best MAE/MSE on Wiki, ETTh1, and Exchange across horizons H ∈ {48, 96, 144, 192} (see paper Table 1).
- **Random seeds:** FTS consistently reduces sensitivity to seeds (e.g. Preference up to 99.61% on Exchange 96-step, Table 2).
- **Input / layer perturbation:** With optimization (IPO, ILPO), FTS recovers much of the performance drop under NP (see paper Table 3).
- **Ablation:** Full loss combination L_total performs best; removing L_snp, L_cps, or L_sib degrades robustness (see paper Table 4).

---

## Citation

```bibtex
@inproceedings{lai2025fts,
  title={From Guesswork to Guarantee: Towards Faithful Multimedia Web Forecasting with TimeSieve},
  author={Songning Lai and Ninghui Feng and Jiechao Gao and Hao Wang and Haochen Sui and Xin Zou and Jiayu Yang and Wenshuo Chen and Lijie Hu and Hang Zhao and Xuming Hu and Yutao Yue},
  booktitle={Proceedings of the 33rd ACM International Conference on Multimedia (MM '25)},
  year={2025},
  address={Dublin, Ireland},
}
```

---

## License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE).

## References

- [TimeSieve](https://github.com/ninghuifeng/TimeSieve)
- [PatchTST](https://github.com/yuqinie98/PatchTST), [DLinear](https://github.com/cure-lab/LTSF-Linear), [Autoformer](https://github.com/thuml/Autoformer), etc. (see paper)
