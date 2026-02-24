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

**Faithful TimeSieve (FTS)** is an enhanced framework for improving the **reliability and robustness** of time series forecasting in multimedia-rich web settings (e.g. video streaming workloads, ad click prediction). While [TimeSieve](https://github.com/ninghuifeng/TimeSieve) achieves strong accuracy, it is sensitive to **random seeds**, **input noise**, **layer noise**, and **parameter perturbations**. FTS systematically detects and mitigates these unfaithfulness issues via a rigorous definition and three auxiliary losses:

- **Similarity in IB Space (Sib)** — Filtered representations in the information-bottleneck space under perturbation stay close (within $\beta$, radius $R_1$).
- **Consistency in Prediction Space (Cps)** — Predictions with original vs. fine-tuned weights $\tilde{\omega}$ differ by at most $\alpha_1$.
- **Stability in Noise Perturbations (Snp)** — Predictions under input perturbation $\delta$ with $\|\delta\| \leq R_2$ differ by at most $\alpha_2$.

The total training objective is:

$$\mathcal{L} = \mathcal{L}_{\mathrm{reg}} + \mathcal{L}_{\mathrm{IB}} + \lambda_1 \mathcal{L}_{\mathrm{sib}} + \lambda_2 \mathcal{L}_{\mathrm{cps}} + \lambda_3 \mathcal{L}_{\mathrm{snp}}$$

where $\mathcal{L}_{\mathrm{reg}}$ is the regression loss, $\mathcal{L}_{\mathrm{IB}}$ is the TimeSieve IB loss, and $\mathcal{L}_{\mathrm{sib}}$, $\mathcal{L}_{\mathrm{cps}}$, $\mathcal{L}_{\mathrm{snp}}$ are the faithfulness auxiliary losses. FTS uses PGD to find worst-case perturbations and batched gradient updates for parameters, achieving **SOTA** on multiple benchmarks while improving stability and consistency.

---

## Key Contributions (from paper)

1. **Comprehensive Faithfulness Assessment** — In-depth analysis of TimeSieve identifying factors that affect its faithfulness (random seeds, input/layer/parameter perturbations).
2. **Definition of Faithful TimeSieve** — Rigorous $(\alpha_1, \alpha_2, \beta, \delta, R_1, R_2)$-Faithful definition with three attributes (Sib, Cps, Snp).
3. **Multimedia-aware Robustness Framework** — Min-max optimization with PGD and content-adaptive stabilization; framework transfers to other time series models (e.g. PatchTST).
4. **Theoretical and Experimental Validation** — Bounds for Sib/Cps/Snp and extensive experiments on Wiki, ETTh1, Exchange; FTS achieves SOTA and strong robustness.

---

## Table of Contents

- [Method](#method)
- [Main Results (Tables)](#main-results-tables)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data & Usage](#data--usage)
- [Citation](#citation)
- [License](#license)

---

## Method

**TimeSieve** uses wavelet decomposition (approximation \(\pi_a\), detail \(\pi_d\)) and an Information Filtering and Compression Block (IFCB) with IB loss. FTS adds:

- **Objective:** Minimize $\mathbb{E}_x[\lambda_1 \mathcal{L}_{\mathrm{sib}} + \lambda_2 \mathcal{L}_{\mathrm{cps}} + \lambda_3 \mathcal{L}_{\mathrm{snp}}]$ over $\tilde{\omega}$, where $\mathcal{L}_{\mathrm{sib}}$ corresponds to $D_1(\hat{\pi}_a(\cdot), \hat{\pi}_a(\cdot+\delta))$ (and similarly for $\hat{\pi}_d$); $\mathcal{L}_{\mathrm{cps}}$ to $D_2(y(x,\tilde{\omega}), y(x,\omega))$; $\mathcal{L}_{\mathrm{snp}}$ to $D_3(y(x,\tilde{\omega}), y(x+\delta,\tilde{\omega}))$.
- **PGD step:** At iteration $p$, update perturbation $\delta_p$ by gradient ascent on the sum of these distances, then project to $\|\delta\| \leq R$; then update $\tilde{\omega}$ by gradient descent on $\mathcal{L}_{\mathrm{reg}} + \mathcal{L}_{\mathrm{IB}} + \lambda_1 \mathcal{L}_{\mathrm{sib}} + \lambda_2 \mathcal{L}_{\mathrm{cps}} + \lambda_3 \mathcal{L}_{\mathrm{snp}}$.
- **Lookback:** We set input length $T = 2H$ (twice the forecast horizon $H$).

---

## Main Results (Tables)

### Table 1: Forecasting results (no perturbation)

Forecast length $H \in \{48, 96, 144, 192\}$, lookback $T = 2H$. **Bold** = best, *italic* = second best.

| Dataset | H | FTS (MAE / MSE) | TS (MAE / MSE) | Koopa | PatchTST | TSMixer | DLinear | NSTformer | LightTS | Autoformer |
|---------|---|------------------|----------------|-------|----------|---------|---------|-----------|---------|------------|
| **Wiki** | 48 | **0.305 / 0.467** | 0.323 / *0.490* | 0.314 / 0.496 | 0.312 / 0.495 | 0.318 / *0.490* | 0.350 / 0.517 | *0.307* / 0.508 | 0.313 / 0.494 | 0.315 / 0.522 |
| | 96 | **0.262 / 0.430** | *0.266 / 0.433* | 0.283 / 0.451 | 0.280 / 0.445 | 0.277 / 0.435 | 0.286 / 0.462 | 0.303 / 0.633 | 0.287 / 0.460 | 0.348 / 0.550 |
| | 144 | **0.268 / 0.445** | *0.270 / 0.447* | 0.284 / 0.459 | 0.280 / 0.451 | 0.320 / 0.496 | 0.287 / 0.458 | 0.309 / 0.521 | 0.291 / 0.471 | 0.360 / 0.616 |
| | 192 | **0.273 / 0.447** | *0.276 / 0.452* | 0.294 / 0.467 | 0.289 / 0.461 | 0.370 / 0.644 | 0.289 / 0.463 | 0.339 / 0.605 | 0.301 / 0.490 | 0.373 / 0.629 |
| **ETTh1** | 48 | **0.360 / 0.340** | *0.361 / 0.341* | 0.385 / 0.364 | 0.375 / 0.342 | 0.432 / 0.407 | 0.372 / 0.342 | 0.465 / 0.614 | 0.406 / 0.404 | 0.432 / 0.678 |
| | 96 | **0.383 / 0.376** | *0.384 / 0.376* | 0.411 / 0.406 | 0.395 / 0.377 | 0.473 / 0.466 | 0.395 / 0.380 | 0.498 / 0.653 | 0.431 / 0.435 | 0.496 / 0.578 |
| | 144 | **0.396 / 0.391** | *0.397 / 0.393* | 0.426 / 0.424 | 0.412 / 0.394 | 0.528 / 0.537 | 0.401 / 0.394 | 0.536 / 0.602 | 0.442 / 0.453 | 0.521 / 0.761 |
| | 192 | **0.406 / 0.402** | *0.408 / 0.402* | 0.434 / 0.430 | 0.437 / 0.416 | 0.592 / 0.642 | 0.416 / 0.408 | 0.543 / 0.684 | 0.457 / 0.471 | 0.568 / 0.598 |
| **Exchange** | 48 | **0.139 / 0.042** | *0.140 / 0.043* | 0.149 / 0.046 | 0.145 / 0.048 | 0.149 / 0.046 | 0.145 / 0.046 | 0.187 / 0.073 | 0.159 / 0.067 | 0.205 / 0.124 |
| | 96 | **0.196 / 0.084** | *0.197 / 0.086* | 0.211 / 0.092 | 0.204 / 0.090 | 0.211 / 0.092 | 0.223 / 0.089 | 0.294 / 0.159 | 0.247 / 0.168 | 0.778 / 0.409 |
| | 144 | **0.242 / 0.123** | *0.243 / 0.124* | 0.265 / 0.141 | 0.265 / 0.138 | 0.265 / 0.141 | 0.256 / 0.133 | 0.375 / 0.292 | 0.272 / 0.310 | 0.680 / 0.671 |
| | 192 | **0.287 / 0.170** | *0.292 / 0.179* | 0.329 / 0.212 | 0.298 / 0.181 | 0.329 / 0.212 | 0.301 / 0.182 | 0.464 / 0.494 | 0.354 / 0.403 | 0.979 / 0.544 |

### Table 2: Random seed stability (Exchange, 96-step, MSE)

Baseline seed 2021*. FTS is less sensitive to seeds; **Preference (%)** quantifies combined performance and stability (higher = FTS preferred).

| Seed | TS (MSE) | FTS (MSE) | Preference (%) |
|------|----------|-----------|----------------|
| 2021* | 0.0929 | 0.0868 | — |
| 2022 | 0.0989 | 0.0867 | **99.61%** |
| 2023 | 0.0819 | 0.0864 | **96.09%** |
| 2024 | 0.0993 | 0.0860 | **87.95%** |
| 2025 | 0.0917 | 0.0878 | **4.49%** |
| 2026 | 0.0960 | 0.0876 | **69.27%** |
| 2027 | 0.0826 | 0.0892 | **74.16%** |
| 2028 | 0.0818 | 0.0863 | **95.81%** |
| 2029 | 0.1160 | 0.0879 | **94.69%** |
| 2030 | 0.0897 | 0.0898 | **1.53%** |

### Table 3: Robustness under perturbation (summary)

NP = no perturbation, NPO = no perturbation with FTS optimization, IP = input perturbation, IPO = IP with optimization, ILP = intermediate-layer perturbation, ILPO = ILP with optimization. FTS (NPO/IPO/ILPO) consistently improves over TS (NP/IP/ILP); e.g. Wiki 48-step IP: MAE 0.433→0.367 (IPO), ILP: MAE 0.453→0.399 (ILPO). See paper for full table.

### Table 4: Loss ablation (ETTh1 & Exchange, MAE/MSE)

Full combination $\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{reg}} + \mathcal{L}_{\mathrm{IB}} + \lambda_1 \mathcal{L}_{\mathrm{sib}} + \lambda_2 \mathcal{L}_{\mathrm{cps}} + \lambda_3 \mathcal{L}_{\mathrm{snp}}$ achieves best. Removing $\mathcal{L}_{\mathrm{snp}}$, $\mathcal{L}_{\mathrm{cps}}$, or $\mathcal{L}_{\mathrm{sib}}$ degrades robustness; see paper Table 4 for all variants.

---

## Figures (from paper)

| Description | Figure |
|-------------|--------|
| **TS vs FTS under 10 random seeds** (Wiki/ETTh1/Exchange) | ![first_per](figures/first_per.png) |
| **FTS framework** | ![framework](figures/model.png) |
| **Heatmaps (Wiki, H=48,96,144,192)** | See [project page](https://xll0328.github.io/fts/) for full heatmaps. |

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
└── figures/                # Paper figures
```

---

## Data & Usage

**Datasets:** ETT (ETTh1), Exchange, Wiki pageviews (see paper). Lookback $T = 2H$. Place data under `./data/` or set `--root_path` and `--data_path`.

**Example (long-term forecasting with FTS):**

```bash
python run_rob.py --task_name long_term_forecast --is_training 1 --model_id FTS_ETTh1 --model Timesieve \
  --data ETTh1 --root_path ./data/ETT/ --data_path ETTh1.csv \
  --seq_len 96 --pred_len 96 --features M --checkpoints ./checkpoints/
```

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
