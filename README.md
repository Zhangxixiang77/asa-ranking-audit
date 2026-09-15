# Ranking-reliability audit for automated speaking assessment

Code and protocol for *The Hidden Unfairness of Spoken Language Assessment: Error Parity Masks Ranking Gaps Between Children and Adults* (submitted to ICASSP 2027).

Automated speaking assessment (ASA) systems are usually audited for fairness with error metrics such as MAE. The decisions they feed are placement decisions, which depend on ranking rather than on average error. This repository contains the audit protocol we use to measure ranking reliability per subgroup, plus the scripts that produce every number in the paper.

On speechocean762, a frozen wav2vec2-base encoder with a standard MLP head ranks adults at macro PCC 0.751 and children at 0.291, while the MAE gap between the two groups is 0.001. The gap replicates on WavLM (0.732 vs 0.179), survives matched-score stratification and variance matching, and does not close under per-group calibration, residual calibration, subgroup-weighted ranking loss, or end-to-end fine-tuning of the encoder.

## What the audit needs

The protocol takes test-set predictions, reference scores, and one subgroup label per utterance. It reads nothing from inside the model, so it runs on a black-box commercial grader as readily as on the models here. Four steps:

1. **Per-subgroup ranking reliability.** Compute per-dimension Pearson correlation between predicted and reference scores within each subgroup, then macro-average over dimensions whose reference scores are not degenerate. Report how many dimensions survive. In speechocean762, adult completeness has zero variance and has to be dropped; averaging it in would silently distort the macro number.
2. **Stratified paired bootstrap.** Resample the subgroups separately, 1,000 replicates, and report an interval for every difference. On the gaps reported here this rarely changes a verdict, but subgroup sizes in deployed assessment data are often very uneven, and this is what separates a gap from sampling noise.
3. **Variance matching.** Before interpreting any gap, draw subsamples of the higher-variance group matched to the lower-variance group on per-dimension score SD and recompute the correlation. Whatever survives is the part that needs explaining. Skip this and a real gap is indistinguishable from an artifact of a compressed score range. In our case it accounts for 22% of the gap.
4. **Pairwise concordance.** Corroborate with the fraction of correctly ordered utterance pairs (reference scores differing by at least 1, ties counted as 0.5). It depends only on the predicted ordering, so it is immune to the marginal variances that make step 3 necessary.

The whole audit is one forward pass over the test split plus the bootstrap. On a corpus of this size it finishes in minutes on CPU, which is cheap enough to put in a release checklist.

## Setup

```bash
pip install -r requirements.txt

wget https://www.openslr.org/resources/101/speechocean762.tar.gz
tar -zxvf speechocean762.tar.gz
```

speechocean762 is about 520 MB. Users in mainland China may find the MagicData mirror faster: `https://openslr.magicdatatech.com/resources/101/speechocean762.tar.gz`. `02_extract_feats.py` probes huggingface.co at startup and falls back to `hf-mirror.com` if it is unreachable; pass `--hf-mirror` to set the endpoint yourself or `--no-auto-net` to turn the probing off. If `HF_ENDPOINT` is already set or a proxy is running, the script leaves both alone.

## Running the pipeline

```bash
# 1. parse the corpus, detect the child/adult split, write the metadata card
python 01_prepare_data.py --data-root /path/to/speechocean762 --out-dir ./data

# 2. extract frozen wav2vec2-base features (GPU, 20-60 min; smoke test first)
python 02_extract_feats.py --data-dir ./data --max-utts 200
python 02_extract_feats.py --data-dir ./data

# 3. train and evaluate the model family
python 03_run_experiments.py --data-dir ./data

# 4. write the audit report
python 04_make_report.py --data-dir ./data

# 5. age-bin reliability curve and the matched-tertile analysis
python 05_age_analysis.py --data-dir ./data

# 6. variance-matched subsampling (step 3 of the protocol)
python 06_variance_check.py --data-dir ./data

# 7. end-to-end encoder fine-tuning (B2-FT)
python 07_finetune_encoder.py --data-dir ./data

# 8. paired bootstrap for the fine-tuning contrast
python 08_paired_ft_bootstrap.py --data-dir ./data

# figure 1 in the paper
python plot_figure1.py --data-dir ./data
```

Step 1 finds the child/adult threshold from the gap between the two modes of the age distribution, falling back to the median if there is no gap, and prints the histogram so you can check it. Override with `--child-max-age`. It also warns when the test subgroups are badly unbalanced; `--rebalance` re-splits by subgroup while keeping speakers disjoint.

Step 2 caches features and resumes after interruption. Delete `data/feats/` after changing the encoder or the pooling.

### Cross-backbone replication

```bash
mv ./data/feats ./data/feats_w2v2
python 02_extract_feats.py --data-dir ./data --model microsoft/wavlm-base
python 03_run_experiments.py --data-dir ./data --out-suffix wavlm
python 04_make_report.py     --data-dir ./data --out-suffix wavlm
python 05_age_analysis.py    --data-dir ./data --out-suffix wavlm
```

Results go to a suffixed directory, so the wav2vec2 run is not overwritten. To switch back: `rm -rf ./data/feats && mv ./data/feats_w2v2 ./data/feats`.

## Outputs

| Path | Contents |
|---|---|
| `data/metadata_card.md` | age histogram, subgroup by split counts, score distributions |
| `data/results/SUMMARY.md` | per-subgroup PCC and MAE, paired bootstrap CIs, Wilcoxon p-values |
| `data/results/*_preds.jsonl` | per-utterance predictions, the input to any further analysis |
| `data/results_wavlm/` | the same set for the WavLM replication |

The per-utterance prediction files are the ones to keep. Every table in the paper is computed from them, so a reviewer or a reader can recompute any of it without retraining.

## Where each result comes from

| Paper | Script |
|---|---|
| Table 1 (score SD by dimension) | `01_prepare_data.py`, also written to the metadata card |
| Table 2 (main results, both backbones) | `03_run_experiments.py`, `07_finetune_encoder.py` |
| Table 3, matched score tertiles | `05_age_analysis.py` |
| Table 3, variance-matched subsampling | `06_variance_check.py` |
| Table 3, pairwise concordance | `04_make_report.py` |
| Table 4 (remedy matrix) | `03_run_experiments.py`, `08_paired_ft_bootstrap.py` |
| Figure 1 (reliability by age bin) | `plot_figure1.py` |

The frozen-encoder family (B0, B2, B6, Ours, and the subgroup-weighted ranking variants B2-Rank and Ours-Rank) trains in `03_run_experiments.py`. B2-FT unfreezes the encoder and trains separately in `07_finetune_encoder.py`, at encoder learning rate 1e-5 and head learning rate 1e-3.

## Scope

One corpus, one L1, and one subgroup axis. The protocol itself is corpus-agnostic and we release it for exactly that reason, but nothing here establishes that the size or the shape of the gap transfers. The 16 to 18 age band is absent from speechocean762, so the age curve has a hole in it and we make no claim about where in that region reliability rises. Age and proficiency are entangled by design in this corpus.

## Data and licensing

speechocean762 is distributed under CC BY 4.0 by SpeechOcean and Xiaomi ([OpenSLR 101](https://www.openslr.org/101/)). No new recordings were collected for this work and no human subjects were recruited. Speaker age and gender are used as distributed with the corpus. The code in this repository is released under the MIT license.

## Citation

```bibtex
@inproceedings{zhang2027ranking,
  title     = {The Hidden Unfairness of Spoken Language Assessment:
               Error Parity Masks Ranking Gaps Between Children and Adults},
  author    = {Zhang, Xixiang and Zhang, Zhi},
  booktitle = {Proc. IEEE ICASSP},
  year      = {2027},
  note      = {Under review}
}
```

If you use the corpus, cite it as well:

```bibtex
@inproceedings{zhang2021speechocean762,
  title     = {speechocean762: An Open-Source Non-native English Speech Corpus
               for Pronunciation Assessment},
  author    = {Zhang, Junbo and Zhang, Zhiwen and Wang, Yongqing and Yan, Zhiyong
               and Song, Qiong and Huang, Yukai and Li, Ke and Povey, Daniel
               and Wang, Yujun},
  booktitle = {Proc. Interspeech},
  pages     = {3710--3714},
  year      = {2021}
}
```

## Use of AI assistance

An AI assistant (Kimi, Moonshot AI) was used to draft parts of the code in this repository and to edit prose in the accompanying paper. The authors reviewed and verified all code, ran all experiments, and take full responsibility for the contents of both.

## Contact

Xixiang Zhang (first author), 52300936020@stu.ecnu.edu.cn
Zhi Zhang (corresponding author), zhangzhi@mail.ecnu.edu.cn
East China Normal University, Shanghai, China
