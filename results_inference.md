# CISLNet inference experiments: Vis+ channels & architecture latency

**As of 2026-10-22.** Three experiment families on the **canonical frozen test** (`dataset/data_frozen`, **n=6929**). Figures: `[results_figures_inference/](./results_figures_inference/)`.

- **Exp 1 — Vis+ IR count** (vp1/vp3/vp5, 2026-10)
- **Exp 2 — train-time architecture → inference speed** (groups A/B/C/D, 2026-10)
- **Exp 3 — VIS-only input & decoder-capacity ladder** (`visonly`, `dec10/25/75`, `encdec50`, 2026-10-16 → 10-22)

Primary quality metric: **LPIPS** (ConvNeXt-TCGA, Zhang lin0). Latency is measured only for Exp 2/3 (`scripts/cislnet_exp_eval.py`: 10 warmup + 100 timed forwards, CUDA sync, 1024², batch=1).

Related: `[results_organ.md](./results_organ.md)` (organ holdouts), `[results_c13.md](./results_c13.md)` (c13 epoch curves), `[results.md](./results.md)` (Exp 4: no-attn + objective ablations).

This is **not** the organ holdout protocol in `[results_organ.md](./results_organ.md)` (different test lists, n≠6929).

Refresh figures with:

```bash
python scripts/plot_inference_results.py    # Exp 1 + 2 figures
python scripts/plot_capacity_results.py     # Exp 3 figures (below)
```

---



## Experiment 1 — Vis+ IR count (vp1 / vp3 / vp5)



### Motivation

Full CISLNet uses **13 channels** (10 IR + 3 VIS). Vis+ asks how much IR is still needed when VIS is always available: keep RGB VIS and add the top-N IR bands in wavelength priority order **1650 → 1081 → 1765 → 1140 → 1547** (indices `8,0,9,1,6`). That yields Vis+1 / +3 / +5 → `input_nc` 4 / 6 / 8 (`jobs/cislnet/submit_visplus.sh`, `jobs/_lib/cislnet_channels.sh`).

### Setup


| Run                | Channels         | IR bands       | Train                             | Eval        |
| ------------------ | ---------------- | -------------- | --------------------------------- | ----------- |
| **c13** (baseline) | 13 (10 IR + VIS) | all            | `c13` best-LPIPS (=e24 as latest) | `test/c13`  |
| vp1                | 4 (1 IR + VIS)   | 1650           | 20 ep, CBAM, same frozen train    | `cn_te_vp1` |
| vp3                | 6 (3 IR + VIS)   | 1650,1081,1765 | 20 ep                             | `cn_te_vp3` |
| vp5                | 8 (5 IR + VIS)   | five listed    | 20 ep                             | `cn_te_vp5` |


Same AttUNet + contrastive pix2pix stack; only the **input channel set** changes. No architecture latency ablation here.

![Vis+ LPIPS vs c13 on frozen test](./results_figures_inference/fig_visplus_lpips.png)

### Results (LPIPS ↓)


| Run     | IR # | LPIPS ↓      | PSNR  | FID  |
| ------- | ---- | ------------ | ----- | ---- |
| **c13** | 10   | **0.001872** | 18.00 | 6.10 |
| vp3     | 3    | 0.001928     | 17.57 | 8.18 |
| vp5     | 5    | 0.001942     | 17.42 | 7.27 |
| vp1     | 1    | 0.002077     | 16.68 | 8.54 |


Source: `results/cislnet/test/{c13,vp1,vp3,vp5}/patch_eval_test/metrics.json` (CSV: `results_figures_inference/visplus_metrics.csv`). The `c13` leaf has since been re-evaluated (2026-09-28): LPIPS 0.001819 / FID 5.84 — see the Exp 3 note. Figures above keep the numbers they were built with.

**Takeaway:** On this frozen split, **full c13 still has the best LPIPS**. Vis+3 is the best Vis+ setting and closes most of the gap; Vis+1 is clearly worse. Vis+ does not beat c13 on LPIPS.

WSI JPEG previews (10 slides): **vp1 / vp3 / vp5 all complete** under `results/cislnet/test/vp{1,3,5}/wsi_pred/`.

---



## Experiment 2 — Train-time architecture → inference speed (Groups A/B/C/D)



### Motivation

Default skip attention is **CBAM** (~47 ms/patch at 1024). Ablations replace **attention** (group A) or **structure/capacity** (group B) without changing the frozen train split, then measure latency vs LPIPS (`jobs/cislnet/experiments/`). Eval path is `cislnet_exp_eval.py` → `tracking.csv`.

### Grouping (canonical)


| Group  | Meaning                       | Runs                                                                                                        |
| ------ | ----------------------------- | ----------------------------------------------------------------------------------------------------------- |
| **A**  | Skip **attention**            | A1 CoordAtt, A2 Triplet, A3 SCSE, A4 GCBlock, **A5 ECA** (`b4_eca_*` legacy tag), **A6 SimAM** (`b5_simam_*`) |
| **B**  | **Structure / capacity**      | B1 half decoder, B2 SCConv, B3 ConvNeXt, B6 depthwise, B7 PixelShuffle                                      |
| **C**  | Color space                   | C1 linear HE                                                                                                |
| **IO** | Data path only (not a D knob) | fp16 on-disk IR                                                                                             |
| **D**  | Stack top A × top B           | Fresh trains that combine one attention winner with one structure winner                                    |



| Run          | Change                               | Group | Status             |
| ------------ | ------------------------------------ | ----- | ------------------ |
| vs13 / c13   | CBAM (baseline)                      | ref   | ✓ tracked          |
| A1 CoordAtt  | `--attention_type coordatt`          | A     | ✓                  |
| A2 Triplet   | `triplet`                            | A     | ✓                  |
| A3 SCSE      | `scse`                               | A     | ✓                  |
| A4 GCBlock   | `gcblock`                            | A     | ✓                  |
| A5 ECA       | `eca` (run tag `b4_eca_c13`)         | A     | ✓                  |
| A6 SimAM     | `simam` (run tag `b5_simam_c13`)     | A     | ✓                  |
| B1 half      | `--decoder_width_mult 0.5`, `ndf=32` | B     | ✓                  |
| B2 SCConv    | `--decoder_conv scconv`              | B     | ✓                  |
| B3 ConvNeXt  | `--encoder_backbone convnext_tiny`   | B     | ✓                  |
| B6 depthwise | `--decoder_conv depthwise`           | B     | ✓                  |
| B7 PixelShuf | `--upsample_mode pixelshuffle`       | B     | ✓                  |
| fp16 IR      | `DATAROOT=data_frozen_fp16`          | IO    | ✓                  |
| C1 linear    | `--he_color_space linear`            | C     | ✓ (quality poor)   |
| D stacks     | top-2 A × top-2 B                    | D     | ✓ all 4 complete   |
| D followups  | GCBlock stacks + depthwise stacks    | D     | ✓ 5 / 5 complete   |


A/B/C/D trains used **20 epochs**. Canonical baseline is `results/cislnet/c13` (best-LPIPS epoch 24 promoted to `latest`).

![Group A attention Pareto](./results_figures_inference/fig_latency_lpips_A.png)

![Group B structure Pareto](./results_figures_inference/fig_latency_lpips_B.png)

![Groups A + B Pareto](./results_figures_inference/fig_latency_lpips_AB.png)

![Group D stacks Pareto](./results_figures_inference/fig_latency_lpips_D.png)

![Latency vs LPIPS Pareto for A/B ablations + D stacks (diamonds)](./results_figures_inference/fig_latency_lpips.png)

![Latency bars sorted by ms, incl. D stacks](./results_figures_inference/fig_latency_bars.png)


### Results — single-knob A/B (latency + LPIPS)

Sorted by latency. Latest `tracking.csv` row per run.


| Run             | Group | Latency (ms) | Speedup vs c13 | LPIPS (CN)   | FID      |
| --------------- | ----- | ------------ | -------------- | ------------ | -------- |
| B2 SCConv       | B     | 83.21        | 0.56×          | 0.001961     | 7.49     |
| B3 ConvNeXt     | B     | 48.44        | 0.96×          | 0.001915     | 9.28     |
| B7 PixelShuffle | B     | 46.81        | 1.00×          | 0.001902     | 6.44     |
| C1 linear       | C     | 46.62        | 1.00×          | 0.003089     | 53.29    |
| fp16 IR         | IO    | 46.62        | 1.00×          | 0.001861     | 7.27     |
| c13 / CBAM      | ref   | 46.59        | 1.00×          | 0.001871     | 6.08     |
| B6 depthwise    | B     | 42.62        | 1.09×          | 0.001922     | 7.11     |
| B1 half decoder | B     | 36.44        | 1.28×          | 0.001865     | 7.43     |
| A1 CoordAtt     | A     | 33.87        | 1.38×          | 0.001940     | 7.17     |
| A2 Triplet      | A     | 31.80        | 1.46×          | **0.001817** | 6.70     |
| A6 SimAM        | A     | 26.51        | 1.76×          | 0.001928     | 7.88     |
| A4 GCBlock      | A     | 25.71        | 1.81×          | 0.001862     | **5.21** |
| A3 SCSE         | A     | 25.68        | 1.81×          | 0.001890     | 6.43     |
| **A5 ECA**      | A     | **25.03**    | **1.86×**      | 0.001861     | 6.16     |


Source: `results/cislnet/experiments/tracking.csv` (CSV: `results_figures_inference/latency_metrics.csv`). C1 omitted from figures (scale outlier).

**Takeaway:** Attention swaps (group A) dominate the speed/quality Pareto. **A5 ECA** is fastest at LPIPS≈CBAM; **A2 Triplet** has best LPIPS; **A4** has best FID. Structure (group B) helps less: B1 is the best B by ConvNeXt-LPIPS. **C1 linear** is unusable (FID 53).

### D experiments — what they mean

A and B each change **one** knob (attention *or* structure; the other stays at CBAM / full decoder). **D is a stacking / interaction test:** pick the top-2 attentions and top-2 structures by ConvNeXt-LPIPS (then latency), train a **fresh** model with that pair, and ask whether speed and quality both improve vs the parents — or whether the knobs interfere.

Selector: `scripts/select_pareto_d1.py`. Env files: `results/cislnet/experiments/d_combos/`. All four jobs completed 2026-10-09.

Followup batch (manual, `results/cislnet/experiments/d_combos_followup/`, submitted 2026-10-10): GCBlock × {half, PS, DW} plus ECA × DW and Triplet × DW — i.e. the best-FID attention (A4 GCBlock, FID 5.21) stacked with each structure, and the depthwise structure (B6) stacked with each top attention. **5 / 5 complete** (F5 eval finished 2026-10-19).

| Rank | Combo | = parents | Latency | Speedup | LPIPS ↓ | FID | vs parents |
| ---- | ----- | --------- | ------- | ------- | ------- | --- | ---------- |
| A1×B1 | `d1_resnet50_triplet_c13_half` | A2 + B1 | 20.14 | 2.31× | 0.001891 | 5.41 | Faster than both; LPIPS **worse** than A2 (0.001817) |
| A1×B2 | `d1_resnet50_triplet_c13_ps` | A2 + B7 | 31.84 | 1.46× | 0.001889 | **5.16** | Latency ≈ A2 alone; LPIPS worse than A2; **best FID in D** |
| A2×B1 | `d1_resnet50_eca_c13_half` | A5 + B1 | **16.58** | **2.81×** | 0.001962 | 7.60 | **Fastest in whole sweep**; quality **regresses** vs ECA and B1 |
| A2×B2 | `d1_resnet50_eca_c13_ps` | A5 + B7 | 24.95 | 1.87× | **0.001848** | 5.24 | Latency ≈ ECA; LPIPS **beats** CBAM / ECA / B7 — only clear **positive** quality stack |
| F1 | `d1_resnet50_gcblock_c13_half` | A4 + B1 | 17.09 | 2.73× | 0.001905 | 7.21 | Faster than both; LPIPS worse than A4 (0.001862) and B1 (0.001865) — same half-decoder interference |
| F2 | `d1_resnet50_gcblock_c13_ps` | A4 + B7 | 25.70 | 1.81× | 0.001877 | 7.06 | Latency ≈ A4 alone; LPIPS worse than A4, better than B7; FID worse than both parents — PS does not help GCBlock |
| F3 | `d1_resnet50_eca_c13_dw` | A5 + B6 | 20.96 | 2.22× | 0.001913 | 5.90 | Faster than both; LPIPS worse than ECA (≈ B6 solo); FID **beats** both parents (6.16 / 7.11) |
| F4 | `d1_resnet50_gcblock_c13_dw` | A4 + B6 | 21.73 | 2.14× | **0.001844** | 5.75 | Faster than both; LPIPS **beats** both parents and CBAM — new **best-LPIPS D stack**, 2nd positive quality stack |
| F5 | `d1_resnet50_triplet_c13_dw` | A2 + B6 | 27.95 | 1.67× | 0.001825 | 9.03 | Faster than both parents; LPIPS ≈ Triplet solo (0.001817) and beats CBAM (0.001871); FID **worse than both** (6.70 / 7.11) — LPIPS held up, FID did not |

Parent reference: CBAM 46.6 ms / 0.001871 / FID 6.08 · A2 31.8 / **0.001817** / 6.70 · A4 GCBlock 25.7 / 0.001862 / **5.21** · A5 ECA 25.0 / 0.001861 / 6.16 · B1 36.4 / 0.001865 / 7.43 · B6 DW 42.6 / 0.001922 / 7.11 · B7 46.8 / 0.001902 / 6.44.

**F5 completed 2026-10-19:** `d1_resnet50_triplet_c13_dw` (job 11505030 train → eval hop) — Triplet × depthwise, 20 epochs. Final: 27.95 ms (1.67×), LPIPS 0.001825, FID 9.03, PSNR 18.22.

**Interpretation**

1. D only “wins” if the combo beats the better parent on the metric you care about. Stacking is not automatically additive.
2. **Half decoder + attention → big speed** (ECA+half 16.6 ms, GCBlock+half 17.1 ms, Triplet+half 20.1 ms) but **hurts ConvNeXt-LPIPS** in all three cases — capacity cut and attention change interfere on quality.
3. **PixelShuffle is nearly speed-neutral** (same as solo B7). With ECA it helps LPIPS/FID; with Triplet it mainly helps FID while LPIPS stays worse than solo Triplet; with GCBlock it helps neither (FID 7.06 vs parents 5.21/6.44).
4. **Depthwise stacks are the speed/quality middle ground** (~21–28 ms, 1.7–2.2×): ECA+DW trades LPIPS for FID; **GCBlock+DW is the standout — it beats both parents on LPIPS** (0.001844 vs 0.001862/0.001922) at 2.14×, taking the D LPIPS crown from ECA+PS (0.001848). **Triplet+DW keeps the attention’s LPIPS** (0.001825 ≈ A2’s 0.001817, beats CBAM) but **loses FID badly** (9.03 vs parents 6.70/7.11) — the depthwise cut interferes with FID when the attention is Triplet.
5. **Practical picks from the full A/B/D sweep**
   - Best LPIPS overall: still **solo A2 Triplet** (0.001817) — no D combo beat it.
   - Best speed with usable quality: **solo A5 ECA** (~25 ms, LPIPS≈CBAM), or **GCBlock+DW** (~21.7 ms, LPIPS 0.001844) for a faster + slightly better-quality stack.
   - Avoid **ECA+half** unless latency is the only objective (2.81× with a clear quality drop).
   - **Triplet+half** is a speed/FID compromise (~2.3×, strong FID) if you accept losing A2’s LPIPS crown.

---

## Experiment 3 — VIS-only input & decoder-capacity ladder

**As of 2026-10-22.** Five fresh 20-epoch trains on the same frozen split through the same `train_eval.sh` harness as Exp 2 (`jobs/cislnet/experiments/{visonly_c13,dec10_c13,dec25_c13,dec75_c13,encdec50_c13}.sh`; `tracking.csv` + `patch_eval_test`). Trained 2026-10-16 → 2026-10-21, all completed.

Two orthogonal cost questions:

1. **Does the model actually need IR when RGB VIS is always on?** Exp 1 kept VIS + top-N IR (`Vis+N`) and never tested the zero-IR endpoint — its ladder's worst member was Vis+1. Here we test that endpoint: **0 IR**.
2. **How much decoder is enough, and is the encoder cheap to cut too?** A straight **decoder-width ladder** 10 / 25 / 50 / 75 / 100% (Exp 2's B1 half-decoder is the 50% rung), plus one stacked run that halves **encoder and decoder together** (`encdec50`) to test whether the ResNet50 encoder is cheaper to shrink than the decoder.

### Setup

| Intuitive name | Run tag | Change | Input | Decoder width / NDF | Reference points pulled from |
| -------------- | ------- | ------ | ----- | -------------------- | ----------------------------- |
| **Full CISLNet** | `c13` | reference (ResNet50 + CBAM, full width) | 13ch (10 IR + VIS) | 1.00 / 64 | Exp 1/2 baseline |
| **VIS-only (0 IR)** | `visonly_c13` | same arch, RGB VIS only | 3ch | 1.00 / 64 | vs Full CISLNet, Exp 1 Vis+1 |
| **Vis+1 / +3 / +5** | `vp1/vp3/vp5` | _(Exp 1, prior)_ | 4/6/8ch | 1.00 / 64 | compare IR ladder |
| **Decoder 10%** | `dec10_c13` | `--decoder_width_mult 0.10`, NDF 6 | 13ch | 0.10 / 6 | vs Decoder 50% (B1), full |
| **Decoder 25%** | `dec25_c13` | width 0.25, NDF 16 | 13ch | 0.25 / 16 | vs Decoder 50%, full |
| **Decoder 50% (B1)** | `b1_half_c13` | _(Exp 2 B1, prior)_ | 13ch | 0.50 / 32 | — |
| **Decoder 75%** | `dec75_c13` | width 0.75, NDF 48 | 13ch | 0.75 / 48 | vs full |
| **Enc+Dec 50%** | `encdec50_c13` | encoder **and** decoder at 0.50, NDF 32 | 13ch | 0.50 / 32 | vs B1 (decoder-only 50%) |

![Exp 3 — how many IR bands does VIS still need (LPIPS + FID vs IR count)](./results_figures_inference/fig_ir_need.png)

![Exp 3 — decoder-width ladder: latency (left) and quality (right), plus the Enc+Dec 50% stack](./results_figures_inference/fig_decoder_ladder.png)

![Exp 3 — capacity/IR family vs prior A/D standouts on the latency–LPIPS Pareto](./results_figures_inference/fig_capacity_pareto.png)

### Results

| Intuitive name | Run tag | PSNR | SSIM | FID | LPIPS (CN) ↓ | Latency (ms) | Speedup vs full |
| -------------- | ------- | ---- | ---- | --- | ------------ | ------------: | --------------: |
| **Full CISLNet** | `c13` | 18.09 | 0.466 | 5.84 | 0.001819 | 46.6 | 1.00× |
| **VIS-only (0 IR)** | `visonly_c13` | 16.98 | 0.441 | 7.53 | 0.002055 | 46.3 | 1.00× |
| Vis+1 _(Exp 1)_ | `vp1` | 16.68 | 0.437 | 8.54 | 0.002077 | — | — |
| Vis+3 _(Exp 1)_ | `vp3` | 17.57 | 0.452 | 8.18 | 0.001928 | — | — |
| Vis+5 _(Exp 1)_ | `vp5` | 17.42 | 0.443 | 7.27 | 0.001942 | — | — |
| **Decoder 10%** | `dec10_c13` | 17.26 | 0.411 | 13.29 | 0.002015 | 26.7 | 1.74× |
| **Decoder 25%** | `dec25_c13` | 17.66 | 0.451 | 10.46 | 0.001896 | 28.4 | 1.64× |
| Decoder 50% (B1) _(Exp 2)_ | `b1_half_c13` | 17.86 | 0.461 | 7.43 | 0.001865 | 36.4 | 1.28× |
| **Decoder 75%** | `dec75_c13` | 18.01 | 0.467 | 6.55 | 0.001863 | 41.9 | 1.11× |
| **Enc+Dec 50%** | `encdec50_c13` | 17.80 | 0.446 | 7.82 | 0.001916 | 25.3 | 1.84× |

Prior standouts shown for context (Exp 2): A2 Triplet 31.8 ms / 0.001817 / FID 6.70 · A5 ECA 25.0 / 0.001861 / 6.16 · A4 GCBlock 25.7 / 0.001862 / **5.21** · GCBlock+DW 21.7 / 0.001844 / 5.75.

Sources: `results/cislnet/test/{visonly_c13,dec10_c13,dec25_c13,dec75_c13,encdec50_c13,vp1,vp3,vp5,c13,b1_half_c13}/patch_eval_test/metrics.json`; latencies from `tracking.csv` (CSV: `results_figures_inference/capacity_metrics.csv`).

> Full-CISLNet quality numbers use the current `test/c13` leaf (re-eval 2026-09-28: LPIPS 0.001819 / FID 5.84); the older `vs13` tracking row reads LPIPS 0.001871 / FID 6.08 — same promoted epoch-24 checkpoint. Latency 46.6 ms is the same-architecture `vs13` row (c13's own leaf has no timing). Re-timed runs repeat within ±0.2 ms on the same lane, but cross-job latency has node-level noise.

### Takeaways

1. **One IR band is worse than none.** VIS-only (0.002055 / FID 7.53) ties Vis+1 on LPIPS (0.002077) and **beats it on FID by 1.0** — adding exactly the 1650 nm band alone buys nothing; the top band is only useful on top of other bands. If you need to drop IR, drop **all** of it.
2. **IR is still real signal.** Going full 13ch → VIS-only costs LPIPS +13% (0.002055 vs 0.001819) and FID +1.7, with no latency savings (46.3 vs 46.6 ms — the IR stem is cheap). Dropping channels is a quality trade with no speed payoff; use it only for storage/sensor reasons, not inference budget.
3. **Decoder quality is flat down to 50%, then falls off a cliff at 25%/10%.** Decoder 75% is within LPIPS noise of full (0.001863 vs 0.001819) and only 1.11× faster; Decoder 50% is Exp 2's B1 shown again here (0.001865, 1.28×). Below 50%: FID doubles by 10% (13.29 vs 5.84) and SSIM/PSNR sag — the decoder had residual capacity that was doing perceptual work FID sees first.
4. **Halving the encoder too is almost free.** Enc+Dec 50% is **faster than every pure-decoder cut** (25.3 ms, 1.84×) while staying better than Decoder 10%/25% on every metric (LPIPS 0.001916 / FID 7.82). The ResNet50 encoder is not the bottleneck — the decoder is. It gives up ~4% LPIPS vs Decoder 50% solo (0.001916 vs 0.001865) but runs 1.4× faster; vs Decoder 10% it dominates outright.
5. **Capacity cuts are a speed play, not a quality play.** Nothing in this family touches prior LPIPS best (A2 Triplet 0.001817) or GCBlock(+DW)'s FID (5.21/5.75). For "fastest at c13-like quality" the answer is unchanged (ECA / GCBlock+DW); for "fastest at 13% LPIPS toll + zero hardware changes", Enc+Dec 50% is the new pick.
6. **WSI legs**: all five runs have finished train jobs; only VIS-only/Decoder-10%/Decoder-25% have WSI jobs run (2026-10-19, published to the `campus_ablations_bestlpips` viewer as `e20`; those WSI steps write to the online viewer, not local `wsi_pred/`). Full-size WSI for `dec75/encdec50` is not queued yet.

---

## Reproduce

```bash
# Exp 1 — Vis+
./jobs/cislnet/submit_visplus.sh

# Exp 2 — architecture
EPOCHS=20 bash jobs/cislnet/experiments/submit_abc.sh
sbatch jobs/cislnet/experiments/b4_eca.sh    # A5
sbatch jobs/cislnet/experiments/b5_simam.sh  # A6
sbatch jobs/cislnet/experiments/b6_depthwise.sh
sbatch jobs/cislnet/experiments/b7_pixelshuffle.sh
sbatch jobs/cislnet/experiments/fp16_ir.sh

# D combos (top-2 attention × top-2 structure)
bash jobs/cislnet/experiments/submit_d_combos.sh

# Exp 3 — VIS-only + decoder-capacity ladder
sbatch jobs/cislnet/experiments/visonly_c13.sh
sbatch jobs/cislnet/experiments/dec10_c13.sh
sbatch jobs/cislnet/experiments/dec25_c13.sh
sbatch jobs/cislnet/experiments/dec75_c13.sh
sbatch jobs/cislnet/experiments/encdec50_c13.sh

# figures
python scripts/plot_inference_results.py    # Exp 1 + 2
python scripts/plot_capacity_results.py     # Exp 3
```

Plain c13 patch leaf: `NUM_CHANNELS=13 RUN=c13 NAME=c13 sbatch jobs/cislnet/test/test.sh`.
