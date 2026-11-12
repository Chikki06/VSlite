# Research Dossier — CISLNet: 13-channel IR+VIS → Virtual H&E Staining

**Audited:** `/projects/illinois/eng/bioe/rxb/akshat/VSlite` (git head `59c2569`, active Feb–Oct 2026)
**Evidence legend:** **[V]** verified in repo artifacts (code + metrics/tracking files opened during this audit) · **[R]** reported in docs without surviving independent evidence · **[H]** hypothesis.

---

## 1. Executive summary

You built **CISLNet**, a contrastive-GAN image-to-image translator that turns **10-band multispectral infrared + registered visible-light pairs from clinical frozen-section slides into synthetic H&E** (the functionally "virtual staining" direction used for intraoperative margin workflows). Data is real clinical material hosted by the Cancer Center at Illinois / Beckman CISL inference viewer (`remote_upload.md:8`): 239 slides × 3 modalities, ~60k train / 6,929 test 1024² patches over 194 slides.

Three experiment campaigns, all reproduced from artifacts:
≠
1. **Exp 1 — Vis+ IR-band ladder** (Vis+1/+3/+5 vs full 13ch, plus the zero-IR endpoint): full 13ch remains best on the custom perceptual metric; Vis+3 is the efficient point; **adding exactly one IR band is worse than none on FID**; dropping IR gives **no** latency savings.
2. **Exp 2 — factorized latency × quality architecture sweep** (6 skip-attention variants × 5 structure/capacity knobs + 9 interaction "D"-stacks): attention choice dominates the Pareto. Your quoted 46.59 ms (CBAM baseline), 25.03 ms (ECA), 21.73 ms (GCBlock+depthwise) all **verify exactly** against `results/cislnet/experiments/tracking.csv` and per-run `metrics.json`.
3. **Exp 3 — VIS-only + decoder/encoder capacity ladder**: quality is flat to 50% decoder width then collapses; the ResNet50 encoder is genuinely cheap to halve.

Plus two side thrusts: a **matched-optimizer-step organ-holdout study** (all-samples vs P+K+U pools vs organ-only at a frozen 1.59M-sample budget — organ-specific wins FID on all three organs) and a **quantization/deployment thread** (int8/fp8 essentially loss-free, int4/fp4 collapse).

**Strongest honest claim set:** (a) lightweight attention swaps (ECA/GCBlock) + depthwise decoder give a verified ~1.8–2.8× inference-speedup frontier at equal-or-better perceptual quality; (b) the IR-sufficiency curve including the Vis+1 pathological endpoint; (c) organ-specific training halves FID at matched compute.
**Biggest methodological gaps:** everything is **n=1 run, no seed control, no CIs/significance tests**; the headline "best-LPIPS" margins between some configs (e.g., Triplet 0.0018170 vs CBAM 0.0018710) are **within measurement noise**; the organ-holdout has **patient-sibling overlap** (e.g., `3899-8` holdout vs `3899-1..9` train); and the perceptual metric partially re-uses the training-loss encoder. Details and fixes below.

---

## 2. Task, data, and integrity of splits

**Task.** Input → H&E: input = 10 IR bands (disk order `1081, 1140, 1237, 1275, 1401, 1480, 1547, 1585, 1650, 1765 nm`, confirmed in `tissue_images.db → tissue__images.bands`) × 3 VIS RGB = 13 channels; target = 3-ch H&E. IR native `(354, 353, 10)` @ 2 µm/px is **cubically resized to 1024²** every load (`data/VS_dataset.py:570–581`; AGENTS.md's `472–482` cite has drifted); VIS/HE are uint8 `(1024,1024,3)` @ 0.69 µm/px. Patches were cut **outside the repo** (external prep volume, `trainset_maker/datafinder.py` in git history `4bd7c38`) on a **non-overlapping 1024-px lattice in VIS space**, background rejected.

**Frozen split [V].** `dataset/data_frozen/test.txt` = slide prefixes `4466`(Lung), `10445`(Kidney), `4520`(Prostate ×4 blocks), `10442`(Breast), `3840`(Prostate ×6 blocks) → 6,929 patches / 17 slides / 5 patients; train = 60,047. Loaders split **slide-level** (`data/VS_dataset.py:332–363`); val is deliberately present-but-empty (no checkpoint selection). A no-leakage checker exists and runs as a unit test (`scripts/check_no_leakage_test_txt.py`, `tests/test_no_leak_test_txt_split.py`). **Patient-level overlap for the canonical split: none found** (whole patient families are on the test side).

**Organ-exp splits [V, with a real confound].** `scripts/build_organ_exp_lists.py:31–43` hardcodes holdouts (Prostate 2,379 / Kidney 807 / Uterus 538 patches ≈ 11.6% each), selected by an offline seed-42 search — seed only used for search, not runtime. Slide-level: clean. **Patient-level: contaminated** — holdout `3899-8` has siblings `3899-1..9` in train; same for `3932-3`, `3945-3`, `4249-2`, `3974-3`, `5387-3`. Serial blocks of the same patient share scanner/processing style, so part of organ-specific FID gains could be patient-style overfitting, not organ knowledge. This needs a patient-block split before publication.

---

## 3. Architecture (reconstructed) [V]

Text diagram (13ch, as-run canonical):

```
input 13ch (10 IR + 3 VIS RGB), 1024²
  └ ResNet50 ImageNet encoder, conv1 → 13ch (pretrained-width truncation for enc_mult<1)
      taps: conv1(64) → layer1(256) → layer2(512) → layer3(1024) → layer4(2048)  [stride /32]

  CBAM attention on bottleneck (2048)
      ├ Ups1: ConvT(2048→1024) ⊕ CBAM(feat2) → Conv3×3 ×2 (BN+ReLU)  → 1024
      ├ Ups2: ConvT(1024→512) ⊕ CBAM(feat1)                          → 512
      ├ Ups3: ConvT(512→256)  ⊕ CBAM(feat0)                          → 256
      ├ Ups4: ConvT(256→64)   ⊕ CBAM(feat)                           → 64
      └ final: ConvT(64→32) → ReflectionPad3 + Conv7×7(3) → Tanh  ── H&E [-1,1]
```

- **G = `AttUnetGenerator`** `models/networks/generator.py:331–588`. Skip attention sits at bottleneck + every skip (5 modules). `decoder_width_mult` with 1×1 projectors (`:413–418`), `encoder_width_mult` via `_slim_resnet50_width` truncating pretrained kernels (`:272–328`) — a nontrivial original implementation detail. Alternative decoder convs: SCConv, depthwise-separable (`Up`, `:692–751`); final-up PixelShuffle option (`:445–480`).
- **Attention blocks** `models/networks/attention_blocks.py:311–343`: CBAM (in-tree, `generator.py:612–662`), CoordAtt, Triplet, SCSE, GCBlock (ContextBlock2d att-pool + channel_add/mul), ECA (k=3), SimAM — all vendored/adapted from public MIT repos (headers at `:3–9`). Unit test exists (`tests/test_cbam_block.py`).
- **D = Multiscale PatchGAN ×2, n_layers=4, ndf=64, spectral+InstanceNorm; sees only 3-ch H&E** (`models/networks/discriminator.py:102–108` hard-returns 3).
- **F_q = frozen ConvNeXt-Tiny, TCGA-fine-tuned** (`models/networks/contrastive_encoder.py:70–80` → `ConvNeXtFeatureExtractor`, `models/networks/architecture.py:132–223`); trainable MLP head H (256→256) + L2 norm → CUT-style **bidirectional patch-InfoNCE**, 1024 patches/layer, τ=0.07, same-image negatives + query feats (`models/networks/nce.py:8–83`). **Quirk:** `nce_layers='1,2,3,4,5'` yields only 4 feature maps (`out[1:]`; the "layer 5" doesn't exist) and the mean divides by 5 — an effective ~20% NCE-magnitude mismatch (`models/contrastive_pix2pix_model.py:491–523`).
- **As-run loss = LSGAN(λ=1) + PatchNCE(λ=10)** — L1/perceptual/ganFeat **off by default** (`options/train_options.py:196–262`; `README.md:164`). Adam 2e-4, β=(0.5,0.999), wd=0, **constant LR** (the `total_epochs` path overrides decay, `scripts/train.py:44–48`), batch 10, no grad accumulation (only logged), no EMA.
- Alternative **encoder = timm `convnext_tiny.in12k_ft_in1k`** features_only, stem conv rebuilt for 13ch, 4 maps [96,192,384,768] (B3 run).

So: CISLNet is not a genuinely novel *architecture*; it is a well-engineered **ResNet50/CBAM attention-U-Net under NVIDIA CUT (contrastive translation) machinery adapted to 13-channel supervised translation**. The novelty lives in the input modality, the evaluation/ablation system, and the empirical curves.

---

## 4. The evaluation system — ConvNeXt-TCGA LPIPS & FID

**[V] Not ordinary pretrained LPIPS.** `util/lpips_convnext.py:1–138` is a from-scratch reimplementation of Zhang's **lin0 recipe on a ConvNeXt-Tiny fine-tuned on TCGA H&E** (`tcga_tl_checkpoints/convnext_t_dp_spv_supcon_1e-5`, 27.8M params, head stripped):

- Taps = 5 maps: stem + 4 stage outputs (`models/networks/architecture.py:173–177`).
- Per-spatial-position channel unit-norm → squared diff → mean over H×W×C → **equal-weight mean over 5 stages** (`util/lpips_convnext.py:70–85,138`).
- Deviates from official LPIPS: **no NetLin learned 1×1s, no ScalingLayer, no calibration** → numbers are not on the official LPIPS scale and are not comparable outside the repo (documented at `README.md:451`, `results_c13.md:24`).
- Input norms: [-1,1] → [0,1] → ImageNet mean/std; runs at **native patch resolution** (224-resize commented out at `architecture.py:216–217`), while **FID resizes every patch (even 1024²) to 224²** (`util/i2i_metrics.py:102–104`) — the two metrics are sensitive to different detail levels by construction.
- **FID** = Fréchet distance on that same frozen encoder's GAP-768-d final-stage features, streamed with online float64 mean/cov (`util/i2i_metrics.py:126–151, 428–441`); legacy path `evaluation/fid_evaluator.py` (borrowed from mseitzer/pytorch-fid) uses an activation cache — the known "instantly complete" gotcha.
- Checkpoint provenance [R]: filename implies drop-path/supervised-contrastive FT at lr 1e-5; **the fine-tuning script, dataset protocol, and license are not in the repo** — externally produced artifact, must be documented before publication.
- **Circularity risk (real, bounded):** the *same frozen encoder* is (a) the NCE training-loss feature extractor (stage taps; stem dropped), (b) the LPIPS backbone (all 5 taps), (c) the FID backbone, and (d) the optional training perceptual loss for the other stacks. For CISLNet the coupling is weaker (InfoNCE-over-MLP vs raw stage diffs) but nonzero. The VGG-LPIPS cross-check exists **only** in the exp-eval path (`scripts/cislnet_exp_eval.py:216–249`), not the canonical `vs_infer_and_metrics.py` output.

Latency claim setup [V]: `util/exp_metrics.py:35–75` — CUDA-synchronized `perf_counter` mean/std over **10 warmup + 100 timed forwards**, dummy `randn(1, C, 1024, 1024)`, batch=1, fp32, **A100** (`jobs/cislnet/experiments/train_eval.sh:7`, node ccc0388 excluded). Same-job re-timings recorded in tracking.csv repeat within **±0.2 ms**; cross-job/node noise is the dominant uncertainty (`results_inference.md:238`).

---

## 5. Full ablation table [V]

All numbers verified against `results/cislnet/experiments/tracking.csv` + `results/cislnet/test/<run>/patch_eval_test/metrics.json`, n=6,929 everywhere. All ablation trains: **20 epochs, batch 10, 1024², frozen split**; reference = c13 (epoch-24 ckpt of a 30-ep train → `latest`; LPIPS 0.001871 is the `vs13` tracking row, 0.001819 is the c13-leaf re-eval of the same checkpoint).

### Group A — skip attention (structure = default)

| Run | Attention | LPIPS-CN ↓ | FID ↓ | PSNR | SSIM | ms | × vs CBAM | ΔLPIPS vs CBAM (SE≈7.7e-6) |
|---|---|---|---:|---:|---:|---:|---:|---|
| vs13 (ref) | CBAM | 0.001871 (leaf re-eval 0.001819) | 6.08 (5.84) | 18.00 | 0.461 | **46.59±0.06** | 1.00× | — |
| a1 | CoordAtt | 0.001940 | 7.17 | 17.49 | 0.448 | 33.87 | 1.38× | +3.7% |
| a2 | Triplet | **0.001817** | 6.70 | 18.10 | 0.479 | 31.80 | 1.46× | **−0.10 % (≈noise)** |
| a3 | SCSE | 0.001890 | 6.43 | 17.78 | 0.458 | 25.68 | 1.81× | +1.0% |
| a4 | GCBlock | 0.001862 | **5.21** | 17.98 | 0.465 | 25.71 | 1.81× | −0.5% (noise-adjacent) |
| a5 (`b4_eca_*`) | ECA | 0.001861 | 6.16 | 17.93 | 0.460 | **25.03** | **1.86×** | −0.5% (noise-adjacent) |
| a6 (`b5_simam_*`) | SimAM | 0.001928 | 7.88 | 17.69 | 0.448 | 26.51 | 1.76× | +3.0% |

### Group B — structure/capacity (attention = CBAM) + C / IO

| Run | Change | LPIPS-CN ↓ | FID | PSNR | SSIM | ms | × | note |
|---|---|---|---:|---:|---:|---:|---:|---|
| b2 | SCConv decoder | 0.001961 | 7.49 | 17.57 | 0.445 | 83.21 | **0.56×** | slower than baseline; std 18.1 outlier |
| b3 | ConvNeXt-Tiny encoder | 0.001915 | 9.28 | 17.73 | 0.467 | 48.44 | 0.96× | |
| b7 | PixelShuffle final up | 0.001902 | 6.44 | 17.82 | 0.450 | 46.81 | 1.00× | |
| c1 | linear HE color (**negative**) | 0.003089 | 53.29 | 12.26 | 0.299 | 46.62 | 1.00× | unusable; earlier NaN run dir `c1_linear_c13.nan_bad_20261003_161131` |
| fp16 (IO) | on-disk IR fp16 | 0.001861 | 7.27 | 17.94 | 0.458 | 46.62 | 1.00× | data path only, not a D knob |
| b6 | depthwise decoder (DW3×3+PW1×1) | 0.001922 | 7.11 | 17.70 | 0.442 | 42.62 | 1.09× | |
| b1 | half decoder + ndf32 | 0.001865 | 7.43 | 17.86 | 0.461 | 36.44 | 1.28× | |

### Group D — interaction stacks (fresh 20-ep trains)

| Combo | ms | × | LPIPS ↓ | FID | vs parents |
|---|---:|---:|---:|---:|---|
| Triplet+half | 20.14 | 2.31 | 0.001891 | 5.41 | faster than both; LPIPS worse than A2 |
| Triplet+PS | 31.84 | 1.46 | 0.001889 | **5.16 (best FID in D)** | FID wins |
| ECA+half | **16.58** | **2.81** | 0.001962 | 7.60 | fastest overall; quality regresses |
| ECA+PS | 24.95 | 1.87 | 0.001848 | 5.24 | LPIPS beats CBAM/ECA/B7 — positive stack |
| GCBlock+half | 17.09 | 2.73 | 0.001905 | 7.21 | half-decoder interference again |
| GCBlock+PS | 25.70 | 1.81 | 0.001877 | 7.06 | neither metric helps |
| ECA+DW | 20.96 | 2.22 | 0.001913 | 5.90 | **FID beats both parents** |
| **GCBlock+DW** | **21.73** | **2.14** | **0.001844** | 5.75 | **beats both parents on LPIPS** — best-LPIPS D stack |
| Triplet+DW | 27.95 | 1.67 | 0.001825 | 9.03 | LPIPS held (~A2); FID collapses |

Selection logic [V]: `scripts/select_pareto_d1.py` ranks top-2 attentions × top-2 structures by ConvNeXt-LPIPS then latency; 4 combos + 5 follow-ups all completed (env files in `results/cislnet/experiments/d_combos{,_followup}/`).

### Exp 3 — capacity ladder & VIS-only (re-verified in metrics.json)

| Config | LPIPS ↓ | FID | PSNR | SSIM | ms | × |
|---|---:|---:|---:|---:|---:|---:|
| c13 full | 0.001819 | 5.84 | 18.09 | 0.466 | 46.6ᵀ | 1.00 |
| VIS-only (0 IR) | 0.002055 | 7.53 | 16.98 | 0.441 | 46.3 | 1.00 |
| Decoder 75% (ndf48) | 0.001863 | 6.55 | 18.01 | 0.467 | 41.9 | 1.11 |
| Decoder 50% (=B1) | 0.001865 | 7.43 | 17.86 | 0.461 | 36.4 | 1.28 |
| Decoder 25% | 0.001896 | 10.46 | 17.66 | 0.451 | 28.4 | 1.64 |
| Decoder 10% | 0.002015 | 13.29 | 17.26 | 0.411 | 26.7 | 1.74 |
| Enc 50% + Dec 50% | 0.001916 | 7.82 | 17.80 | 0.446 | 25.3 | **1.84** |

ᵀ Latency from the same-architecture `vs13` row (c13's own leaf was never timed).

**Attempted–not-completed:** patch-size sweep (`jobs/cislnet/submit_patch_sweep.sh`: 512 @27ep ✅ trained; 1024 @12ep ⚠; 2048 @3ep ⚠, batch 2) — **no patch-eval metrics ever produced**, only an ad-hoc NCE-vs-epoch figure (`results/cislnet/patch_sweep_nce_vs_epoch.png`) whose y-scales are not comparable across sizes. Count as an unfinished experiment, not a finding.

---

## 6. Reference-latency verification (the three numbers you want on a resume)

| Claim | Value | Evidence [V] |
|---|---|---|
| CBAM baseline | **46.59 ms** | `vs13` tracking row: `latency_ms_mean=46.591014, std=0.0576`, ckpt=c13 `latest_net_G.pth`; 10 warmup + 100 timed, CUDA sync, 1024² batch-1 fp32, A100 |
| ECA attention | **25.03 ms** | `b4_eca_c13` row: `25.03257`, std 0.042 |
| GCBlock × depthwise | **21.73 ms** | `d1_resnet50_gcblock_c13_dw`: 21.73324 / duplicate re-eval 21.70056 (±0.03 same job) |
| Fastest-but-lower-quality | ECA+half **16.58 ms** (2.81×, LPIPS 0.00196); GCBlock+half 17.09 ms; dec25 28.4 ms; Enc+Dec50% **25.3 ms @ 1.84×** better than dec10/dec25 on every metric | [V] |

**Comparability caveats to state:** (i) ablations were all 20-ep trains vs a 24-epoch-trained reference ckpt (slightly inflates reference quality; state training-length parity or re-verify at matched epochs); (ii) latency is pure GPU forward on a random tensor — no data loading, no fp16/bf16, no compiler passes, batch=1; (iii) A100-only, one excluded node, no TF32/compile sensitivity report; (iv) the quantization thread's per-patch timings (scavenger/H100, batch 16–32) are **not** comparable to this table and should not be mixed. [V]

**Fastest-at-baseline-quality standing answer [V, but n=1]:** solo ECA (~25 ms, LPIPS ≈ CBAM) or GCBlock+DW (~21.7 ms, LPIPS 0.001844 beats both parents) — pending repeats and a matched-epoch re-check.

---

## 7. IR channel selection (Exp 1) [V] + provenance gaps

| Config | Channels | LPIPS-CN ↓ | FID | PSNR | Note |
|---|---|---:|---:|---:|---|
| c13 | Vis + **10 IR** | **0.0018188** | **5.84** | 18.09 | best on both |
| vp3 | Vis + 3 IR (1650, 1081, 1765) | 0.0019278 (+6.0%) | 8.18 | 17.57 | best Vis+ |
| vp5 | Vis + 5 IR (+1140, +1547) | 0.0019417 (+6.7%) | 7.27 | 17.42 | |
| vp1 | Vis + 1 IR (1650) | 0.0020767 (+14.2%) | 8.54 | 16.68 | |
| VIS-only | Vis + 0 IR | 0.0020548 (+13.0%) | **7.53** | 16.98 | **beats vp1 on FID** |
| vs10 (legacy 15-ep) | 10 IR, no VIS | 0.0020371 | 6.26 | 17.29 | older ckpt, not epoch-matched |
| vs6 (legacy 15-ep) | 3 IR (0,8,9) + VIS | — (no key) | 7.13 | 17.27 | predates LPIPS logging |

Channel index mapping [V]: `jobs/_lib/cislnet_channels.sh:9–60` (VIS+1→`8`; +3→`8,0,9`; +5→`8,0,9,1,6`), consumed by `data/VS_dataset.py:536–568`, concat policy `:655–691`.

Verified findings (SE on LPIPS mean ≈ 7.7e-6, so ≥±6% deltas are solid):

1. **IR is real but small signal**: 13ch → VIS-only costs +13% LPIPS and +1.7 FID with zero latency payoff (IR stem is cheap).
2. **The Vis+1 endpoint is a pathology [striking]**: adding exactly one IR band is worse than none on FID (8.54 vs 7.53) and roughly ties on LPIPS — the 1650 nm band only helps when other bands anchor it.
3. **Vis+3 is the efficiency point**: −6% LPIPS vs full at 6ch.
4. **Budget caveat [must fix]**: c13 was evaluated from a 24-epoch checkpoint; vp runs were 20-epoch trains — the c13-vs-vp gap partly includes ~4 extra epochs of training.
5. **Provenance gap [H]**: the wavelength-priority order `1650 → 1081 → 1765 → 1140 → 1547` is asserted in `submit_visplus.sh:4` and `results_inference.md:32` but **no experiment/document explains how these five wavelengths were chosen**. The channel-importance claim is supported *after* the ladder ran, not as a tested hypothesis.

---

## 8. Organ-specific vs pooled training

**Design [V]** (`scripts/organ_exp_schedule.py`, `jobs/cislnet/submit_organ_exp.sh`): shared architecture/hyperparams (1024², batch 10, lr 2e-4) trained matched on **sample count** (1.5948M samples; batch-10 optimizer steps): all-slides 30 passes (53,164 patches), P+K+U 59 passes (26,944), prostate-only 93 (17,183), kidney-only 282 (5,653), uterus-only 415 (3,840). Holdout = the ~11.6% organ subsets (2,379/807/538), **slides disjoint from train** but **patients not** (§2). Eval grid at matched iters (27k→159k), 3-day walltime hops with afterany resume; two invalid evals from missing P+K+U ckpts 39/49 correctly excluded.

**Results at budget [V, `results_figures_organ/holdout_metrics.csv`]:**

| Comparison (159k iters) | Value |
|---|---|
| Prostate FID (all → OS) | 4.81 → **3.92** |
| Kidney FID (all → OS) | 12.24 → **7.47** |
| Uterus FID (all → OS) | 25.35 → **10.50** |
| LPIPS | nearly flat across protocols; all-samples even best on uterus (0.00224 vs 0.00230) |
| PSNR/SSIM | no consistent winner (e.g., uterus SSIM 0.360 all vs 0.327 OS) |

**Honest interpretation:**

- The claim that survives is narrow: **at matched optimizer steps, organ-specific training improves holdout FID (distribution-level realism), not perceptual distance.** Effect largest for uterus (n=538 — also the noisiest FID; no CIs computed).
- **Confound:** patient-sibling overlap means OS gains may partly be patient-style memorization. A patient-blocked rerun is the single most valuable fix.
- **Both metrics are ConvNeXt-TCGA-based** — not comparable to Inception-FID paper numbers; LPIPS partially overlaps the training-loss encoder.
- Single training run per protocol; conclusions rest on one FID point per cell.

---

## 9. Rigor audit

**[V] Absent from the entire codebase:** a `--seed` option (grep in `options/` = zero hits). All runs inherit default torch RNG: different runs ≠ reproducible; **no repeated training runs of any config** (D-stack rows are re-*evaluations*, not re-trains). No bootstrap CIs, no paired tests; FID reported point-estimate only (fine at n=6,929; problematic at n=538 uterus). No seed-variance on any headline number.

**[V] Evaluation-level variance that does exist:** latency std within run 0.04–0.35 ms with two eval passes per D run agreeing within ±0.2 ms; LPIPS/PSNR/SSIM per-patch stds are stored (`*_std` in every `metrics.json`); a2-vs-CBAM LPIPS delta (0.0000018) ≪ SE (~7.7e-6) — **"Triplet is best LPIPS" is currently noise-level**; "GCBlock lowest FID 5.21 vs 6.08" is a 14% gap that plausibly survives but deserves bootstrap CIs.

**[V] Provenance wrinkles:** the canonical c13's own 18-epoch eval sweep (`submit_c13_epoch_eval.sh`) **never produced metrics** (`results_c13.md:9`: "0 / 18"); cited c13 numbers come from one 2026-09-28 re-eval of the promoted epoch-24 ckpt + the older `vs13` row. The 15-epoch-legacy family (`vs10/vs6/c13_e15_legacy`) mixes epochs (15) with the 20/24-epoch families. A NaN episode in the c1-linear run left `c1_linear_c13.nan_bad_20261003_161131`; the retrain was clean.

**[V] Things done right (keep and cite):** streaming FID with online covariance; slide-level split with an automated no-leakage unit test; invalid/broken evals explicitly excluded; missing-checkpoint audit; fixed preview-tile protocol for cross-run qualitative comparison (`util/fixed_preview.py`); unified tracking CSV with double-pass latency; region-eval for >patch-size inputs (`cislnet/region_eval.py`); a real unit-test suite (`tests/`: CBAM block, projection layers, CLI mappings, color space, leakage).

**[V] Failure-mode evidence that exists but is not written up:** c1-linear collapse (FID 53 — color space matters); b2-SCConv slower than baseline; half-decoder × attention interference (quality regression in 3/3 stacks); int4/fp4 quantization collapse; no per-slide variance table or failure gallery anywhere.

---

## 10. Side thread: quantization/deployment [V]

`results/cislnet/quant/c13/test/*/patch_eval_test/metrics.json` (H100, scavenger; **different timing method — do not merge with the ms table**):

| Precision | Scheme | FID | PSNR |
|---|---|---:|---:|
| int32 (fp32 baseline) | — | 6.08 | 17.99 |
| **fp8 e4m3 native** | per-channel | **5.81** | 17.99 |
| **int8** | packed per-channel | **5.81** | 18.02 |
| int4 | packed | 75.17 | 15.82 |
| "fp4" | int4-nibble (mislabeled) | 138.72 | 9.75 |

QAT variant exists (`quant/c13/qat_8/quant_ablation.json`: 5-epoch QAT, lr 1e-5, frozen D + netF, fake-quant weights+activations). **Claimable [H pending writeup]: int8/fp8 PTQ is accuracy-preserving for virtual staining** — a useful deployment datum currently invisible in the results docs.

---

## 11. Original-contribution map

**Established/inherited [V]:** CUT/contrastive-pix2pix framework + SPADE code lineage (NVIDIA headers, CC BY-NC-SA); AttUNet+CBAM (Woo & Oktay 2018); vendored CoordAtt/Triplet/SCSE/GCNet/ECA/SimAM (public MIT); SCConv (CVPR'23 vendored); Zhang LPIPS concept; Fréchet distance (pytorch-fid lineage); ConvNeXt backbone; per-channel PTQ.

**Original [V]:**
1. The **13-channel (10-band MS-IR + registered VIS) → H&E formulation on clinical frozen-section data** — a data regime no public dataset matches.
2. The **unified ConvNeXt-TCGA metric system** (histopathology-domain lin0 LPIPS reimplementation + streaming GAP-FID + train/test-consistent interfaces), including explicit non-comparability discipline.
3. The **factorized A×B latency-ablation design + D interaction study** on a real 1024² translation model (width projectors + pretrained-encoder-kernel truncation are original engineering).
4. The **Vis+ IR-sufficiency ladder including the zero-IR endpoint** and its counterintuitive vp1 result.
5. The **matched-optimizer-step organ-holdout protocol** (shared sample budget + matched checkpoint grid + holdout-only-as-test discipline).
6. The **capacity ladder's encoder-halving trick** (`_slim_resnet50_width`) and the "encoder cheaper than decoder" finding.
7. The **PTQ/QAT int8/fp8 result on a stain model**.

**Where novelty is weak:** any architecture block (all borrowed); "best-LPIPS" rankings at 0.1% margins (noise-level); latency numbers without repeats.

---

## 12. The three most scientifically significant findings (incl. negatives)

1. **A lightweight-attention + depthwise-decoder frontier for 1024² stain translation.** Swapping CBAM for ECA/GCBlock cuts inference ~1.8× at LPIPS parity (25.0/25.7 vs 46.6 ms); GCBlock+depthwise Pareto-dominates both parents (21.7 ms, LPIPS 0.00184). Negative counterweight: attention + half-decoder always interferes (3/3 stacks quality-regress) and SCConv is *slower* than baseline. (Confidence: deltas high; exact rankings n=1.)
2. **The IR-sufficiency curve — including the single-band pathology.** 13ch > Vis+3 (+6%) > Vis+5 ≈ > Vis-only (+13%); **one band alone is worse than none on FID** (8.54 vs 7.53); dropping IR saves no latency. Verdict: IR is cheap signal, not a compute knob. (Solid vs SE; training-length mismatch caveat.)
3. **Organ-specificity at matched compute — a distribution effect, not a perceptual one.** Organ-only training halves FID on kidney/uterus at the same 1.59M-sample budget while LPIPS/PSNR/SSIM stay flat and P+K+U pooling does not help. Patient-sibling overlap + n=1 acknowledged; needs one confound-fixed rerun.

**Negative results with evidence:** linear H&E color space unusable (FID 53); ConvNeXt-encoder swap not a win (FID 9.3); P+K+U pooling underperforms on uterus; decoder cuts below 50% fall off a cliff (FID doubles at 10%); 10ch < 13ch consistently across eras.

---

## 13. Six resume bullets (accurate, evidence-backed)

I cannot see your existing resume; below each bullet notes **the kind of generic bullet it can replace**. Every claim traces to a cited artifact.

1. *Replaces generic "built GAN/U-Net pipelines" bullets.*
   "Built CISLNet, a 13-channel multispectral-IR + visible-light → virtual-H&E translator (ImageNet-pretrained ResNet50/CBAM U-Net, LSGAN + bidirectional patch-InfoNCE on a frozen ConvNeXt encoder) trained on ~60k clinically registered 1024² patches with a slide-level leak-checked test split (n=6,929)."
2. *Replaces generic "optimized model inference/latency" bullets.*
   "Designed a factorized latency×quality ablation (6 skip-attention × 5 structure knobs, 9 interaction stacks) on real 1024² slides: CBAM 46.6 ms → ECA 25.0 ms (1.86×, quality parity) → GCBlock+depthwise 21.7 ms (2.14×, beats parents on the perceptual metric); uncovered interference effects (half-decoder × attention) and a slower-than-baseline variant (SCConv)."
3. *Replaces generic "used standard metrics" bullets.*
   "Engineered the evaluation stack end-to-end: histopathology-domain LPIPS (Zhang lin0 recipe on a TCGA-fine-tuned ConvNeXt-Tiny), streaming FID with online covariance over 6.9k patches, CUDA-synced batch-1 latency harness (10 warmup/100 timed), VGG-LPIPS cross-checks, and per-patch variance logging."
4. *Replaces generic "ran ablations" bullets.*
   "Mapped the IR-sufficiency curve (Vis+1/+3/+5 vs full 10-band IR vs zero-IR): full spectrum best (LPIPS 0.00182 vs 0.00205 VIS-only; −6% for Vis+3); identified a pathological single-band endpoint (1650 nm alone worse than no IR on FID) — actionable for sensor/storage tradeoffs; IR yields no inference savings."
5. *Replaces generic "improved medical imaging quality" bullets.*
   "Ran a compute-matched (1.59M-sample, matched optimizer steps) organ-holdout study (prostate/kidney/uterus) showing organ-specific training halves FID vs pooled training while perceptual metrics stay flat; isolated data-hygiene issues (patient-sibling overlap, missing checkpoints) with an excluded/broken-eval audit."
6. *Replaces "deployed models"/"quantized networks" bullets.*
   "Validated deployment-grade post-training quantization on the stain model: int8/fp8 preserve FID (5.8 vs 6.1 fp32) and PSNR while int4 collapses (FID 75); measured on H100 with a 5-epoch QAT variant — a quantized-sensitivity study for a 13-channel virtual-staining network."

---

## 14. Missing evidence (for a serious paper / internship)

**Must-have (all verified-absent):**
1. **Seed control + ≥3 repeats** on: c13, a2, a4/a5, GCBlock+DW, ECA+half, and the three organ pools; bootstrap CIs / paired tests on LPIPS/FID deltas (esp. the 0.1%-margin LPIPS claims and uterus n=538 FID).
2. **Patient-blocked organ split** rerun (no sibling slides across sides).
3. **Independent metric validation**: correlation of lin0-ConvNeXt-TCGA LPIPS vs official LPIPS (Alex/VGG) and ideally a small pathologist-reader study; document the TCGA checkpoint's training provenance (dataset, labels, license — currently an undocumented external artifact).
4. **Matched-epoch re-eval** (or retrain to 30 ep) for all 20-epoch ablations and vp runs vs c13@24; complete the c13 even-epoch sweep (currently 0/18 metrics.json).
5. **WSI-level quantitative scoring** (slide-crop PSNR/SSIM/FID + per-slide table with failure examples; today ablations exist only as qualitative viewer outputs, `wsi_pred/` empty for most).
6. **Latency methodology hardening**: single-node rerun of all variants; fp16/bf16 + torch.compile/TensorRT columns; batch impact; reconcile with the quant thread's per-patch timings.
7. **Patch-size sweep completion** or removal from all claims.
8. **External-baseline comparison**: official CUT, pix2pix/plain pix2pix, ≥1 diffusion baseline under the frozen split with the same metrics (currently internal-only).

**Should-have:** per-slide variance + failure-case gallery; significance test for "GCBlock best FID"; justification of the 1650→…→1547 wavelength order; the `nce_layers` phantom-layer fix + sanity retrain; a status doc for the quant thread.

---

## 15. Unresolved questions

1. **How was the IR wavelength priority (1650, 1081, 1765, 1140, 1547) chosen?** Physics rationale (SNR/absorption bands), earlier data-driven scan, or heuristics that the later Vis+ ladder validated? Determines whether Exp 1 is hypothesis-testing or post-hoc.
2. **Do repeated runs / saved seeds exist anywhere** (remote scratch, viewer logs, wandb)? If not, is one re-seed sweep feasible first?
3. **TCGA ConvNeXt lineage**: who trained `convnext_t_dp_spv_supcon_1e-5`, on what data/labels, is the license clear? Is the same encoder as loss-feature + metric acceptable, or should LPIPS move to a second backbone (torchmetrics VGG path is already wired)?
4. **Organ-exp framing**: "distribution shift" (FID story) vs "organ morphology learning" (needs LPIPS wins or a reader study)? Current evidence supports only the former; the sibling-overlap rerun should come first.
5. **Target venue** shapes the fix order: computational-pathology paper (reader study + WSI scores + pathology baselines), systems/deployment paper (latency + quant thread with proper methodology), or ML-methods paper (repeats/CIs, de-emphasize architecture novelty). Which is the plan?
6. **Was the half-decoder (B1) ever paired with CBAM at 30 epochs?** And should the D-stack winners be promoted through the WSI pipeline (currently viewer-side only) to produce the missing qualitative evidence?

---

## Appendix — evidence ledger

| Artifact | Path |
|---|---|
| Master tracking table (all sweep numbers) | `results/cislnet/experiments/tracking.csv` |
| Per-run metrics (verified) | `results/cislnet/test/<run>/patch_eval_test/metrics.json` (n=6929 each) |
| Latency harness | `scripts/cislnet_exp_eval.py:196–213`; `util/exp_metrics.py:35–75`; A100 via `jobs/cislnet/experiments/train_eval.sh:7` |
| LPIPS-TCGA | `util/lpips_convnext.py:70–138`; taps `models/networks/architecture.py:132–223`; ckpt resolution `:34–67` |
| FID streaming | `util/i2i_metrics.py:21–151, 428–441`; legacy cached path `evaluation/fid_evaluator.py` |
| Metric docs/caveats | `results_inference.md:9`; `results_c13.md:24`; `README.md:451` |
| Generator / attentions | `models/networks/generator.py:331–588, 612–662, 692–751`; `models/networks/attention_blocks.py:311–343` |
| Training config as-run | `options/train_options.py:196–262`; `models/contrastive_pix2pix_model.py:26–108, 210–241`; `scripts/train.py:44–48, 108–133` |
| Channels | `jobs/_lib/cislnet_channels.sh:4–60`; `data/VS_dataset.py:526–691`; band source `tissue_images.db → tissue__images.bands` |
| Split & leakage | `dataset/data_frozen/test.txt`; `data/VS_dataset.py:286–363`; `scripts/check_no_leakage_test_txt.py`; `tests/test_no_leak_test_txt_split.py` |
| Organ protocol | `scripts/build_organ_exp_lists.py:27–47`; `scripts/organ_exp_schedule.py`; `jobs/cislnet/submit_organ_exp.sh:127–286`; stats `dataset/organ_exp/manifest.json:25–67` |
| Organ results | `results_figures_organ/holdout_metrics.csv`; `results_organ.md:87–156` |
| D-stack selection | `scripts/select_pareto_d1.py`; `results/cislnet/experiments/d1_selection.md`; env files `d_combos{,_followup}/` |
| Quant thread | `results/cislnet/quant/c13/{fp8,int8,int4,fp4,qat_8}/...`; jobs `jobs/cislnet/quant/*.sh` |
| Unfinished | c13 epoch sweep (0/18 metrics, `results_c13.md:9`); patch sweep (`jobs/cislnet/submit_patch_sweep.sh`; ps512 27ep / ps1024 12ep / ps2048 3ep, no evals) |

**Closing framing:** do not carry "Triplet has the best LPIPS" on the resume — it is within measurement noise. The strong, defensible statements are the **~1.9–2.8× latency frontier at quality parity**, the **IR-sufficiency curve with the single-band pathology**, the **matched-compute organ FID result**, and the **int8/fp8 accuracy preservation** — each with file-level evidence behind it.
