# CISLNet quantization ablations

Modular flags live in `cislnet/quantization/config.py` (`QuantAblationConfig.from_env()`).
Copy [`quant_ablation.env.example`](quant_ablation.env.example) or export variables before `sbatch`.

## Why fp8 was not running true FP8 inference

1. **torchao static FP8 fails on Conv2d** (`'Conv2d' object has no attribute 'in_features'`).
2. Code fell back to **`fp8_e4m3fn_calibrated_baked`**: weights round-trip through e4m3fn then stored as **fp32**.
3. **`run_generator()`** called plain fp32 forward — no e4m3fn storage, no FP8 Tensor Cores.

**Fix (default now):** `QUANT_FP8_NATIVE=1` uses `cislnet/quantization/fp8_native.py`:
- Replaces Conv/Linear with `NativeFP8*` modules
- Weights stored as `torch.float8_e4m3fn`
- Forward dequantizes to compute dtype per layer (true FP8 storage path on H100/H200)
- Checkpoint `storage: native_fp8_module` — load preserves FP8 modules

Re-quantize: `QUANT_FP8_NATIVE=1 sbatch jobs/cislnet/quant/quantize_fp8.sh`

## int8/int4 PTQ ablations

| Env | Effect |
|-----|--------|
| `QUANT_USE_ACT_SCALE=0` | **Weight-only PTQ** (fixes CBAM scale inflation bug; default) |
| `QUANT_USE_ACT_SCALE=1` | Merge activation max into weight scale (only valid with runtime act quant) |
| `QUANT_CALIB_PERCENTILE=99.9` | Clip outlier activations during calibration |
| `QUANT_SKIP_CBAM=1` | Keep CBAM layers at fp32 |
| `QUANT_STORE_PARAMS_FP32=1` | BN buffers / biases in fp32 |

## QAT at 1024px

```bash
QUANT_BITS=8 QAT_EPOCHS=5 QUANT_USE_ACT_SCALE=0 sbatch jobs/cislnet/quant/train_qat.sh
```

- Entrypoint: `scripts/train_qat.py`
- Fake-quant STE on Conv/Linear during fine-tune
- Exports packed int8/int4 to `results/cislnet/quant/{run}/qat_{bits}/`
- Ablation flags: `QAT_FAKE_QUANT_WEIGHT`, `QAT_FAKE_QUANT_ACTIVATION`, `QUANT_SKIP_CBAM`, etc.

## FPGA vs AMD Vitis AI (DPUCAHX8H)

**These are different pathways.**

### Custom FPGA GEMM (your stated goal)

1. PTQ or QAT → packed int8/int4 (`net_G_int8.pt`)
2. `QUANT_FPGA_EXPORT=1` during quant → `fpga_export/fpga_manifest.json` + weight blobs
3. Golden reference: `cislnet/quantization/integer/gemm.py` (`int8_conv2d_reference`, `int8_gemm`)
4. Replace `int8_gemm` with your RTL/HLS kernel; manifest gives per-layer shapes/scales

This path gives **true integer weights** and explicit GEMM interfaces.

### AMD Vitis AI + DPUCAHX8H (Alveo U55C/U280)

- **Separate toolchain**: PyTorch → ONNX → `vai_q_tensorflow2` / PyTorch quantizer → `vai_c_xilinx` → `.xmodel` for DPU
- DPUCAHX8H runs **fixed INT8** (and limited BF16) on a **fixed operator set**
- **AttUnet blockers**: ResNet50 encoder, CBAM (GlobalPool + MLP + sigmoid), ConvTranspose2d, skip concatenations — many need CPU fallback or graph surgery (`vitis_ai/libertas` / custom ops)
- Does **not** consume our packed `.pt` format directly
- Best for subgraph deployment (e.g. quantize encoder-only or a simplified backbone), not drop-in full AttUnet

**Recommendation:** Use our packed int8 + `fpga_manifest` for custom FPGA GEMM. Evaluate Vitis AI only if you restructure the network for DPU op support or deploy a partitioned subgraph.

## Job index

| Script | Purpose |
|--------|---------|
| `quantize_fp8.sh` | PTQ fp8 native (H100) |
| `quantize_int8.sh` | PTQ int8 weight-only |
| `quantize_int4.sh` | PTQ int4 |
| `train_qat.sh` | QAT fine-tune → int8/int4 |
| `test_*.sh` | Patch eval + timing |

Example ablation sweep:

```bash
# PTQ int8 fixed (no act scale inflation)
QUANT_USE_ACT_SCALE=0 sbatch jobs/cislnet/quant/quantize_int8.sh

# PTQ int8 + skip CBAM
QUANT_SKIP_CBAM=1 sbatch jobs/cislnet/quant/quantize_int8.sh

# QAT int4
QUANT_BITS=4 QAT_EPOCHS=10 sbatch jobs/cislnet/quant/train_qat.sh
```
