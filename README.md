# Local LLM on CPU (llama.cpp)

<!-- readme-axi type=tool -->

**English** · [Tiếng Việt](README.vi.md)

Notes and a ready-to-run script for **CPU-only inference** of a vision-capable MoE model with
[llama.cpp](https://github.com/ggml-org/llama.cpp) on a small Windows mini PC — no GPU.
**Result:** Gemma 4 26B A4B (Q8_0) + vision, 256K context, **~11.5 tokens/s** generation on an i5-14400T with 64 GB DDR5.

> 🇻🇳 Built and maintained by one person in Việt Nam — if it saves you time, please
> [support the project](#support-the-project).

## Hardware

| Component | Value |
|---|---|
| CPU | Intel Core i5-14400T (6P + 4E) |
| RAM | 64 GB DDR5-4800, dual channel |
| OS | Windows 11 Pro |

## Why MoE on CPU

Token generation on CPU is limited by **memory bandwidth**: every token reads the active weights from RAM.
Gemma 4 26B A4B activates only ~4B of its 26B parameters per token (8 of 128 experts), so it generates
several times faster than a dense model of similar size while keeping most of the quality.

## Benchmarks (i5-14400T, DDR5-4800)

**Quantization** (8 threads):

| Quant | Size | Prompt (t/s) | Generation (t/s) |
|---|---|---|---|
| Q4_K_M | 15.7 GB | 42.98 | 13.35 |
| Q6_K | 21.6 GB | 39.24 | 12.34 |
| **Q8_0** | **25.9 GB** | **44.02** | **11.51** |

Q8_0 was chosen: best quality, only ~14 % slower than Q4_K_M.

**Threads** (Q4_K_M):

| Threads | Prompt (t/s) | Generation (t/s) |
|---|---|---|
| **8** | **42.98** | **13.35** |
| 10 | 37.49 | 12.72 |
| 12 | 37.37 | 12.19 |
| 14 | 37.90 | 11.50 |

8 threads is optimal — E-cores add memory-bandwidth contention rather than speed.

## Installation

1. Download a pre-built llama.cpp CPU release (`llama-*-bin-win-cpu-x64.zip`) into `llama.cpp/`.
2. Download the model and vision projector from
   [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF):
   `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` and `mmproj-F16.gguf`.

## Usage

1. Set `MODEL_DIR` in [`run-server.bat`](run-server.bat), then double-click it.
2. Open <http://127.0.0.1:8080> (web UI with image upload). The server also listens on the LAN.

Key flags used:

| Flag | Purpose |
|---|---|
| `--mmproj mmproj-F16.gguf` | Enable vision |
| `-fa on -ctk q8_0 -ctv q8_0` | Flash attention + 8-bit KV cache (~3.2 GB at 256K context) |
| `-c 262144 -b 4096 -ub 512` | 256K context, large prompt batches |
| `--threads 8 --threads-batch 10` | Best split for 6P + 4E cores |
| `--temp 1.0 --top-p 0.95 --top-k 64` | Google's recommended sampling for Gemma |

Total RAM in use: about **30.6 GB** of 64 GB.

## More notes

Full working notes (in Vietnamese) — alternative models, RAM budget, prompt formats, session log: [CLAUDE.md](CLAUDE.md).

<!-- support:start -->
## Support the project

<p align="center">
  <img src="https://raw.githubusercontent.com/ngoviet/.github/main/assets/support/banner.svg" width="720" alt="Built and maintained by one person in Việt Nam">
</p>

<p align="center">
  <a href="https://buymeacoffee.com/ngoviet"><img src="https://raw.githubusercontent.com/ngoviet/.github/main/assets/support/btn-coffee.svg" height="56" alt="Buy me a coffee"></a>
  &nbsp;
  <a href="https://github.com/sponsors/ngoviet"><img src="https://raw.githubusercontent.com/ngoviet/.github/main/assets/support/btn-sponsors.svg" height="56" alt="GitHub Sponsors"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/ngoviet/.github/main/assets/support/vietqr-card.svg" width="300" alt="VietQR — MB Bank — NGO DUC VIET — 911051212">
</p>

<p align="center"><sub>🇻🇳 VietQR: scan with any Vietnamese banking app, transfer note <code>Ung ho local-model-cpu</code>. Thank you for keeping this project alive!</sub></p>
<p align="center"><sub>Crypto (BEP20 network): <code>0xcbf0402b232cb24b4d998f09a4884f87b5f74750</code> · USDT · USDC · BNB · ETH · BTC</sub></p>
<!-- support:end -->

## License

[MIT](LICENSE) © Ngô Đức Việt.
