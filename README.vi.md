# Local LLM on CPU (llama.cpp)

<!-- readme-axi type=tool -->

[English](README.md) · **Tiếng Việt**

Ghi chép và script chạy sẵn để **suy luận chỉ bằng CPU** một model MoE có khả năng nhìn ảnh (vision) bằng
[llama.cpp](https://github.com/ggml-org/llama.cpp) trên một mini PC Windows nhỏ — không có GPU.
**Kết quả:** Gemma 4 26B A4B (Q8_0) + vision, context 256K, sinh **~11.5 token/s** trên i5-14400T với 64 GB DDR5.

> 🇻🇳 Dự án do một người Việt tự xây dựng và duy trì — nếu nó giúp ích cho bạn, hãy
> [ủng hộ dự án](#ủng-hộ-dự-án).

## Phần cứng

| Thành phần | Giá trị |
|---|---|
| CPU | Intel Core i5-14400T (6P + 4E) |
| RAM | 64 GB DDR5-4800, dual channel |
| Hệ điều hành | Windows 11 Pro |

## Vì sao chọn MoE trên CPU

Tốc độ sinh token trên CPU bị giới hạn bởi **băng thông bộ nhớ**: mỗi token phải đọc các trọng số đang active từ
RAM. Gemma 4 26B A4B mỗi token chỉ kích hoạt ~4B trong 26B tham số (8 trên 128 expert), nên sinh nhanh hơn nhiều
lần so với một model dense cỡ tương đương mà vẫn giữ được phần lớn chất lượng.

## Benchmark (i5-14400T, DDR5-4800)

**Quantization** (8 thread):

| Quant | Kích thước | Prompt (t/s) | Sinh (t/s) |
|---|---|---|---|
| Q4_K_M | 15.7 GB | 42.98 | 13.35 |
| Q6_K | 21.6 GB | 39.24 | 12.34 |
| **Q8_0** | **25.9 GB** | **44.02** | **11.51** |

Chọn Q8_0: chất lượng tốt nhất, chỉ chậm hơn Q4_K_M khoảng ~14 %.

**Số thread** (Q4_K_M):

| Thread | Prompt (t/s) | Sinh (t/s) |
|---|---|---|
| **8** | **42.98** | **13.35** |
| 10 | 37.49 | 12.72 |
| 12 | 37.37 | 12.19 |
| 14 | 37.90 | 11.50 |

8 thread là tối ưu — thêm E-core chỉ làm tăng tranh chấp băng thông bộ nhớ chứ không tăng tốc.

## Cài đặt

1. Tải bản llama.cpp dựng sẵn cho CPU (`llama-*-bin-win-cpu-x64.zip`) vào `llama.cpp/`.
2. Tải model và vision projector từ
   [unsloth/gemma-4-26B-A4B-it-GGUF](https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF):
   `gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf` và `mmproj-F16.gguf`.

## Cách dùng

1. Đặt `MODEL_DIR` trong [`run-server.bat`](run-server.bat), rồi double-click để chạy.
2. Mở <http://127.0.0.1:8080> (web UI có tải ảnh lên). Server cũng lắng nghe trên mạng LAN.

Các cờ chính đang dùng:

| Cờ | Mục đích |
|---|---|
| `--mmproj mmproj-F16.gguf` | Bật vision |
| `-fa on -ctk q8_0 -ctv q8_0` | Flash attention + KV cache 8-bit (~3.2 GB ở context 256K) |
| `-c 262144 -b 4096 -ub 512` | Context 256K, batch prompt lớn |
| `--threads 8 --threads-batch 10` | Cách chia tốt nhất cho 6 nhân P + 4 nhân E |
| `--temp 1.0 --top-p 0.95 --top-k 64` | Tham số sampling Google khuyến nghị cho Gemma |

Tổng RAM đang dùng: khoảng **30.6 GB** trên 64 GB.

## Ghi chú thêm

Ghi chép làm việc đầy đủ (tiếng Việt) — các model thay thế, ngân sách RAM, định dạng prompt, nhật ký các phiên:
[CLAUDE.md](CLAUDE.md).

<!-- support:start -->
## Ủng hộ dự án

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

<p align="center"><sub>🇻🇳 VietQR: quét bằng app ngân hàng bất kỳ, nội dung <code>Ung ho local-model-cpu</code>. Cảm ơn bạn đã giúp dự án tiếp tục phát triển!</sub></p>
<p align="center"><sub>Tiền mã hoá (mạng BEP20): <code>0xcbf0402b232cb24b4d998f09a4884f87b5f74750</code> · USDT · USDC · BNB · ETH · BTC</sub></p>
<!-- support:end -->

## Giấy phép

[MIT](LICENSE) © Ngô Đức Việt.
