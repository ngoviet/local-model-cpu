# Local AI Model — CPU Inference trên Lenovo Mini PC (64GB RAM)

Máy Lenovo mini PC (i5-14400T, 64GB DDR5 4800), CPU-only. Chạy local model qua llama.cpp.
Ưu tiên kiến trúc **MoE** để tăng tốc token generation trên CPU.

## Cấu hình đã xác minh (2026-04-30)

```bash
./llama-server.exe \
  -m "D:\AI\Models\gemma-4-26B-A4B-it\gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf" \
  --mmproj "D:\AI\Models\gemma-4-26B-A4B-it\mmproj-F16.gguf" \
  -fa on -ctk q8_0 -ctv q8_0 \
  -c 262144 -b 4096 -ub 512 \
  --threads 8 --threads-batch 10 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --reasoning off \
  --host 0.0.0.0 --port 8080
```

| Thông số | Giá trị |
|----------|---------|
| Model | Gemma 4 26B A4B (MoE, 128 experts, 8 active/token) |
| Quant | Q8_0 (Unsloth Dynamic, 25.94 GiB) |
| Vision | mmproj-F16.gguf (1.14 GiB, gemma4v encoder) |
| Context | 262144 (256K max) |
| KV Cache | Non-SWA 2.7 GB + SWA 0.5 GB = 3.2 GB |
| Flash Attention | Bật (`-fa on`) |
| Tốc độ Gen | **11.51 t/s** (8 threads) |
| Tốc độ Prompt | **44.02 t/s** |
| Tổng RAM | **~30.6 GB / 64 GB** |
| LAN | `http://<ip>:8080` (`--host 0.0.0.0`) |
| Script | `run-server.bat` (double-click để chạy) |

### Benchmark quantization (i5-14400T, DDR5 4800, 8 threads)

| Quant | Size | Prompt (t/s) | Gen (t/s) | RAM | Ghi chú |
|-------|------|-------------|-----------|-----|---------|
| Q4_K_M | 15.7 GB | 42.98 | 13.35 | ~20 GB | Nhanh nhất, chất lượng tốt |
| Q6_K | 21.6 GB | 39.24 | 12.34 | ~25 GB | Sweet spot |
| **Q8_0** | **25.9 GB** | **44.02** | **11.51** | **~31 GB** | **Chất lượng cao nhất** |

→ Q8_0 prompt processing nhanh nhất do dequantize đơn giản hơn Q4/Q6.

## Model khuyến nghị

### Chính: Gemma 4 26B A4B (MoE)

| Thuộc tính | Giá trị |
|------------|---------|
| Kiến trúc | MoE — 26B tổng, **~4B active** (128 experts, 8 active/token) |
| Vision | Có (cần file mmproj-F16.gguf ~1.1 GB) |
| Audio | Không (chỉ có trên E2B/E4B) |
| Context | 256K (khuyến nghị 32K cho responsive) |
| License | Apache 2.0 |
| Tốc độ CPU Q8_0 | **11.5 t/s** (i5-14400T, DDR5 4800, 8 threads) |
| HuggingFace GGUF | `unsloth/gemma-4-26B-A4B-it-GGUF` |

### Benchmark các quantization (i5-14400T, DDR5 4800, 8 threads)

| Quant | Size | Prompt (t/s) | Gen (t/s) | RAM |
|-------|------|-------------|-----------|-----|
| Q4_K_M | 15.7 GB | 42.98 | 13.35 | ~20 GB |
| Q6_K | 21.6 GB | 39.24 | 12.34 | ~25 GB |
| **Q8_0** | **25.9 GB** | **44.02** | **11.51** | **~29 GB** |

→ **Q8_0 được chọn** — chất lượng cao nhất, chỉ chậm hơn Q4 ~14%.

### Dự phòng: Qwen3-VL-30B-A3B (MoE)

| Thuộc tính | Giá trị |
|------------|---------|
| Kiến trúc | MoE — 30B tổng, **~3B active** |
| Quant Q4_K_M | **~18.6 GB** |
| Vision | Có (VL, tích hợp sẵn trong GGUF) |
| Thinking mode | Có |
| Context | 128K |
| License | Apache 2.0 |
| Tốc độ CPU benchmark | 12–33 t/s (DDR4/DDR5) |
| HuggingFace GGUF | `bartowski/Qwen_Qwen3-VL-30B-A3B-Instruct-GGUF` |
| Ollama | `qwen3-vl:30b-a3b-instruct-q4_K_M` |

**Tải về:**
```bash
huggingface-cli download bartowski/Qwen_Qwen3-VL-30B-A3B-Instruct-GGUF \
  --include "Qwen_Qwen3-VL-30B-A3B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models
```

## Tại sao MoE thắng trên CPU

Token generation bị giới hạn bởi **băng thông RAM** (memory bandwidth). Giải thích đơn giản: mỗi token được sinh ra cần đọc toàn bộ weights của model từ RAM. Với MoE:

- **Chỉ ~3–4B params active** mỗi token → đọc ~3–4GB weights (ở Q4)
- Dense model (vd: Gemma 4 31B) → đọc toàn bộ ~17–20GB mỗi token
- Kết quả: MoE **nhanh hơn 5–6 lần** so với dense model cùng kích thước

## Benchmark CPU thực tế

### Gemma 4 26B A4B Q4_K_M trên i5-14400T + DDR5 4800

| Threads | Prompt (t/s) | Generation (t/s) |
|---------|-------------|-------------------|
| 6 | — | — |
| 8 | 42.98 | **13.35** |
| 10 | 37.49 | 12.72 |
| 12 | 37.37 | 12.19 |
| 14 | 37.90 | 11.50 |

→ **8 threads là tối ưu** cho i5-14400T (6P + 4E). E-cores không giúp ích cho memory-bandwidth-bound work, thậm chí gây cạnh tranh băng thông.

### Qwen3-30B-A3B Q4 (tham khảo từ cộng đồng)

| Cấu hình RAM | Tốc độ (token/s) |
|--------------|-------------------|
| DDR5 6400 dual-channel (AMD 9950X) | ~33 t/s |
| DDR5 5600 dual-channel | 18–20 t/s |
| DDR4 3600 dual-channel (i7-1185G7) | 10–14 t/s |

Gemma 4 26B A4B (~4B active) sẽ chậm hơn một chút so với Qwen (~3B active) do active params lớn hơn, nhưng chất lượng sinh cao hơn.

## Cài đặt

### Option A: Pre-built binary (không cần build)

Tải bản binary mới nhất từ [GitHub Releases](https://github.com/ggml-org/llama.cpp/releases), chọn `llama-b*-bin-win-cpu-x64.zip`.

```bash
# Tải và giải nén
curl -L -o llama-cpu-win.zip "https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-b8984-bin-win-cpu-x64.zip"
unzip llama-cpu-win.zip
```

### Option B: Build từ source (Windows, cần cmake + MSVC)

```powershell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
cmake -B build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF
cmake --build build --config Release
```

### Option C: Ollama (đơn giản nhất)

1. Cài Ollama cho Windows: `winget install Ollama.Ollama`
2. Chạy model:
```bash
ollama run gemma4:26b-a4b-it-q4_K_M
```
3. Test vision:
```bash
ollama run gemma4:26b-a4b-it-q4_K_M
>>> /image "C:\path\to\image.jpg" Mô tả bức ảnh này
```

## Chạy với llama.cpp

### llama-server (web UI + vision) — **đã xác minh hoạt động**

```bash
./llama-server.exe \
  -m "D:/AI/Models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
  --mmproj "D:/AI/Models/gemma-4-26B-A4B-it/mmproj-F16.gguf" \
  -fa on \
  -ctk q8_0 -ctv q8_0 \
  -c 32768 \
  --threads 8 \
  --threads-batch 10 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  --reasoning on \
  --host 127.0.0.1 --port 8080
```

Mở http://127.0.0.1:8080 — web UI hỗ trợ upload ảnh trực tiếp.

### llama-cli (terminal, text-only)

```bash
./llama-cli.exe \
  -m "D:/AI/Models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
  -fa on -ctk q8_0 -ctv q8_0 \
  -c 32768 --threads 8 \
  --temp 1.0 --top-p 0.95 --top-k 64 \
  -n 512 -p "Xin chào"
```

### llama-mtmd-cli (terminal, có vision)

```bash
./llama-mtmd-cli.exe \
  -m "D:/AI/Models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
  --mmproj "D:/AI/Models/gemma-4-26B-A4B-it/mmproj-F16.gguf" \
  -fa on -ctk q8_0 -ctv q8_0 \
  -c 32768 --threads 8 \
  --temp 1.0 --top-p 0.95 --top-k 64
```

## Tối ưu CPU inference

### Flags quan trọng của llama.cpp

| Flag | Ý nghĩa |
|------|---------|
| `--mmproj <file>` | **Bắt buộc** để có vision (file mmproj-F16.gguf) |
| `-fa on` | Flash Attention — giảm RAM usage cho context lớn |
| `-ctk q8_0 -ctv q8_0` | Quantize KV cache xuống 8-bit — tiết kiệm ~50% RAM cho cache |
| `-c 32768` | Context length (có thể tăng lên nếu cần, theo dõi RAM) |
| `--threads N` | Số thread cho generation (i5-14400T: **8 là tối ưu**) |
| `--threads-batch N` | Thread cho prompt processing (có thể cao hơn `--threads`) |
| `--temp 1.0 --top-p 0.95 --top-k 64` | Tham số sampling khuyến nghị bởi Google |
| `--reasoning on` | Bật/tắt thinking mode của Gemma 4 |
| `--no-mmap` | Pre-allocate toàn bộ weights vào RAM (ổn định hơn) |
| `--visual-token-budget N` | Chất lượng vision: 280 (chat), 560 (UI), 1120 (OCR) |
| `-rtr` | Re-tokenize prompt sau mỗi request (cho server) |

### Cách tìm số thread tối ưu

```bash
./llama-bench.exe \
  -m "D:/AI/Models/gemma-4-26B-A4B-it/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf" \
  -n 128 -p 512 \
  --threads <N>
```

- Bắt đầu với số lõi P (6 cho i5-14400T), thử tới 16
- **Quan trọng**: nhiều thread hơn không luôn nhanh hơn — E-cores có thể giảm hiệu năng
- DDR5 bandwidth có hạn, quá nhiều thread gây cạnh tranh

### RAM budget (64GB tổng) — thực tế đã đo

```
Model weights (Q4_K_M):   15.7 GB
mmproj (vision encoder):   1.1 GB
KV Cache (32K ctx q8_0):  ~0.8 GB
OS + overhead:            ~2.0 GB
Tổng đã dùng:            ~19.6 GB
Còn dư:                   ~44.4 GB
```

→ Có thể tăng context lên 128K+ hoặc chạy song song nhiều model nếu muốn.

## Prompt format

### Gemma 4 (chat template từ model)
```
<|turn>system
<|think|>
You are a helpful assistant<turn|>
<|turn>user
Hello<turn|>
<|turn>model
Hi there<turn|>
```

- `<|think|>` trong system prompt bật thinking mode
- EOS token: `<turn|>` (token 106)
- Vision: đặt ảnh trước text để kết quả tốt nhất
- Multi-turn: chỉ giữ lại phần visible answer, bỏ thought blocks

### Qwen3-VL (ChatML)
```
<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{message}<|im_end|>
<|im_start|>assistant
```

## Nguồn GGUF chính thức

- **Gemma 4 26B A4B**: https://huggingface.co/bartowski/google_gemma-4-26B-A4B-it-GGUF
- **Qwen3-VL-30B-A3B Instruct**: https://huggingface.co/bartowski/Qwen_Qwen3-VL-30B-A3B-Instruct-GGUF
- **Qwen3-VL-30B-A3B Thinking**: https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Thinking-GGUF
- **Unsloth Gemma 4**: https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF
- **Unsloth Qwen3-VL**: https://huggingface.co/unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF

## Các model tham khảo khác

| Model | Loại | Size Q4 | Vision | Context | Ghi chú |
|-------|------|---------|--------|---------|---------|
| Qwen3 Omni 30B-A3B | MoE | ~19 GB | Có + Audio | 128K | Mixed modality, thay thế tốt nếu cần audio |
| Llama 4 Scout 17B-16E | MoE | ~10 GB | Có | 256K | Nhẹ hơn, nhanh hơn nhưng chất lượng thấp hơn |
| Gemma 4 31B | Dense | ~20 GB | Có | 256K | Chất lượng cao nhất, chậm hơn ~5x |
| Phi-4 Multimodal 14B | Dense | ~9 GB | Có | 128K | Nhẹ, nhanh, chất lượng khá |

## Nghiên cứu: DeepSeek V4 Flash vs Gemma 4 — Coding & Agent

### Vì sao DeepSeek V4 Flash thông minh hơn Gemma 4?

3 yếu tố cốt lõi:

1. **Active params gấp 3.25x**: Flash 13B vs Gemma 4B active/token — giữ được context dài, instruction phức tạp, suy luận đa bước
2. **Tổng kiến thức gấp 11x**: 284B vs 26B params — nhiều expert chuyên sâu hơn
3. **Thiết kế cho Agent**: Engram memory xuyên session, CSA+HCA attention tối ưu 1M context, huấn luyện tool calling

### DeepSeek V4 Flash vs Pro (số liệu chính thức)

| Thông số | V4 Flash | V4 Pro |
|----------|----------|--------|
| Tổng params | 284B MoE | 1,600B MoE |
| Active/token | 13B | 49B |
| Context | 1M | 1M |
| Giá input (API) | $0.14/M | $0.53/M |

#### Coding (Think-Max mode)

| Benchmark | V4 Flash | V4 Pro | Gap |
|-----------|----------|--------|-----|
| SWE-bench Verified | 79.0% | 80.6% | +1.6 |
| SWE-bench Pro | 52.6% | 55.4% | +2.8 |
| LiveCodeBench | 91.6 | 93.5 | +1.9 |
| Codeforces | 3052 | 3206 | +154 |

→ Code: Pro hơn Flash ~2-3 điểm — gap nhỏ.

#### Agent (Think-Max mode)

| Benchmark | V4 Flash | V4 Pro | Gap |
|-----------|----------|--------|-----|
| Terminal Bench 2.0 | 56.9% | 67.9% | **+11.0** |
| BrowseComp | 73.2% | 83.4% | **+10.2** |
| MCPAtlas | 69.0% | 73.6% | +4.6 |
| Toolathlon | 47.8% | 51.8% | +4.0 |

→ Agent: Pro vượt trội, gap 4-11 điểm.

#### Điểm thú vị: Non-Think mode Flash ≈ Pro

| Benchmark | Flash Non-Think | Pro Non-Think |
|-----------|----------------|---------------|
| SWE Verified | **73.7** | 73.6 |
| MMLU-Pro | **83.0** | 82.9 |

→ Flash 13B active tối ưu đến mức ngang Pro 49B ở chế độ không suy luận.

### So sánh với mô hình khác (SWE-bench Verified)

| Model | Score | Ghi chú |
|-------|-------|---------|
| DeepSeek V4 Pro-Max | 80.6% | Cần 16× H100 |
| DeepSeek V4 Flash-Max | 79.0% | 284B, không chạy được trên 64GB |
| Qwen3.6 35B A3B | 73.4% | 35B/3B active, chạy được 64GB, agent MCPMark 37% |
| Gemma 4 31B | 52.0% | Dense, chậm 5x |
| Gemma 4 26B A4B | ~48% | Đang chạy trên máy, agent MCPMark 18% |

### Có chạy local được DeepSeek V4 Flash không?

- **Không** — 284B params, Q2 quantization cũng cần ~70GB
- llama.cpp: WIP, chưa merge (`nisparks/wip/deepseek-v4-support`)
- Tất cả benchmark đều chạy GPU (RTX Pro 6000 96GB+)
- Nếu chạy được CPU: 13B active → ~3-4 t/s với DDR5 4800

### Chi phí API vs Local

| | API DeepSeek V4 | Local ($9K RTX Pro 6000) |
|---|---|---|
| Chi phí/năm | ~$1,332 ($111/tháng) | $9K upfront |
| Break-even | — | ~6-7 năm |
| Lợi ích local | — | Privacy, không rate limit, không phụ thuộc internet |

→ Ở mức dùng hiện tại, API vẫn rẻ hơn. Chỉ đầu tư GPU khi dùng 24/7 hoặc cần privacy tuyệt đối.

### Kết luận

- **Gemma 4 Q8 local** cho task đơn giản, chat, vision
- **DeepSeek V4 Flash API** ($0.14/M) cho code phức tạp và agent
- **DeepSeek V4 Pro API** cho agent multi-step dài, browse, task khó nhất
- **Qwen3.6 35B A3B** nếu muốn local agent tốt hơn (nhưng bị chèn tiếng Trung)

## Quick start checklist

- [x] Tải llama.cpp pre-built binary: `llama-b8984-bin-win-cpu-x64.zip`
- [x] Tải model + mmproj từ Unsloth: `unsloth/gemma-4-26B-A4B-it-GGUF`
- [x] Benchmark tối ưu thread: `./llama-bench.exe ... --threads 8` → **13.35 t/s**
- [x] Chạy server: `./llama-server.exe ... --mmproj ... --threads 8 --reasoning on`
- [x] Xác nhận vision encoder load: `loaded multimodal model`
- [ ] Mở http://127.0.0.1:8080 — chat text
- [ ] Upload ảnh — test vision
- [ ] Kiểm tra RAM usage trong Task Manager (~19.6 GB khi chạy)

## Session Log

### 2026-04-30 — Khởi tạo & tối ưu

1. **Nghiên cứu model**: Chọn Gemma 4 26B A4B (MoE, 128 experts, 8 active) làm primary, Qwen3-VL-30B-A3B làm dự phòng
2. **Tải llama.cpp**: Pre-built binary b8984 (không có cmake/compiler trên máy)
3. **Tải model**: Unsloth Dynamic GGUF + mmproj-F16.gguf (user tự tải, path `D:\AI\Models\gemma-4-26B-A4B-it\`)
4. **Test load ban đầu**: Thành công với Q4_K_M, speed 13.4 t/s (8 threads). PC treo khi dùng 10 threads → giảm còn 8
5. **Benchmark thread**: Test 8/10/12/14 threads — **8 threads tối ưu** cho i5-14400T (E-cores gây chậm)
6. **Benchmark quantization**: Q4_K_M (13.35 t/s), Q6_K (12.34 t/s), Q8_0 (11.51 t/s). Chọn **Q8_0** — chất lượng cao nhất, chỉ chậm hơn Q4 ~14%
7. **Vision**: mmproj-F16.gguf load thành công (`loaded multimodal model`, gemma4v encoder)
8. **Tăng context**: Từ 32K lên **256K (262144)** — KV cache tăng từ 0.8GB lên 3.2GB, tổng RAM ~30.6GB
9. **Thêm `-b 4096`**: Tăng batch size cho prompt processing — xử lý input dài nhanh hơn
10. **LAN access**: `--host 0.0.0.0` — IP LAN `192.168.10.224:8080`
11. **Tắt thinking**: `--reasoning off` (user preference)
12. **Script**: `run-server.bat` — double-click chạy server với toàn bộ cấu hình tối ưu
13. **GitHub**: Push lên `ngoviet/local-model-cpu` (public)

### Thông số máy

| Thành phần | Giá trị |
|------------|---------|
| CPU | Intel Core i5-14400T (6P + 4E, 16 threads) |
| RAM | 64GB DDR5 4800 dual-channel |
| Storage | SSD |
| OS | Windows 11 Pro |
| IP LAN | 192.168.10.224 |

### File quan trọng

| File | Mục đích |
|------|----------|
| `run-server.bat` | Script chạy server, double-click |
| `llama.cpp/` | llama.cpp b8984 pre-built binaries |
| `D:\AI\Models\gemma-4-26B-A4B-it\` | Model GGUF + mmproj |
| `.gitignore` | Exclude binaries, models khỏi git |

### Việc cần làm tiếp

- [ ] Test vision: upload ảnh qua web UI `http://127.0.0.1:8080`
- [ ] Test LAN: truy cập từ máy khác trong mạng
- [ ] Đo RAM thực tế khi chat dài / nhiều ảnh
- [ ] Thử Qwen3-VL-30B-A3B làm model dự phòng
- [ ] Tinh chỉnh `--visual-token-budget` (280/560/1120) nếu cần OCR

### 2026-04-30 (part 2) — Nghiên cứu DeepSeek V4 & so sánh

1. **So sánh DeepSeek V4 Flash vs Pro**: Flash 284B/13B active, Pro 1.6T/49B active
2. **Coding gap Flash→Pro**: Chỉ 2-3 điểm (SWE-bench 79% vs 80.6%) — gap nhỏ
3. **Agent gap Flash→Pro**: 4-11 điểm — Pro vượt trội cho agent
4. **Phát hiện**: Flash Non-Think ngang Pro Non-Think (SWE 73.7 vs 73.6)
5. **Không chạy được local**: DeepSeek V4 Flash 284B — cần ít nhất 70GB VRAM, llama.cpp chưa hỗ trợ chính thức
6. **Gemma 4 vs DeepSeek V4**: 3 lý do chính — active params (4B vs 13B), tổng kiến thức (26B vs 284B), thiết kế agent
7. **Qwen3.6 35B A3B**: SWE-bench 73.4%, chạy được 64GB, agent MCPMark 37% — nhưng bị chèn tiếng Trung
8. **API cost analysis**: $14.81/4 ngày → ~$111/tháng → ~$1,332/năm. Local $9K GPU = break-even ~6-7 năm
9. **Chiến lược**: Gemma 4 Q8 local cho task đơn giản + DeepSeek V4 Flash API ($0.14/M) cho code/agent phức tạp
10. **Cập nhật CLAUDE.md**: Thêm toàn bộ bảng benchmark, so sánh, phân tích chi phí
