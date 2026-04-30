@echo off
setlocal enabledelayedexpansion

:: ============================================================
::  Gemma 4 26B A4B MoE + Vision — llama.cpp Server
::  CPU: i5-14400T | RAM: 64GB DDR5 4800 | Quant: Q8_0 (25.9GB)
::  Context: 256K max | KV Cache: ~3.1 GB
::  Tốc độ: ~11.5 t/s gen | ~44 t/s prompt
:: ============================================================

set LLAMA_DIR=%~dp0llama.cpp
set MODEL_DIR=D:\AI\Models\gemma-4-26B-A4B-it

set MODEL=%MODEL_DIR%\gemma-4-26B-A4B-it-UD-Q8_K_XL.gguf
set MMPROJ=%MODEL_DIR%\mmproj-F16.gguf

:: Lấy LAN IP
for /f "tokens=2 delims=:" %%a in ('ipconfig ^| findstr /i "192.168."') do (
    for /f "tokens=1 delims= " %%b in ("%%a") do set LAN_IP=%%b
)

echo.
echo ========================================
echo  Gemma 4 26B A4B Q8_0 + Vision Server
echo ========================================
echo.
echo  Model:   %MODEL%
echo  Vision:  %MMPROJ%
echo  Context: 262144 (256K max)
echo  Threads: 8 / batch 10
echo  Speed:   ~11.5 t/s generation
echo.
echo  Local:   http://127.0.0.1:8080
if defined LAN_IP (
    echo  LAN:     http://%LAN_IP%:8080
) else (
    echo  LAN:     http://<your-ip>:8080
)
echo.
echo  RAM dùng: ~30.6 GB / 64 GB
echo ========================================
echo.

if not exist "%MODEL%" (
    echo [ERROR] Model file not found: %MODEL%
    pause
    exit /b 1
)

if not exist "%MMPROJ%" (
    echo [ERROR] mmproj file not found: %MMPROJ%
    pause
    exit /b 1
)

echo Starting server... Press Ctrl+C to stop.
echo.

"%LLAMA_DIR%\llama-server.exe" ^
  -m "%MODEL%" ^
  --mmproj "%MMPROJ%" ^
  -fa on ^
  -ctk q8_0 -ctv q8_0 ^
  -c 262144 ^
  -b 4096 -ub 512 ^
  --threads 8 ^
  --threads-batch 10 ^
  --temp 1.0 --top-p 0.95 --top-k 64 ^
  --reasoning off ^
  --host 0.0.0.0 --port 8080

pause
