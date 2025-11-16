"""
Compact raw compressor faithful to paper but storing the sensed matrix Y in a small binary file.

Usage:
  python compress_cs_paper_raw.py input.wav out.bin --rate 0.30 --quantize

Notes:
  - Header is minimal (magic, version, quant flag, seed, k,n,m,num_frames,fs,orig_len,scale)
  - Payload is int8 (if quantize) or float32 otherwise
  - Default frame shape follows paper example: k=8, n=4 (32 samples per frame)
"""
import numpy as np
from scipy.io import wavfile
import struct
import argparse
import os
import time

MAGIC = b'CSRB'   # 4 bytes magic
VERSION = 1

def frame_matrix(signal, m, k):
    """Divide señal en frames de (m×k) como especifica el paper
    Algorithm 1: cada frame contiene m×k muestras
    """
    frame_len = m * k
    num_frames = len(signal) // frame_len
    signal = signal[:num_frames * frame_len]
    frames = signal.reshape(num_frames, m, k)  # (num_frames, m, k)
    return frames, num_frames

def compress_raw(in_wav, out_bin, k=8, n=4, rate=0.5, seed=None, quantize=True):
    if seed is None:
        seed = np.random.randint(0, 2**31 - 1)
    np.random.seed(seed)

    fs, sig = wavfile.read(in_wav)
    
    print("\n" + "="*70)
    print("INFORMACIÓN DEL AUDIO DE ENTRADA")
    print("="*70)
    print(f"Archivo: {in_wav}")
    print(f"Sample rate: {fs} Hz")
    
    # Detectar si es estéreo o mono
    is_stereo = sig.ndim > 1 and sig.shape[1] == 2
    
    if is_stereo:
        print(f"Canales: Estéreo (2 canales)")
        print(f"Muestras originales: {sig.shape[0]} × 2 = {sig.shape[0] * 2:,} valores")
        # Para estéreo: tomar SOLO canal izquierdo (mejor que promedio)
        # Esto preserva la estructura original de la señal
        sig = sig[:, 0]
        print("Preprocesamiento: usando canal izquierdo para CS")
    else:
        print(f"Canales: Mono (1 canal)")
        print(f"Muestras originales: {len(sig):,} valores")
    print("="*70)
    
    # normalize to [-1,1]
    if np.issubdtype(sig.dtype, np.integer):
        sig_f = sig.astype(np.float64) / np.iinfo(sig.dtype).max
    else:
        sig_f = sig.astype(np.float64)

    original_length = len(sig_f)

    # Algorithm 1 del paper:
    # m from rate (paper: rate=0.30 -> m=3, rate=0.50 -> m=4 para k=8)
    m = int(round(k * rate))
    if m < 1 or m >= k:
        raise ValueError("Rate produce m fuera de rango. Usa rate compatible con k.")
    
    # Verificar condición del paper: k>m y k>n
    if not (k > m and k > n):
        raise ValueError(f"Paper requiere k>m y k>n. Actual: k={k}, m={m}, n={n}")

    # Dividir señal en frames de (m×k) como especifica Algorithm 1
    frames_raw, num_frames = frame_matrix(sig_f, m, k)
    if num_frames == 0:
        raise ValueError("Audio demasiado corto para m*k.")

    # ========== COMPRESIÓN MEDIANTE COMPRESSIVE SENSING (CS) ==========
    print("\n" + "="*70)
    print("COMPRESIÓN CS: Generación de Matriz de Medición y Aplicación")
    print("="*70)
    
    # Matriz de medición (k×n) como indica el paper
    print("\n[1] GENERACIÓN DE MATRIZ DE MEDICIÓN:")
    print(f"    Seed (para reproducibilidad): {seed}")
    A = np.random.randn(k, n).astype(np.float64)
    print(f"    A shape: ({k}×{n}) - Matriz de medición gaussiana")
    print(f"    A primeros elementos: {A.flat[:5]}")
    
    print(f"\n[2] PARÁMETROS DE COMPRESIÓN:")
    print(f"    Frames originales X: ({m}×{k}) cada uno")
    print(f"    Matriz de medición A: ({k}×{n})")
    print(f"    Frames comprimidos Y: ({m}×{n}) cada uno")
    print(f"    Total de frames: {num_frames}")
    print(f"    Reducción de dimensión: {m*k} → {m*n} ({100*(1-n/k):.1f}% menos columnas)")
    
    # Algorithm 1: Y = X·A donde X:(m×k), A:(k×n) -> Y:(m×n)
    print(f"\n[3] PROCESO DE COMPRESIÓN (Y = X·A):")
    start_time = time.time()
    
    Y_all = np.empty((num_frames, m, n), dtype=np.float64)
    for i in range(num_frames):
        X = frames_raw[i]    # X shape (m, k) - frame original
        Y = X @ A            # (m, k) @ (k, n) = (m, n) - comprimido
        Y_all[i] = Y
        
        if i < 3 or i == num_frames - 1:
            print(f"    Frame {i+1}: ||X||_F = {np.linalg.norm(X, 'fro'):.6f} → ||Y||_F = {np.linalg.norm(Y, 'fro'):.6f}")
    
    compression_time = time.time() - start_time
    
    print(f"\n[4] TIEMPO DE COMPRESIÓN:")
    print(f"    t = {compression_time:.6f} segundos")
    print("="*70 + "\n")

    # quantize if requested
    quant_flag = 1 if quantize else 0
    if quantize:
        max_abs = float(np.max(np.abs(Y_all)))
        if max_abs == 0:
            scale = 1.0
        else:
            scale = max_abs / 127.0   # map to int8 range
        Y_norm = Y_all / scale
        Y_q = np.round(Y_norm).astype(np.int8)  # values -128..127 (but scale ensures within -127..127)
        payload_size = Y_q.size  # bytes
    else:
        scale = 0.0
        Y_f = Y_all.astype(np.float32)
        payload_size = Y_f.size * 4

    # Write binary file
    with open(out_bin, 'wb') as f:
        # header: magic(4), version(1), quant(1), seed(4), k(2), n(2), m(2), num_frames(4), fs(4), orig_len(4), scale(4)
        header = struct.pack('<4sBBIHHHIII f',
                             MAGIC,
                             VERSION,
                             quant_flag,
                             seed,
                             k,
                             n,
                             m,
                             num_frames,
                             fs,
                             original_length,
                             scale)
        # Note: struct '<4sBBIHHHIII f' packs to size 4+1+1+4+2+2+2+4+4+4+4 = 32 bytes
        f.write(header)

        # payload
        if quantize:
            # write raw bytes of int8 in C order
            f.write(Y_q.tobytes(order='C'))
        else:
            f.write(Y_f.tobytes(order='C'))

    orig_size = os.path.getsize(in_wav)
    comp_size = os.path.getsize(out_bin)
    
    print("\n" + "="*70)
    print("RESUMEN DE COMPRESIÓN")
    print("="*70)
    print(f"Archivo guardado: {out_bin}")
    print(f"Tamaño original:  {orig_size:,} bytes")
    print(f"Tamaño comprimido: {comp_size:,} bytes (payload ~ {payload_size:,} bytes)")
    print(f"Ratio de compresión: {100*(1 - comp_size/orig_size):.2f}% reducción")
    print(f"\nParámetros: k={k}, n={n}, m={m}, frames={num_frames}")
    print(f"Seed: {seed}, Cuantización: {quantize}, Scale: {scale:.6g}")
    print("="*70)

    return out_bin

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Compress audio (paper-style) into compact raw binary.")
    p.add_argument("input", help="Input WAV")
    p.add_argument("output", help="Output binary file (.bin)")
    p.add_argument("--k", type=int, default=8, help="k (rows), default 8 as paper")
    p.add_argument("--n", type=int, default=4, help="n (cols), default 4 as paper")
    p.add_argument("--rate", type=float, default=0.5, help="compression ratio m/k (use 0.3 or 0.5 per paper)")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (optional)")
    p.add_argument("--no-quant", dest="quant", action="store_false", help="Store Y as float32 (no quantization)")
    p.set_defaults(quant=True)
    args = p.parse_args()

    compress_raw(args.input, args.output, k=args.k, n=args.n, rate=args.rate, seed=args.seed, quantize=args.quant)

