"""
This script:
 - reads header and payload
 - regenerates A using seed
 - builds A_pinv (use np.linalg.pinv as robust Moore-Penrose)
 - reconstructs each frame: X_hat = A_pinv @ Y_frame
 - writes WAV and (optionally) prints metrics vs original
"""
import numpy as np
from scipy.io import wavfile
import struct
import argparse
import os
import time
import matplotlib.pyplot as plt

MAGIC = b'CSRB'
VERSION = 1

def read_header_and_payload(binfile):
    with open(binfile, 'rb') as f:
        header_bytes = f.read(32)  # fixed header size as in compressor
        if len(header_bytes) < 32:
            raise ValueError("Archivo demasiado corto o header corrupto.")
        magic, ver, quant_flag, seed, k, n, m, num_frames, fs, orig_len, scale = struct.unpack('<4sBBIHHHIII f', header_bytes)
        if magic != MAGIC:
            raise ValueError("Magic header no coincide. ¿Es un archivo CS raw correcto?")
        if ver != VERSION:
            raise ValueError(f"Version mismatch: found {ver}, expected {VERSION}")

        # compute expected payload length
        count_values = num_frames * m * n
        if quant_flag == 1:
            payload_bytes = count_values  # int8 -> 1 byte each
        else:
            payload_bytes = count_values * 4  # float32

        payload = f.read(payload_bytes)
        if len(payload) != payload_bytes:
            raise ValueError("Payload incomplete/corrupt.")

    header = {
        'quant': bool(quant_flag),
        'seed': int(seed),
        'k': int(k),
        'n': int(n),
        'm': int(m),
        'num_frames': int(num_frames),
        'fs': int(fs),
        'orig_len': int(orig_len),
        'scale': float(scale)
    }
    return header, payload

def decompress_raw(binfile, out_wav="reconstructed_raw.wav", original_wav=None):
    header, payload = read_header_and_payload(binfile)
    quant = header['quant']
    seed = header['seed']
    k = header['k']; n = header['n']; m = header['m']
    num_frames = header['num_frames']
    fs = header['fs']
    orig_len = header['orig_len']
    scale = header['scale']

    count_values = num_frames * m * n

    # interpret payload
    if quant:
        Y_all = np.frombuffer(payload, dtype=np.int8, count=count_values).astype(np.float64)
        # reshape to (num_frames, m, n)
        Y_all = Y_all.reshape((num_frames, m, n))
        # dequantize:
        if scale <= 0:
            raise ValueError("Scale inválida en archivo cuantizado.")
        Y_all = Y_all * scale
    else:
        Y_all = np.frombuffer(payload, dtype=np.float32, count=count_values).astype(np.float64)
        Y_all = Y_all.reshape((num_frames, m, n))

    # Algorithm 2 del paper: regenerar matriz de medición A (k×n)
    np.random.seed(seed)
    A = np.random.randn(k, n).astype(np.float64)

    # ========== MÉTODO ITERATIVO: Newton-Schultz para aproximar pseudoinversa ==========
    print("\n" + "="*70)
    print("MÉTODO ITERATIVO: Newton-Schultz para Pseudoinversa de Moore-Penrose")
    print("="*70)
    
    # Valores Iniciales (V.I.)
    # Para A:(k×n), queremos A⁺:(n×k)
    # Newton-Schultz: Yₖ₊₁ = Yₖ(2I - AYₖ) donde Y₀ = α·A^T
    print("\n[1] VALORES INICIALES (V.I.):")
    frobenius_norm_sq = np.linalg.norm(A, 'fro')**2
    Y0 = (1.0 / frobenius_norm_sq) * A.T  # Y0: (n×k)
    print(f"    Y₀ = (1/||A||²_F) * A^T")
    print(f"    ||A||²_F = {frobenius_norm_sq:.6e}")
    print(f"    Y₀ shape: {Y0.shape} (debe ser n×k)")
    print(f"    Y₀ primeros elementos: {Y0.flat[:5]}")
    
    # Parámetros del método iterativo
    tol = 1e-10
    iterMax = 100
    
    # Inicializar
    Yk = Y0.copy()  # (n×k)
    Ik = np.eye(k)  # Matriz identidad k×k (para A:(k×n), AY:(k×k))
    
    # Medir tiempo de ejecución
    start_time = time.time()
    
    # Iteración Newton-Schultz: Yₖ₊₁ = Yₖ(2I - AYₖ)
    # Y:(n×k), A:(k×n), AY:(k×n)·(n×k) = (k×k), I:(k×k)
    # Yₖ(2I - AYₖ): (n×k)·(k×k) = (n×k) ✓
    print(f"\n[2] ITERACIONES (tolerancia = {tol:.0e}, máx iteraciones = {iterMax}):")
    for iter_k in range(1, iterMax + 1):
        Yk = Yk @ (2 * Ik - A @ Yk)  # (n×k) @ (k×k) = (n×k)
        
        # Calcular error: ||A·Yₖ·A - A||_F
        error = np.linalg.norm(A @ Yk @ A - A, 'fro')
        
        if iter_k <= 5 or iter_k % 10 == 0 or error < tol:
            print(f"    Iteración {iter_k:3d}: error = {error:.6e}")
        
        if error < tol:
            break
    
    execution_time = time.time() - start_time
    A_pinv = Yk
    
    # Resultados del método iterativo
    print(f"\n[3] APROXIMACIÓN DE LA SOLUCIÓN:")
    print(f"    A⁺ (pseudoinversa) shape: {A_pinv.shape}")
    print(f"    A⁺ primeros elementos: {A_pinv.flat[:5]}")
    
    print(f"\n[4] ERROR FINAL:")
    final_error = np.linalg.norm(A @ A_pinv @ A - A, 'fro')
    print(f"    ||A·A⁺·A - A||_F = {final_error:.6e}")
    
    print(f"\n[5] NÚMERO DE ITERACIONES:")
    print(f"    k = {iter_k}")
    
    print(f"\n[6] TIEMPO DE EJECUCIÓN:")
    print(f"    t = {execution_time:.6f} segundos")
    print("="*70 + "\n")

    # ========== RECONSTRUCCIÓN DE SEÑAL ==========
    print("=" * 70)
    print("RECONSTRUCCIÓN DE SEÑAL")
    print("=" * 70)
    
    # Medir tiempo de reconstrucción
    recon_start_time = time.time()

    # Algorithm 2: Reconstruir X_hat = Y · A+ donde Y:(m×n), A+:(n×k) -> X:(m×k)
    X_all = np.empty((num_frames, m, k), dtype=np.float64)
    for i in range(num_frames):
        Yf = Y_all[i]       # (m, n) - frame comprimido
        X_all[i] = Yf @ A_pinv  # (m, n) @ (n, k) = (m, k) - reconstruido
        
        if i < 3 or i == num_frames - 1:
            print(f"Frame {i+1} reconstruido: ||X̂||_F = {np.linalg.norm(X_all[i], 'fro'):.6f}")

    # Algorithm 2: Unir todos los frames y reshape a vector 1D
    # X_all shape: (num_frames, m, k)
    recon = X_all.reshape(num_frames * m * k)[:orig_len]
    
    recon_time = time.time() - recon_start_time
    
    print(f"\nTiempo de reconstrucción: {recon_time:.6f} segundos")
    print("=" * 70 + "\n")

    # clip to [-1,1] and convert to int16
    recon = np.clip(recon, -1.0, 1.0)
    wav_int16 = (recon * 32767.0).astype(np.int16)
    wavfile.write(out_wav, fs, wav_int16)

    comp_size = os.path.getsize(binfile)
    wav_size = os.path.getsize(out_wav)
    print(f"Decompressed saved: {out_wav}")
    print(f"Compressed file: {binfile} -> {comp_size:,} bytes")
    print(f"Reconstructed WAV: {out_wav} -> {wav_size:,} bytes")
    print(f"k={k}, n={n}, m={m}, frames={num_frames}, seed={seed}, quant={quant}, scale={scale}")

    # optionally compute metrics vs original
    metrics = None
    if original_wav is not None:
        fs_o, orig = wavfile.read(original_wav)
        # Usar mismo canal que en compresión (izquierdo si es estéreo)
        if orig.ndim > 1 and orig.shape[1] == 2:
            orig = orig[:, 0]  # Canal izquierdo
            print("Original estéreo: usando canal izquierdo para comparación")
        if np.issubdtype(orig.dtype, np.integer):
            orig_f = orig.astype(np.float64) / np.iinfo(orig.dtype).max
        else:
            orig_f = orig.astype(np.float64)
        L = min(len(orig_f), len(recon))
        orig_f = orig_f[:L]; rec = recon[:L]
        R = float(np.corrcoef(orig_f, rec)[0,1])
        M = float(np.mean((orig_f - rec)**2))
        maxv = float(np.max(np.abs(orig_f))) if np.max(np.abs(orig_f))>0 else 1.0
        PSNR = 10 * np.log10((maxv**2)/M) if M>0 else float('inf')
        # simple SSIM 1D
        mu_x = orig_f.mean(); mu_y = rec.mean()
        var_x = orig_f.var(); var_y = rec.var()
        cov = float(np.mean((orig_f - mu_x)*(rec - mu_y)))
        Lc = max(max(abs(orig_f)),1e-12)
        c1 = (0.01*Lc)**2; c2 = (0.03*Lc)**2
        SSIM = ((2*mu_x*mu_y + c1) * (2*cov + c2)) / ((mu_x**2 + mu_y**2 + c1)*(var_x + var_y + c2))
        metrics = {"Pearson":R, "MSE":M, "PSNR":PSNR, "SSIM":SSIM}
        print("\nMetrics vs original:")
        print(f" Pearson: {R:.6f}")
        print(f" MSE:     {M:.6e}")
        print(f" PSNR:    {PSNR:.3f} dB")
        print(f" SSIM:    {SSIM:.6f}")
        
        # ========== GRÁFICAS COMPARATIVAS ==========
        print("\n" + "="*70)
        print("GENERANDO GRÁFICAS COMPARATIVAS")
        print("="*70)
        
        # Crear figura con 2 subplots
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # Determinar rango de muestras para visualización (primeros 50000 puntos o menos)
        max_samples = min(50000, L)
        time_axis = np.arange(max_samples) / fs
        
        # Subplot 1: Señal Original
        axes[0].plot(time_axis, orig_f[:max_samples], 'b-', linewidth=0.5, alpha=0.7)
        axes[0].set_title('Señal Original', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Tiempo (s)', fontsize=11)
        axes[0].set_ylabel('Amplitud', fontsize=11)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, time_axis[-1]])
        
        # Subplot 2: Señal Reconstruida
        axes[1].plot(time_axis, rec[:max_samples], 'r-', linewidth=0.5, alpha=0.7)
        axes[1].set_title(f'Señal Reconstruida (CS + Newton-Schultz) - Pearson={R:.4f}, PSNR={PSNR:.2f} dB', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Tiempo (s)', fontsize=11)
        axes[1].set_ylabel('Amplitud', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, time_axis[-1]])
        
        plt.tight_layout()
        
        # Guardar figura
        fig_filename = out_wav.replace('.wav', '_comparison.png')
        plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
        print(f"Gráfica guardada: {fig_filename}")
        
        # Mostrar figura
        plt.show()
        print("="*70)

    return out_wav, metrics

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Decompress the raw CS binary file and optionally compute metrics")
    p.add_argument("input", help="Compressed raw binary (.bin)")
    p.add_argument("output", help="Output WAV file")
    p.add_argument("--original", help="Original WAV to compute metrics (optional)", default=None)
    args = p.parse_args()

    decompress_raw(args.input, args.output, original_wav=args.original)
