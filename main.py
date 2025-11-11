"""
Compresión y reconstrucción de audio por Compressive Sensing + pseudoinversa.
Implementa:
 - compresión: y = Phi @ x
 - reconstrucción por pseudo-inversa (numpy.linalg.pinv)
 - reconstrucción alternativa: A^+ = (A^T A)^{-1} A^T donde (A^T A)^{-1} se calcula
   con Newton-Schulz iterativo (cuando A^T A es invertible y bien condicionada).
 - métricas: MSE, PSNR, Pearson R, SSIM (1D approx)
 - ploteo de señales originales vs reconstruidas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
from scipy.stats import pearsonr
import math
import argparse
import os

# ---------- Utilidades métricas ----------
def mse(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return np.mean((x - y) ** 2)

def psnr(x, y, data_range=None):
    m = mse(x, y)
    if m == 0:
        return float('inf')
    if data_range is None:
        # infer range from signal type
        data_range = np.max(np.abs(x))
        if data_range == 0:
            data_range = 1.0
    return 10 * np.log10((data_range ** 2) / m)

def pearson_corr(x, y):
    x = np.ravel(x)
    y = np.ravel(y)
    if x.size < 2:
        return 1.0
    r, _ = pearsonr(x, y)
    return r

def ssim_1d(x, y, K1=0.01, K2=0.03, L=None):
    # Implementación simplificada del SSIM para señales 1D (ventana global)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if L is None:
        L = np.max(np.abs(x)) - np.min(np.abs(x))
        if L == 0:
            L = 1.0
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    mu_x = x.mean()
    mu_y = y.mean()
    sigma_x2 = ((x - mu_x) ** 2).mean()
    sigma_y2 = ((y - mu_y) ** 2).mean()
    cov = ((x - mu_x) * (y - mu_y)).mean()
    num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
    den = (mu_x**2 + mu_y**2 + c1) * (sigma_x2 + sigma_y2 + c2)
    if den == 0:
        return 1.0 if num == 0 else 0.0
    return num / den

# ---------- Newton-Schulz para inversa (cuadrada) ----------
def newton_schulz_inverse(A, tol=1e-8, max_iter=150, verbose=False):
    """
    Calcula A^{-1} aproximado por Newton-Schulz.
    Necesita que A sea cuadrada y razonablemente condicionada.
    Inicialización: Y0 = A^T / ||A||_F^2
    Iteración: Y_{k+1} = Y_k (2 I - A Y_k)
    """
    A = np.asarray(A, dtype=float)
    assert A.shape[0] == A.shape[1], "Newton-Schulz requiere matriz cuadrada."
    n = A.shape[0]
    normF2 = np.linalg.norm(A, 'fro')**2
    if normF2 == 0:
        raise ValueError("Matriz nula.")
    Y = A.T / normF2
    I = np.eye(n)
    for k in range(max_iter):
        AY = A @ Y
        err = np.linalg.norm(AY - I, ord='fro')
        if verbose and k % 20 == 0:
            print(f"  Newton-Schulz iter {k}: error={err:.2e}")
        if err < tol:
            if verbose:
                print(f"  Converged in {k} iterations")
            break
        Y = Y @ (2 * I - AY)
    return Y

# ---------- Pseudoinversa usando Newton-Schulz sobre (A^T A) ----------
_pseudoinv_cache = {}

def pseudoinv_via_newton(A, tol=1e-8, max_iter=250, cache_key=None):
    """
    Aproxima la pseudo-inversa A^+ usando Newton-Schulz.
    - Si m >= n (más filas que columnas): A^+ = (A^T A)^{-1} A^T
    - Si m < n (más columnas que filas): A^+ = A^T (A A^T)^{-1}
    Calcula la inversa necesaria con Newton-Schulz.
    Usa caché para evitar recalcular la misma pseudoinversa.
    """
    # Si tenemos caché, devolverlo
    if cache_key is not None and cache_key in _pseudoinv_cache:
        return _pseudoinv_cache[cache_key]
    
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    
    if m >= n:
        # Caso columnas independientes: usar (A^T A)^{-1} A^T
        ATA = A.T @ A
        if np.linalg.matrix_rank(ATA) < ATA.shape[0]:
            raise np.linalg.LinAlgError("A^T A singular - Newton pseudoinverse no aplicable.")
        inv_ATA = newton_schulz_inverse(ATA, tol=tol, max_iter=max_iter)
        Ap = inv_ATA @ A.T
    else:
        # Caso filas independientes (m < n): usar A^T (A A^T)^{-1}
        AAT = A @ A.T
        if np.linalg.matrix_rank(AAT) < AAT.shape[0]:
            raise np.linalg.LinAlgError("A A^T singular - Newton pseudoinverse no aplicable.")
        inv_AAT = newton_schulz_inverse(AAT, tol=tol, max_iter=max_iter)
        Ap = A.T @ inv_AAT
    
    # Guardar en caché
    if cache_key is not None:
        _pseudoinv_cache[cache_key] = Ap
    
    return Ap

# ---------- Proceso de compresión y reconstrucción ----------
def compress_frame(Phi, x):
    # x debe ser vector columna (n,)
    return Phi @ x

def reconstruct_via_pinv(Phi, y):
    # Método robusto: Moore-Penrose via SVD
    Ap = np.linalg.pinv(Phi)
    return Ap @ y

def reconstruct_via_newton(Phi, y, tol=1e-8, max_iter=250):
    # Usar caché ya que Phi es la misma para todos los frames
    Ap = pseudoinv_via_newton(Phi, tol=tol, max_iter=max_iter, cache_key='Phi')
    return Ap @ y

# ---------- Pipeline completo ----------
def cs_compress_reconstruct(signal_vec, Phi, frame_len, method='pinv', overlap=0, verbose=False, use_window=False):
    """
    - signal_vec: 1D numpy array
    - Phi: measurement matrix of size m x n (n = frame_len)
    - frame_len: n
    - method: 'pinv' or 'newton'
    - overlap: cantidad de muestras que se solapan entre frames (por ejemplo 0 o n//2)
    - verbose: mostrar progreso
    - use_window: aplicar ventana de Hann para suavizar transiciones
    """
    n = frame_len
    assert Phi.shape[1] == n, "Phi debe tener n columnas."
    step = n - overlap
    idx = 0
    # zero-pad al final si hace falta
    pad = (-(len(signal_vec) - n) % step) if len(signal_vec) < n or (len(signal_vec)-n)%step != 0 else 0
    if pad > 0:
        signal_vec = np.concatenate([signal_vec, np.zeros(pad)])
    recon = np.zeros_like(signal_vec, dtype=float)
    weight = np.zeros_like(signal_vec, dtype=float)  # para overlap-add
    
    # Crear ventana de Hann si se solicita
    window = np.hanning(n) if use_window else np.ones(n)
    
    # Pre-calcular pseudoinversa para newton (solo una vez)
    if method == 'newton' and verbose:
        print("Calculando pseudoinversa (solo una vez)...")
        pseudoinv_via_newton(Phi, cache_key='Phi')
    
    total_frames = (len(signal_vec) - n) // step + 1
    processed = 0
    
    while idx + n <= len(signal_vec):
        x = signal_vec[idx:idx+n].astype(float)
        y = compress_frame(Phi, x)
        if method == 'pinv':
            xhat = reconstruct_via_pinv(Phi, y)
        elif method == 'newton':
            xhat = reconstruct_via_newton(Phi, y)
        else:
            raise ValueError("method debe ser 'pinv' o 'newton'")
        recon[idx:idx+n] += xhat * window
        weight[idx:idx+n] += window
        idx += step
        processed += 1
        
        if verbose and processed % 1000 == 0:
            print(f"Procesados {processed}/{total_frames} frames ({100*processed/total_frames:.1f}%)")
    
    # normalizar solapamientos
    nonzero = weight > 0
    recon[nonzero] /= weight[nonzero]
    return recon[:len(signal_vec)-pad]  # recortar padding

# ---------- Ejemplo de uso con un .wav ----------
def run_example(wav_path, frame_len=512, m_measure=256, method='pinv', overlap=0, plot=True, max_duration=None):
    sr, data = wavfile.read(wav_path)
    
    # normalizar PRIMERO a float antes de hacer mean para evitar overflow
    if np.issubdtype(data.dtype, np.integer):
        maxv = np.iinfo(data.dtype).max
        data = data.astype(float) / maxv
    else:
        data = data.astype(float)
    
    # maneja estéreo: promedio a mono DESPUÉS de normalizar
    if data.ndim > 1:
        data = data.mean(axis=1)
    
    # Limitar duración si se especifica (en segundos)
    if max_duration is not None and max_duration > 0:
        max_samples = int(sr * max_duration)
        if len(data) > max_samples:
            print(f"Limitando audio de {len(data)/sr:.1f}s a {max_duration}s ({max_samples} muestras)")
            data = data[:max_samples]
    # construir matriz de medición Gaussiana (seed para reproducibilidad)
    np.random.seed(42)
    Phi = np.random.normal(0, 1.0, size=(m_measure, frame_len))
    # Normalizar filas de Phi (importante para estabilidad numérica)
    Phi = Phi / np.linalg.norm(Phi, axis=1, keepdims=True)
    
    # Calcular tasa de compresión
    compression_ratio = m_measure / frame_len
    compression_percent = compression_ratio * 100
    
    print(f"SR={sr}, samples={len(data)}, frame={frame_len}, m={m_measure}, método={method}")
    print(f"Rango de datos: [{data.min():.3f}, {data.max():.3f}]")
    print(f"Tasa de compresión: {compression_percent:.1f}% (m/n = {m_measure}/{frame_len})")
    recon = cs_compress_reconstruct(data, Phi, frame_len, method=method, overlap=overlap, verbose=True, use_window=False)
    # métricas
    mse_val = mse(data, recon)
    psnr_val = psnr(data, recon, data_range=1.0)
    pearson_val = pearson_corr(data, recon)
    ssim_val = ssim_1d(data, recon, L=2.0)  # L ~ rango de la señal (-1 .. 1) => L=2
    print(f"MSE={mse_val:.6e}, PSNR={psnr_val:.2f} dB, Pearson R={pearson_val:.6f}, SSIM={ssim_val:.6f}")
    if plot:
        t = np.arange(len(data))/sr
        # dibujar un segmento (ej. primeros 2 segundos o totalidad si corta)
        seg_samples = min(len(data), sr*3)
        plt.figure(figsize=(10,6))
        plt.subplot(211)
        plt.title("Original (segmento)")
        plt.plot(t[:seg_samples], data[:seg_samples])
        plt.subplot(212)
        plt.title("Reconstruido (segmento)")
        plt.plot(t[:seg_samples], recon[:seg_samples])
        plt.xlabel("Tiempo [s]")
        plt.tight_layout()
        plt.show()
    return {
        'sr': sr,
        'orig': data,
        'recon': recon,
        'mse': mse_val,
        'psnr': psnr_val,
        'pearson': pearson_val,
        'ssim': ssim_val,
        'Phi': Phi
    }

# ---------- CLI simple ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CS audio example")
    parser.add_argument("--wav", type=str, required=True, help="Ruta al archivo .wav")
    parser.add_argument("--frame", type=int, default=512, help="Tamaño de frame n")
    parser.add_argument("--m", type=int, default=256, help="Número de mediciones m (filas de Phi)")
    parser.add_argument("--method", type=str, default="newton", choices=["pinv","newton"], help="Reconstrucción via pinv o newton")
    parser.add_argument("--overlap", type=int, default=495, help="Overlap en muestras")
    parser.add_argument("--duration", type=float, default=10.0, help="Duración máxima en segundos (default: 10s)")
    args = parser.parse_args()
    if not os.path.exists(args.wav):
        raise FileNotFoundError("No se encontró el archivo wav.")
    run_example(args.wav, frame_len=args.frame, m_measure=args.m, method=args.method, 
                overlap=args.overlap, max_duration=args.duration)
