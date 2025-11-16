import numpy as np
import soundfile as sf
from scipy.fftpack import idct
import matplotlib.pyplot as plt
import time
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class AudioDecompressor:
    """
    Descompresion de audio usando Compressive Sensing + DCT
    Basado en el paper: "Provably secure and efficient audio compression 
    based on compressive sensing" (Abood et al., 2023)
    """
    
    def __init__(self, compression_rate, frame_size):
        self.compression_rate = compression_rate
        self.frame_size = frame_size
        self.num_coeffs = int(frame_size * compression_rate)
        
        print(f"Configuracion de descompresion:")
        print(f"  - Frame size: {self.frame_size}")
        print(f"  - Coeficientes: {self.num_coeffs} ({compression_rate*100}%)")
        print(f"  - DCT: Activado")
    
    def generate_measurement_matrix(self, seed):
        """Regenera la matriz Gaussiana con la semilla original"""
        np.random.seed(seed)
        matrix = np.random.randn(self.num_coeffs, self.num_coeffs)
        matrix = matrix / np.sqrt(self.num_coeffs)
        return matrix
    
    def reconstruct_audio(self, compressed_signal, original_length, seed, max_val):
        """
        Reconstruccion usando Moore-Penrose pseudoinverse (Algorithm 2 del paper):
        1. Regenerar matriz de medicion con la semilla
        2. Calcular pseudoinversa de Moore-Penrose (Ecuacion 6)
        3. Desencriptar coeficientes (X = A+ * Y)
        4. Rellenar coeficientes DCT con ceros
        5. Aplicar IDCT para reconstruir senal temporal
        """
        start_time = time.time()
        
        # Regenerar matriz de medicion
        measurement_matrix = self.generate_measurement_matrix(seed)
        
        # Calcular pseudoinversa de Moore-Penrose (Ecuacion 6 del paper)
        # A+ = A^T * (A * A^T)^-1
        pseudoinverse = np.linalg.pinv(measurement_matrix)
        
        # Dividir en frames
        num_frames = len(compressed_signal) // self.num_coeffs
        compressed_frames = compressed_signal.reshape(num_frames, self.num_coeffs)
        
        reconstructed_frames = []
        
        for encrypted_coeffs in compressed_frames:
            # Desencriptar usando pseudoinversa (Ecuacion 7)
            sparse_coeffs = pseudoinverse @ encrypted_coeffs
            
            # Rellenar coeficientes DCT con ceros
            full_dct_coeffs = np.zeros(self.frame_size)
            full_dct_coeffs[:self.num_coeffs] = sparse_coeffs
            
            # Aplicar IDCT para reconstruir frame temporal
            reconstructed_frame = idct(full_dct_coeffs, norm='ortho')
            
            reconstructed_frames.append(reconstructed_frame)
        
        # Concatenar frames y recortar al tamano original
        reconstructed_signal = np.concatenate(reconstructed_frames)[:original_length]
        
        # Desnormalizar
        reconstructed_signal = reconstructed_signal * max_val
        
        stats = {'reconstruction_time': time.time() - start_time}
        
        return reconstructed_signal, stats
    
    def load_compressed(self, filename):
        """Carga datos comprimidos"""
        with open(filename, 'rb') as f:
            package = pickle.load(f)
        
        compressed_signal = (
            package['compressed_data'].astype(np.float64) / 32767.0 * 
            package['max_compressed']
        )
        
        return compressed_signal, package
    
    def evaluate_reconstruction(self, original, reconstructed):
        """
        Metricas de calidad segun el paper (Seccion 4.2, 4.3, 4.4):
        - Pearson Correlation (R)
        - MSE (Mean Square Error)
        - PSNR (Peak Signal-to-Noise Ratio)
        - SSIM (Structural Similarity Index)
        """
        
        # 1. Pearson Correlation (Seccion 4.2)
        correlation = np.corrcoef(original, reconstructed)[0, 1]
        
        # 2. MSE (Seccion 4.3)
        mse = np.mean((original - reconstructed) ** 2)
        
        # 3. PSNR (Seccion 4.3)
        max_val = np.max(np.abs(original))
        if mse > 0:
            psnr = 10 * np.log10((max_val ** 2) / mse)
        else:
            psnr = float('inf')
        
        # 4. SSIM (Seccion 4.4)
        mean_x = np.mean(original)
        mean_y = np.mean(reconstructed)
        var_x = np.var(original)
        var_y = np.var(reconstructed)
        cov_xy = np.mean((original - mean_x) * (reconstructed - mean_y))
        
        # Constantes para evitar division por cero
        L = max_val  # Rango de valores
        c1 = (0.01 * L) ** 2
        c2 = (0.03 * L) ** 2
        
        ssim = ((2 * mean_x * mean_y + c1) * (2 * cov_xy + c2)) / \
               ((mean_x**2 + mean_y**2 + c1) * (var_x + var_y + c2))
        
        return {
            'Correlation': correlation,
            'MSE': mse,
            'PSNR': psnr,
            'SSIM': ssim
        }
    
    def plot_comparison(self, original, compressed, reconstructed, output_file):
        """Genera graficas comparativas"""
        fig = plt.figure(figsize=(16, 8))
        
        # Configuracion de grilla
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # 1. Senal Original (completa)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(original, linewidth=0.5, color='blue', alpha=0.8)
        ax1.set_title('Senal Original', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Amplitud', fontsize=10)
        ax1.set_xlabel('Muestras', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. Senal Comprimida
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.plot(compressed, linewidth=0.5, color='orange', alpha=0.8)
        ax2.set_title('Senal Comprimida (CS+DCT)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Amplitud', fontsize=10)
        ax2.set_xlabel('Muestras', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Senal Reconstruida (completa)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(reconstructed, linewidth=0.5, color='green', alpha=0.8)
        ax3.set_title('Senal Reconstruida', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Amplitud', fontsize=10)
        ax3.set_xlabel('Muestras', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Grafica guardada: {output_file}")
        print("\nMostrando graficas comparativas...")
        plt.show()


def decompress_audio_file(compressed_file, output_file=None, original_file=None, 
                          sample_rate=44100):
    """
    Funcion principal para descomprimir un archivo de audio
    """
    print("\n" + "=" * 70)
    print("DESCOMPRESION DE AUDIO CON COMPRESSIVE SENSING + DCT")
    print("=" * 70)
    
    # Cargar archivo comprimido
    print(f"\nCargando: {compressed_file}")
    
    with open(compressed_file, 'rb') as f:
        package = pickle.load(f)
    
    # Crear descompresor
    decompressor = AudioDecompressor(
        compression_rate=package['compression_rate'],
        frame_size=package['frame_size']
    )
    
    # Cargar datos comprimidos
    compressed_signal, _ = decompressor.load_compressed(compressed_file)
    
    # Reconstruir
    print("\n" + "-" * 70)
    print("RECONSTRUYENDO...")
    print("-" * 70)
    
    reconstructed_signal, stats = decompressor.reconstruct_audio(
        compressed_signal,
        package['original_length'],
        package['seed'],
        package['max_val']
    )
    
    print(f"Muestras reconstruidas: {len(reconstructed_signal)}")
    print(f"Tiempo: {stats['reconstruction_time']:.4f}s")
    
    # Guardar audio reconstruido
    if output_file is None:
        base_name = os.path.splitext(compressed_file)[0]
        output_file = f"{base_name}_reconstructed.wav"
    
    sf.write(output_file, reconstructed_signal, sample_rate)
    print(f"\nAudio guardado: {output_file}")
    
    # Evaluar calidad si se proporciona el original
    metrics = None
    if original_file is not None:
        print("\n" + "=" * 70)
        print("ANALISIS DE CALIDAD (Paper: Abood et al., 2023)")
        print("=" * 70)
        
        original_signal, _ = sf.read(original_file)
        if len(original_signal.shape) > 1:
            original_signal = original_signal[:, 0]
        
        metrics = decompressor.evaluate_reconstruction(original_signal, reconstructed_signal)
        
        print("\nMetricas de Reconstruccion:")
        print("-" * 70)
        print(f"Pearson Correlation (R):  {metrics['Correlation']:.6f}")
        print(f"MSE:                      {metrics['MSE']:.6e}")
        print(f"PSNR:                     {metrics['PSNR']:.2f} dB")
        print(f"SSIM:                     {metrics['SSIM']:.6f}")
        
        # Interpretacion basada en valores del paper (Tabla 3)
        if metrics['Correlation'] >= 0.99:
            status = "EXCELENTE (>= 0.99)"
        elif metrics['Correlation'] >= 0.98:
            status = "MUY BUENA (>= 0.98)"
        elif metrics['Correlation'] >= 0.95:
            status = "BUENA (>= 0.95)"
        elif metrics['Correlation'] >= 0.90:
            status = "REGULAR (>= 0.90)"
        else:
            status = "BAJA (< 0.90)"
        
        print(f"\nCalidad de Reconstruccion: {status}")
        print("-" * 70)
        
        # Generar graficas comparativas
        print("\n" + "-" * 70)
        print("GENERANDO GRAFICAS COMPARATIVAS")
        print("-" * 70)
        
        plot_file = os.path.splitext(compressed_file)[0] + '_comparison.png'
        decompressor.plot_comparison(
            original_signal, 
            compressed_signal, 
            reconstructed_signal, 
            plot_file
        )
    else:
        print("\nNota: Proporcione el archivo original con --original para ver metricas")
    
    print("\n" + "=" * 70)
    print("Descompresion completada")
    print("=" * 70 + "\n")
    
    return output_file, metrics


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUso: python decompress.py <archivo_comprimido> [opciones]")
        print("\nOpciones:")
        print("  --output ARCHIVO     Nombre de archivo de salida (default: auto)")
        print("  --original ARCHIVO   Archivo original para evaluar calidad")
        print("  --sample-rate RATE   Tasa de muestreo (default: 44100)")
        print("\nEjemplo:")
        print("  python decompress.py audio_50pct_f1024_dct.pkl")
        print("  python decompress.py compressed.pkl --original audio.wav")
        print("  python decompress.py compressed.pkl --output resultado.wav --sample-rate 22050\n")
        sys.exit(1)
    
    compressed_file = sys.argv[1]
    
    # Parsear opciones
    output_file = None
    original_file = None
    sample_rate = 44100
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--output':
            output_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--original':
            original_file = sys.argv[i + 1]
            i += 2
        elif sys.argv[i] == '--sample-rate':
            sample_rate = int(sys.argv[i + 1])
            i += 2
        else:
            i += 1
    
    # Descomprimir
    decompress_audio_file(
        compressed_file,
        output_file=output_file,
        original_file=original_file,
        sample_rate=sample_rate

    )
