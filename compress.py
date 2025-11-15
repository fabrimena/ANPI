import numpy as np
import soundfile as sf
from scipy.fftpack import dct
import time
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

class AudioCompressor:
    """
    Compresion de audio usando Compressive Sensing + DCT
    Basado en el paper: "Provably secure and efficient audio compression 
    based on compressive sensing" (Abood et al., 2023)
    """
    
    def __init__(self, compression_rate=0.5, frame_size=1024):
        self.compression_rate = compression_rate
        self.frame_size = frame_size
        self.num_coeffs = int(frame_size * compression_rate)
        self.seed = None
        
        print(f"Configuracion de compresion:")
        print(f"  - Frame size: {self.frame_size}")
        print(f"  - Coeficientes: {self.num_coeffs} ({compression_rate*100}%)")
        print(f"  - DCT: Activado")
    
    def preprocess_audio(self, audio_signal):
        """Pre-procesamiento para mejorar SNR"""
        max_val = np.max(np.abs(audio_signal))
        if max_val > 0:
            normalized = audio_signal / max_val
        else:
            normalized = audio_signal
        
        return normalized, max_val
    
    def generate_measurement_matrix(self, seed=None):
        """Genera matriz Gaussiana como en el paper (Seccion 4)"""
        if seed is None:
            self.seed = np.random.randint(0, 2**31)
        else:
            self.seed = seed
        
        np.random.seed(self.seed)
        matrix = np.random.randn(self.num_coeffs, self.num_coeffs)
        matrix = matrix / np.sqrt(self.num_coeffs)
        
        return matrix
    
    def compress_audio(self, audio_signal):
        """
        Compresion usando CS + DCT (Algorithm 1 del paper):
        1. Pre-procesar y dividir en frames
        2. DCT en cada frame para obtener representacion dispersa
        3. Tomar primeros N coeficientes DCT
        4. Multiplicar por matriz de medicion Gaussiana
        """
        start_time = time.time()
        
        processed_signal, max_val = self.preprocess_audio(audio_signal)
        measurement_matrix = self.generate_measurement_matrix()
        
        original_length = len(audio_signal)
        
        # Padding con reflexion
        pad_length = self.frame_size - (len(processed_signal) % self.frame_size)
        if pad_length != self.frame_size:
            processed_signal = np.pad(processed_signal, (0, pad_length), mode='reflect')
        
        # Dividir en frames (matrices pequenas como en el paper)
        num_frames = len(processed_signal) // self.frame_size
        frames = processed_signal.reshape(num_frames, self.frame_size)
        
        compressed_frames = []
        
        for frame in frames:
            # Aplicar DCT para obtener coeficientes dispersos
            dct_coeffs = dct(frame, norm='ortho')
            sparse_coeffs = dct_coeffs[:self.num_coeffs]
            
            # Multiplicar por matriz de medicion (Y = A * X)
            encrypted_coeffs = measurement_matrix @ sparse_coeffs
            compressed_frames.append(encrypted_coeffs)
        
        compressed_signal = np.concatenate(compressed_frames)
        
        stats = {
            'original_length': original_length,
            'compressed_length': len(compressed_signal),
            'compression_ratio': len(compressed_signal) / original_length,
            'compression_time': time.time() - start_time,
            'num_frames': num_frames,
            'seed': self.seed,
            'max_val': max_val
        }
        
        return compressed_signal, stats
    
    def save_compressed(self, compressed_signal, stats, filename="audio_compressed.pkl"):
        """Guarda datos comprimidos + metadatos"""
        max_compressed = np.max(np.abs(compressed_signal))
        if max_compressed > 0:
            normalized = compressed_signal / max_compressed
        else:
            normalized = compressed_signal
        
        compressed_int16 = (normalized * 32767).astype(np.int16)
        
        data_package = {
            'compressed_data': compressed_int16,
            'max_compressed': max_compressed,
            'original_length': stats['original_length'],
            'seed': stats['seed'],
            'compression_rate': self.compression_rate,
            'frame_size': self.frame_size,
            'max_val': stats['max_val']
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(data_package, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        size = os.path.getsize(filename)
        print(f"Archivo guardado: {filename} ({size:,} bytes, {size/1024:.2f} KB)")
        
        return size


def compress_audio_file(input_file, output_file=None, compression_rate=0.5, frame_size=1024):
    """
    Funcion principal para comprimir un archivo de audio
    """
    print("\n" + "=" * 70)
    print("COMPRESION DE AUDIO CON COMPRESSIVE SENSING + DCT")
    print("=" * 70)
    
    # Cargar audio
    audio_signal, fs = sf.read(input_file)
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal[:, 0]
    
    original_size = os.path.getsize(input_file)
    print(f"\nAudio original: {len(audio_signal)} muestras @ {fs} Hz")
    print(f"Tamano: {original_size:,} bytes ({original_size/1024:.2f} KB)\n")
    
    # Crear compresor
    compressor = AudioCompressor(
        compression_rate=compression_rate,
        frame_size=frame_size
    )
    
    # Comprimir
    print("\n" + "-" * 70)
    print("COMPRIMIENDO...")
    print("-" * 70)
    compressed_signal, stats = compressor.compress_audio(audio_signal)
    
    print(f"Muestras: {stats['original_length']} -> {stats['compressed_length']}")
    print(f"Ratio: {stats['compression_ratio']:.2%}")
    print(f"Tiempo: {stats['compression_time']:.4f}s")
    print(f"Semilla: {stats['seed']}\n")
    
    # Guardar
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_{int(compression_rate*100)}pct_f{frame_size}_dct.pkl"
    
    file_size = compressor.save_compressed(compressed_signal, stats, output_file)
    
    # Comparacion
    reduction = (1 - file_size / original_size) * 100
    
    print(f"\n{'RESULTADO':^70}")
    print("=" * 70)
    print(f"Original:     {original_size:>10,} bytes ({original_size/1024:>7.2f} KB)")
    print(f"Comprimido:   {file_size:>10,} bytes ({file_size/1024:>7.2f} KB)")
    print(f"Reduccion:    {reduction:>10.2f}%")
    print("=" * 70)
    
    print(f"\nCompresion completada: {output_file}\n")
    
    return output_file


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("\nUso: python compress.py <archivo_audio> [opciones]")
        print("\nOpciones:")
        print("  --rate RATE          Tasa de compresion (default: 0.5)")
        print("  --frame FRAME        Tamano de frame (default: 1024)")
        print("  --output ARCHIVO     Nombre de archivo de salida")
        print("\nEjemplo:")
        print("  python compress.py audio.wav")
        print("  python compress.py audio.wav --rate 0.3 --frame 2048")
        print("  python compress.py audio.wav --output compressed.pkl\n")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Parsear opciones
    compression_rate = 0.5
    frame_size = 1024
    output_file = None
    
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--rate':
            compression_rate = float(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--frame':
            frame_size = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--output':
            output_file = sys.argv[i + 1]
            i += 2
        else:
            i += 1
    
    # Comprimir
    compress_audio_file(
        input_file,
        output_file=output_file,
        compression_rate=compression_rate,
        frame_size=frame_size
    )