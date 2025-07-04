import av
import cv2
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from sali_encode_3 import encode  # Import your existing encode function

# ===== Your Metrics Functions =====
def calculate_psnr(original, compressed):
    """Calculate Peak Signal to Noise Ratio (PSNR)."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def calculate_ssim(original, compressed):
    """Calculate Structural Similarity Index (SSIM)."""
    return ssim(original, compressed, channel_axis=2)

def calculate_compression_ratio(original_file, compressed_file):
    """Calculate the compression ratio."""
    original_size = os.path.getsize(original_file)
    compressed_size = os.path.getsize(compressed_file)
    return original_size / compressed_size

def measure_compression_metrics(original_file, compressed_file):
    """Calculate PSNR, SSIM, and compression ratio for the video."""
    original_capture = cv2.VideoCapture(original_file)
    compressed_capture = cv2.VideoCapture(compressed_file)

    frame_count_orig = int(original_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_comp = int(compressed_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(frame_count_orig, frame_count_comp)

    psnr_values = []
    ssim_values = []

    print(f"[INFO] Comparing {frame_count} frames...")

    for _ in tqdm(range(frame_count), desc="Processing frames", ncols=100):
        retval_orig, orig_frame = original_capture.read()
        retval_comp, comp_frame = compressed_capture.read()

        if not retval_orig or not retval_comp:
            break

        # Convert to grayscale for PSNR
        orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_BGR2GRAY)
        comp_resized = cv2.resize(comp_gray, (orig_gray.shape[1], orig_gray.shape[0]))

        # Compute metrics
        psnr_values.append(calculate_psnr(orig_gray, comp_resized))
        ssim_values.append(calculate_ssim(orig_frame, 
                                       cv2.resize(comp_frame, 
                                                 (orig_frame.shape[1], orig_frame.shape[0]))))

    original_capture.release()
    compressed_capture.release()

    if len(psnr_values) == 0:
        raise RuntimeError("No frames were processed. Check your video paths or decoding issues.")

    return {
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'ratio': calculate_compression_ratio(original_file, compressed_file)
    }

# ===== Compression Function =====
def compress_with_bitrate(input_path, output_path, target_bitrate_kbps):
    """Compress video with specific bitrate"""
    container = av.open(input_path)
    video_stream = container.streams.video[0]
    
    output_container = av.open(output_path, mode='w')
    stream = output_container.add_stream('libx264', rate=video_stream.average_rate)
    
    stream.bit_rate = target_bitrate_kbps * 1000  # Convert to bps
    stream.pix_fmt = 'yuv420p'
    stream.options = {
        'preset': 'slow',
        'x264-params': f"aq-mode=3:psy-rd=1.0:deblock=-1,-1"
    }

    for frame in container.decode(video=0):
        frame_bgr = frame.to_ndarray(format='bgr24')
        encoded_frame = encode(frame_bgr, salient_patch_size=4, num_col=2)
        
        if not hasattr(stream, 'width'):
            stream.width = encoded_frame.shape[1]
            stream.height = encoded_frame.shape[0]
            
        frame_rgb = cv2.cvtColor(encoded_frame, cv2.COLOR_BGR2RGB)
        output_container.mux(stream.encode(av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')))
    
    # Flush encoder
    for packet in stream.encode():
        output_container.mux(packet)
    
    container.close()
    output_container.close()

# ===== Main Test Function =====
def run_bitrate_tests(input_video, output_folder, bitrates=None):
    """Test different bitrate settings"""
    if bitrates is None:
        bitrates = [2500, 5000, 8000, 12000]  # Default test values in kbps
    
    results = []
    os.makedirs(output_folder, exist_ok=True)
    
    for bitrate in bitrates:
        output_path = f"{output_folder}/output_{bitrate}kbps.mp4"
        print(f"\n{'='*40}")
        print(f"Testing {bitrate} kbps...")
        print(f"Output: {output_path}")
        
        # Step 1: Compress with target bitrate
        compress_with_bitrate(input_video, output_path, bitrate)
        
        # Step 2: Calculate metrics
        print("Calculating metrics...")
        metrics = measure_compression_metrics(input_video, output_path)
        metrics['bitrate'] = bitrate
        
        results.append(metrics)
        print(f"\nResults for {bitrate}kbps:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"Ratio: {metrics['ratio']:.2f}x")
    
    # Save and display results
    df = pd.DataFrame(results)
    results_file = f"{output_folder}/test_results_encode3.csv"
    df.to_csv(results_file, index=False)
    
    print("\n" + "="*40)
    print("All tests completed!")
    print(f"Results saved to: {results_file}")
    print("="*40 + "\n")
    
    return df

if __name__ == "__main__":
    # Configuration
    input_video = "video/segments/cube/Y0P0/2/Y0P0_0.mp4"  # Your input video
    output_folder = "bitrate_tests_output"
    
    # Custom bitrates to test (in kbps)
    test_bitrates = [2000, 4000, 6000, 8000, 10000]  # Modify as needed
    
    print("Starting bitrate optimization tests...")
    results = run_bitrate_tests(input_video, output_folder, bitrates=test_bitrates)
    
    # Print final results table
    print("\nFinal Results Summary:")
    print(results[['bitrate', 'psnr', 'ssim', 'ratio']].to_string(index=False))