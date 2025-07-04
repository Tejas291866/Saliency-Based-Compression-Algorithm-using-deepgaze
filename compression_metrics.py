import cv2
import os
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

def calculate_psnr(original, compressed):
    """Calculate Peak Signal to Noise Ratio (PSNR)."""
    mse = np.mean((original - compressed) ** 2)
    if mse == 0:
        return 100  # perfect match
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

        # Convert to grayscale
        orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        comp_gray = cv2.cvtColor(comp_frame, cv2.COLOR_BGR2GRAY)

        # Resize compressed to match original
        comp_resized = cv2.resize(comp_gray, (orig_gray.shape[1], orig_gray.shape[0]))
        comp_frame_resized = cv2.resize(comp_frame, (orig_frame.shape[1], orig_frame.shape[0]))

        # Compute metrics
        psnr_value = calculate_psnr(orig_gray, comp_resized)
        ssim_value = calculate_ssim(orig_frame, comp_frame_resized)

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    original_capture.release()
    compressed_capture.release()

    if len(psnr_values) == 0:
        raise RuntimeError("No frames were processed. Check your video paths or decoding issues.")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    compression_ratio = calculate_compression_ratio(original_file, compressed_file)

    return avg_psnr, avg_ssim, compression_ratio

# === Example Usage ===

original_video_path = r'video\segments\cube\Y0P0\2\Y0P0_0.mp4'
compressed_video_path = r'video\segments\sali_encoded\Y0P0\2\Y0P0_0.mp4'

print(f"\n[INFO] Measuring compression metrics for:")
print(f"Original:   {original_video_path}")
print(f"Compressed: {compressed_video_path}\n")

psnr, ssim_value, compression_ratio = measure_compression_metrics(original_video_path, compressed_video_path)

print(f"\n[RESULTS]")
print(f"PSNR:              {psnr:.2f} dB")
print(f"SSIM:              {ssim_value:.4f}")
print(f"Compression Ratio: {compression_ratio:.2f}x")
