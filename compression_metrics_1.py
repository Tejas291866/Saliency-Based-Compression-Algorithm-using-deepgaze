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
    # Assuming original and compressed are already grayscale or handled by SSIM's channel_axis
    return ssim(original, compressed, channel_axis=2 if original.ndim == 3 else None) # Adjusted for grayscale or color

def calculate_compression_ratio(original_file, encoded_file):
    """Calculate the compression ratio."""
    original_size = os.path.getsize(original_file)
    encoded_size = os.path.getsize(encoded_file) # Renamed to encoded_file
    return original_size / encoded_size

def measure_compression_metrics(original_video_path, encoded_video_path, decoded_video_path):
    """Calculate PSNR, SSIM, and compression ratio for the video.
    
    Args:
        original_video_path (str): Path to the original video file.
        encoded_video_path (str): Path to the *encoded* (compressed) video file
                                   used for compression ratio.
        decoded_video_path (str): Path to the *decoded* video file (output of your decoder)
                                  used for PSNR/SSIM.
    """
    original_capture = cv2.VideoCapture(original_video_path)
    decoded_capture = cv2.VideoCapture(decoded_video_path) # Changed to decoded_capture

    frame_count_orig = int(original_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count_dec = int(decoded_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_count = min(frame_count_orig, frame_count_dec)

    psnr_values = []
    ssim_values = []

    print(f"[INFO] Comparing {frame_count} frames for PSNR/SSIM...")

    for _ in tqdm(range(frame_count), desc="Processing frames", ncols=100):
        retval_orig, orig_frame = original_capture.read()
        retval_dec, dec_frame = decoded_capture.read() # Changed to dec_frame

        if not retval_orig or not retval_dec:
            break

        # Convert to grayscale for PSNR
        orig_gray = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2GRAY)
        dec_gray = cv2.cvtColor(dec_frame, cv2.COLOR_BGR2GRAY)

        # Resize decoded to match original (important if decoder output resolution differs)
        # Assuming original_frame is the reference for size
        dec_gray_resized = cv2.resize(dec_gray, (orig_gray.shape[1], orig_gray.shape[0]))
        dec_frame_resized = cv2.resize(dec_frame, (orig_frame.shape[1], orig_frame.shape[0]))


        # Compute metrics
        psnr_value = calculate_psnr(orig_gray, dec_gray_resized)
        ssim_value = calculate_ssim(orig_frame, dec_frame_resized) # SSIM needs color for channel_axis=2

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)

    original_capture.release()
    decoded_capture.release() # Release decoded capture

    if len(psnr_values) == 0:
        raise RuntimeError("No frames were processed. Check your video paths or decoding issues.")

    avg_psnr = np.mean(psnr_values)
    avg_ssim = np.mean(ssim_values)
    
    # Compression ratio still uses original and *encoded* file
    compression_ratio = calculate_compression_ratio(original_video_path, encoded_video_path)

    return avg_psnr, avg_ssim, compression_ratio

# === Example Usage ===

original_video_path = r'video\segments\cube\Y0P0\2\Y0P0_0.mp4'
encoded_video_path = r'video\segments\sali_encoded\Y0P0\2\Y0P0_0.mp4' # This is your compressed file on disk
decoded_video_path = r'D:\compression algo\video\segments\sali_decoded\Y0P0\2\decoded_output.mp4' # <--- YOU NEED TO GENERATE THIS FILE WITH YOUR DECODER FIRST

print(f"\n[INFO] Measuring compression metrics for:")
print(f"Original:   {original_video_path}")
print(f"Encoded:    {encoded_video_path}")
print(f"Decoded:    {decoded_video_path}\n") # Added decoded path for clarity

# Ensure you have run your custom decoder and it has produced 'decoded_output_Y0P0_0.mp4'
# BEFORE running this Python script.
if not os.path.exists(decoded_video_path):
    print(f"[ERROR] Decoded video file not found: {decoded_video_path}")
    print("Please ensure your separate decoder has run and generated this file.")
else:
    psnr, ssim_value, compression_ratio = measure_compression_metrics(original_video_path, encoded_video_path, decoded_video_path)

    print(f"\n[RESULTS]")
    print(f"PSNR:              {psnr:.2f} dB")
    print(f"SSIM:              {ssim_value:.4f}")
    print(f"Compression Ratio: {compression_ratio:.2f}x")