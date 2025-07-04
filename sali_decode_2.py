import os
import cv2
import numpy as np
from pyramid_b_decoding import pyramid_b_decoding

# Input and output paths
input_path = r"D:\compression algo\video\segments\sali_encoded\Y0P0\2\Y0P0_0.mp4"
output_dir = r"D:\compression algo\video\segments\sali_decoded\Y0P0\2"
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

output_file = os.path.join(output_dir, "decoded_output.mp4")
saliency_output_file = os.path.join(output_dir, "saliency_output.avi") # Or .mp4 if you prefer

# Open the compressed video
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {input_path}")

# Get original dimensions from the first frame
ret, frame = cap.read()
if not ret:
    raise RuntimeError("Failed to read the first frame of the video.")

height, width = frame.shape[:2]
h = height  # full height
w = width

# Calculate the width of the pyramid and saliency parts based on the encoder
pyramid_width = 2 * h
saliency_width = w - pyramid_width # The remaining width is the saliency part

# Set up VideoWriters
fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID' for AVI
decoded_size = (w // 3 * 3, h // 2 * 2)  # Approximate original dimensions for pyramid
out_decoded = cv2.VideoWriter(output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), decoded_size)

saliency_size = (saliency_width, h)
out_saliency = cv2.VideoWriter(saliency_output_file, fourcc, cap.get(cv2.CAP_PROP_FPS), saliency_size)


# Reset capture to start
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Split frame into pyramid and saliency
    pyra = frame[:, :pyramid_width, :]
    sali = frame[:, pyramid_width:, :]

    # Decode the pyramid part
    decoded_frame_partial = pyramid_b_decoding(pyra)
    decoded_frame = cv2.resize(decoded_frame_partial, decoded_size)
    out_decoded.write(decoded_frame)

    # Optionally, you can process or save the saliency part here
    # For now, let's just save it as a separate video
    out_saliency.write(sali)

# Release everything
cap.release()
out_decoded.release()
out_saliency.release()
print(f"Decoding complete. Decoded output saved to: {output_file}")
print(f"Saliency output saved to: {saliency_output_file}")