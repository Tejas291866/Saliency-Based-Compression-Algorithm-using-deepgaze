import av
import os
import cv2
import numpy as np
import pandas as pd
import multiprocessing
from saliency_update import *
from ai_saliency import SaliencyPredictor



saliency_predictor = SaliencyPredictor(default_width=256, default_height=256)

def rotate_img(img, iteration = 1):
    #0: remain same, +n: rotate left in 90*n degree, -n: rotate right in 90*n degree (some part of img is sliced off)
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    rotated_img = np.zeros(img.shape, dtype = np.uint8)

    if iteration == -1:

        for idx in range(height):
            rotated_img[:, height-1-idx] = img[idx, :]

    elif iteration == 1:

        for idx in range(height):
            rotated_img[:, idx] = np.flip(img[idx, :], axis = 0)

    elif iteration == 0:

        rotated_img = img

    else:

        for idx in range(height):
            rotated_img[height-1-idx, :] = np.flip(img[idx, :], axis = 0)

    return rotated_img

def frame_split(frame):
    #return a list of screens in an order of right, left, up, down, front and back
    if len(frame.shape) == 3:
        height, width, _ = frame.shape
    else:
        height, width = frame.shape

    unit_height = int(height / 2)
    unit_width = int(width / 3)

    #1st layer
    right_frame = frame[:unit_height, :unit_width]
    left_frame = frame[:unit_height, unit_width:2*unit_width]
    up_frame = frame[:unit_height, 2*unit_width:]
    #2nd layer
    down_frame = frame[unit_height:, :unit_width]
    front_frame = frame[unit_height:, unit_width:2*unit_width]
    back_frame = frame[unit_height:, 2*unit_width:]

    return [right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]

def trapezoid_1(img): # starting from small frame / 2 + 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(row / 4)):
        layer = cv2.resize(img[idx*4:(idx+1)*4, :], (int(row/2) + 2 + 2*idx,1))
        _, length, _ = layer.shape

        empty_pallete[idx, int(row/4)-1-idx: int(row/4)-1-idx + length, :  ] = layer

        #cv2.imshow('original', img[idx*4:(idx+1)*4, :])
        #cv2.imshow('resized', layer)

    #cv2.imshow('pallete', empty_pallete)
    #cv2.waitKey(0)

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def trapezoid_2(img): # starting from small frame / 2

    if len(img.shape) == 3:
        row, col, channel = img.shape

    else:
        row, col = img.shape
        channel = 1

    empty_pallete = np.zeros((row, col, channel), dtype = np.uint8)

    for idx in range(int(row / 4)):
        layer = cv2.resize(img[idx*4:(idx+1)*4, :], (int(row/2) + 2*idx,1))
        _, length, _ = layer.shape

        empty_pallete[idx, int(row/4)-idx: int(row/4)-idx + length, :  ] = layer

    return empty_pallete #shape of row, col, channel (1024, 1024, 3)

def pyramid_b_encoding(img):

    #[right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]
    split_img = frame_split(img)

    row, col, channel = split_img[0].shape
    pallete = np.zeros((row, col, channel), dtype = np.uint8)

    #right: rotate left once and trapezoid_1 and rotate right, indexing from the right
    #left: rotate right once and trapezoid_1 and rotate left, indexing from the left
    #up: rotate none and trapezoid_2 rotate twice, indexing from the bottom
    #down: rotate twice and trapezoid_2, indexing from the top

    right_img = split_img[0]
    right_img = rotate_img(right_img, iteration = 1)

    right_trapezoid = trapezoid_1(right_img)
    right_trapezoid = rotate_img(right_trapezoid, iteration = -1)

    left_img = split_img[1]
    left_img = rotate_img(left_img, iteration = -1)

    left_trapezoid = trapezoid_1(left_img)
    left_trapezoid = rotate_img(left_trapezoid, iteration = 1)

    up_img = split_img[2]

    up_trapezoid = trapezoid_2(up_img)
    up_trapezoid = rotate_img(up_trapezoid, iteration = 2)


    down_img = split_img[3]

    down_img = rotate_img(down_img, iteration = 2)
    down_trapezoid = trapezoid_2(down_img)

    front_img = split_img[4]

    back_img = split_img[5]
    back_img = cv2.resize(back_img, None, fx = 0.5, fy = 0.5)

    height, width, _ = back_img.shape

    pallete[:, :int(col/4), :] = np.maximum(pallete[:, :int(col/4), :], right_trapezoid[:, -int(col/4):, :]) #right
    pallete[:, -int(col/4):, :] = np.maximum(pallete[:, -int(col/4):, :], left_trapezoid[:, :int(col/4), :]) #left
    pallete[:int(row/4), :, :] = np.maximum(pallete[:int(row/4), :, :], up_trapezoid[-int(row/4):, :, :]) #up
    pallete[-int(row/4):, :, :] = np.maximum(pallete[-int(row/4):, :, :], down_trapezoid[:int(row/4), :, :]) #down
    pallete[int(row/4):int(row/4)+height, int(col/4):int(col/4)+width, :] = back_img

    pyra_c_frame = np.concatenate([front_img, pallete], axis = 1)

    return pyra_c_frame

def frame_split_1(frame):
    #return a list of screens in an order of right, left, up, down, front and back
    if len(frame.shape) == 3:
        height, width, _ = frame.shape
    else:
        height, width = frame.shape

    unit_height = int(height / 2)
    unit_width = int(width / 3)

    #1st layer
    right_frame = frame[:unit_height, :unit_width]
    left_frame = frame[:unit_height, unit_width:2*unit_width]
    up_frame = frame[:unit_height, 2*unit_width:]
    #2nd layer
    down_frame = frame[unit_height:, :unit_width]
    front_frame = frame[unit_height:, unit_width:2*unit_width]
    back_frame = frame[unit_height:, 2*unit_width:]

    return [right_frame, left_frame, up_frame, down_frame, front_frame, back_frame]

def json_sort_by_name(y_p_combo_path):
    json_file_list = np.array(os.listdir(y_p_combo_path))
    idx_list = [int(json_file.split('.')[0].split('_')[-1]) for json_file in json_file_list]
    sorted_idx_list = np.argsort(idx_list)
    sorted_json_file_list = json_file_list[sorted_idx_list]

    return sorted_json_file_list

def json_path_generate(path, j2f_ratio):

    json_path_list = []

    sorted_json_file_list = json_sort_by_name(path)

    for json_file in sorted_json_file_list:
        for _ in range(j2f_ratio):
            json_path_list.append(os.path.join(path, json_file))

    return np.array(json_path_list).ravel()

def saliency_patching(cube_frame, saliency_map, salient_patch_size, num_col):
    frame_height, frame_width, _ = cube_frame.shape
    side_length = int(frame_width / 3)
    window_size = int(side_length / salient_patch_size)  # patch size in pixels

    saliency_map = cv2.resize(saliency_map, (frame_width, frame_height))
    saliency_map = saliency_map.astype(np.float32) / 255.0  # normalize

    patch_scores = []
    patches = []
    
    for y in range(0, frame_height - window_size + 1, window_size):
        for x in range(0, frame_width - window_size + 1, window_size):
            patch_saliency = saliency_map[y:y + window_size, x:x + window_size]
            avg_score = np.mean(patch_saliency)
            var_score = np.var(patch_saliency)
            score = 0.85 * avg_score + 0.15 * var_score
            patch = cube_frame[y:y + window_size, x:x + window_size]
            patch_scores.append((score, patch))

    patch_scores.sort(key=lambda x: x[0], reverse=True)

    # Get top patches
    top_patches = [patch for _, patch in patch_scores[:salient_patch_size * num_col]]

    # Assemble into image
    rows = []
    for i in range(0, len(top_patches), num_col):
        row = np.concatenate(top_patches[i:i + num_col], axis=1)
        rows.append(row)
    sali_img = np.concatenate(rows, axis=0)

    return sali_img


def encode(cube_frame, salient_patch_size, num_col):
    pyra = pyramid_b_encoding(cube_frame)
    saliency_map = saliency_predictor.predict(cube_frame)  # AI-generated saliency map
    sali = saliency_patching(cube_frame, saliency_map, salient_patch_size, num_col)
    
    combined = np.concatenate([pyra, sali], axis=1)
    
    # Optional: slight denoising to smooth transitions
    combined = cv2.bilateralFilter(combined, d=5, sigmaColor=50, sigmaSpace=50)

    return combined


def compress_video_with_pyav(input_video_path, output_video_path, salient_patch_size=4, num_col=2):
    container = av.open(input_video_path)
    fps = container.streams.video[0].average_rate
    width = None
    height = None

    output_container = av.open(output_video_path, mode='w')
    stream = output_container.add_stream('libx264', rate=fps)
    stream.bit_rate = 2000  # Optional: set a reasonable bit rate
    stream.pix_fmt = 'yuv420p'
    stream.options = {'crf': '26', 'preset': 'slow'}  # CRF controls quality/size


    for frame in container.decode(video=0):
        frame_bgr = frame.to_ndarray(format='bgr24')
        encoded_frame = encode(frame_bgr, salient_patch_size, num_col)

        if width is None or height is None:
            height, width, _ = encoded_frame.shape
            stream.width = width
            stream.height = height
            stream.pix_fmt = 'yuv420p'

        frame_rgb = cv2.cvtColor(encoded_frame, cv2.COLOR_BGR2RGB)
        video_frame = av.VideoFrame.from_ndarray(frame_rgb, format='rgb24')
        packet = stream.encode(video_frame)
        if packet:
            output_container.mux(packet)

    # Flush encoder
    for packet in stream.encode():
        output_container.mux(packet)

    container.close()
    output_container.close()

input_base_path = 'video/segments/cube'
output_base_path = 'video/segments/sali_encoded'

for y_p_combo in os.listdir(input_base_path):
    y_p_combo_path = os.path.join(input_base_path, y_p_combo)
    y_p_result_path = os.path.join(output_base_path, y_p_combo)

    for duration in os.listdir(y_p_combo_path):
        duration_path = os.path.join(y_p_combo_path, duration)
        duration_result_path = os.path.join(y_p_result_path, duration)

        os.makedirs(duration_result_path, exist_ok=True)

        for file_name in sorted(os.listdir(duration_path)):
            input_file = os.path.join(duration_path, file_name)
            output_file = os.path.join(duration_result_path, file_name)

            compress_video_with_pyav(input_file, output_file)











