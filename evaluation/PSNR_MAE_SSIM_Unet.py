import cv2
import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from tqdm import tqdm
import re

def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def calculate_ssim(img1, img2):
    min_dim = min(img1.shape[0], img1.shape[1], img2.shape[0], img2.shape[1])
    win_size = min(7, min_dim - 1)
    return ssim(img1, img2, multichannel=True, win_size=win_size, channel_axis=-1)

def extract_subject_id(filename):
    match = re.search(r'(.*?)__slice', filename)
    if match:
        return match.group(1)
    else:
        return None

def evaluate_metrics(ground_truth_dir, generated_dir):
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    subject_metrics = defaultdict(lambda: {'psnr': [], 'mae': [], 'ssim': []})

    for gt_file in tqdm(ground_truth_files, desc="Calculating evaluation metrics for GT vs synthetic data"):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        gt_image = cv2.imread(gt_path)
        if gt_image is None:
            print(f"Failed to load ground truth image: {gt_path}")
            continue

        gen_file = gt_file  # Since both are in jpg format, filenames should be the same
        gen_path = os.path.join(generated_dir, gen_file)

        # Debugging statements
        if not os.path.exists(gen_path):
            print(f"Generated image does not exist: {gen_path}")
            continue
        gen_image = cv2.imread(gen_path)
        if gen_image is None:
            print(f"Failed to load generated image: {gen_path}")
            continue

        psnr = calculate_psnr(gt_image, gen_image)
        mae = calculate_mae(gt_image, gen_image)

        try:
            ssim_value = calculate_ssim(gt_image, gen_image)
        except ValueError as e:
            print(f"Skipping SSIM calculation for {gt_file} and {gen_file} due to error: {e}")
            ssim_value = np.nan

        subject_id = extract_subject_id(gt_file)
        if subject_id is None:
            print(f"Filename format error: {gt_file}")
            continue

        subject_metrics[subject_id]['psnr'].append(psnr)
        subject_metrics[subject_id]['mae'].append(mae)
        subject_metrics[subject_id]['ssim'].append(ssim_value)

    subject_data = []

    for subject_id, metrics in subject_metrics.items():
        avg_psnr = np.mean(metrics['psnr'])
        stddev_psnr = np.std(metrics['psnr'])
        avg_mae = np.mean(metrics['mae'])
        stddev_mae = np.std(metrics['mae'])
        avg_ssim = np.nanmean(metrics['ssim'])
        stddev_ssim = np.nanstd(metrics['ssim'])

        print(f'Subject {subject_id} - Average PSNR: {avg_psnr}, Std Dev: {stddev_psnr}')
        print(f'Subject {subject_id} - Average MAE: {avg_mae}, Std Dev: {stddev_mae}')
        print(f'Subject {subject_id} - Average SSIM: {avg_ssim}, Std Dev: {stddev_ssim}')

        subject_data.append([subject_id, avg_psnr, stddev_psnr, avg_mae, stddev_mae, avg_ssim, stddev_ssim])

    df = pd.DataFrame(subject_data, columns=['Subject', 'Avg_PSNR', 'StdDev_PSNR', 'Avg_MAE', 'StdDev_MAE', 'Avg_SSIM', 'StdDev_SSIM'])
    df.to_csv('metrics_per_subject.csv', index=False)

    return df

if __name__ == "__main__":
    ground_truth_dir = 'data/slices_n98_s320x320_z88/test/B'
    generated_dir = 'Unetbaseline/results/predictions/10_epoch'

    evaluate_metrics(ground_truth_dir, generated_dir)
