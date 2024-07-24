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

def extract_phase(filename):
    match = re.search(r'_(\d{1,2})\.\w+$', filename)
    if match:
        return match.group(1)
    else:
        return None

def evaluate_metrics(ground_truth_dir, generated_dir):
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    phase_metrics = defaultdict(lambda: {'psnr': [], 'mae': [], 'ssim': []})

    for gt_file in tqdm(ground_truth_files, desc="Calculating evaluation metrics for GT vs synthetic data"):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        gt_image = cv2.imread(gt_path)
        if gt_image is None:
            print(f"Failed to load ground truth image: {gt_path}")
            continue

        generated_folder = os.path.join(generated_dir, os.path.splitext(gt_file)[0])
        if not os.path.exists(generated_folder):
            print(f"Generated images folder does not exist: {generated_folder}")
            continue

        generated_files = sorted(os.listdir(generated_folder))
        psnr_values = []
        mae_values = []
        ssim_values = []

        for gen_file in generated_files:
            gen_path = os.path.join(generated_folder, gen_file)
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

            psnr_values.append(psnr)
            mae_values.append(mae)
            ssim_values.append(ssim_value)

        avg_psnr = np.mean(psnr_values)
        avg_mae = np.mean(mae_values)
        avg_ssim = np.nanmean(ssim_values)

        # Extract the phase from the filename
        phase = extract_phase(gt_file)
        if phase is None:
            print(f"Filename format error: {gt_file}")
            continue

        phase_metrics[phase]['psnr'].append(avg_psnr)
        phase_metrics[phase]['mae'].append(avg_mae)
        phase_metrics[phase]['ssim'].append(avg_ssim)

    phase_data = []

    for phase, metrics in phase_metrics.items():
        avg_psnr = np.mean(metrics['psnr'])
        stddev_psnr = np.std(metrics['psnr'])
        avg_mae = np.mean(metrics['mae'])
        stddev_mae = np.std(metrics['mae'])
        avg_ssim = np.mean(metrics['ssim'])
        stddev_ssim = np.std(metrics['ssim'])

        print(f'Phase {phase} - Average PSNR: {avg_psnr}, Std Dev: {stddev_psnr}')
        print(f'Phase {phase} - Average MAE: {avg_mae}, Std Dev: {stddev_mae}')
        print(f'Phase {phase} - Average SSIM: {avg_ssim}, Std Dev: {stddev_ssim}')

        phase_data.append([phase, avg_psnr, stddev_psnr, avg_mae, stddev_mae, avg_ssim, stddev_ssim])

    df = pd.DataFrame(phase_data, columns=['Phase', 'Avg_PSNR', 'StdDev_PSNR', 'Avg_MAE', 'StdDev_MAE', 'Avg_SSIM', 'StdDev_SSIM'])
    df.to_csv('BBDM_metrics_per_phase.csv', index=False)

    return df

if __name__ == "__main__":
    ground_truth_dir = 'results/BBDM_base/BrownianBridge/sample_to_eval/ground_truth' #BBDM_n98_s256x256_z88_e10
    generated_dir = 'results/BBDM_topconfigs/BrownianBridge/sample_to_eval/200'

    evaluate_metrics(ground_truth_dir, generated_dir)
