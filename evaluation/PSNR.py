import cv2
import os
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from collections import defaultdict
from tqdm import tqdm

def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def calculate_mae(img1, img2):
    return np.mean(np.abs(img1 - img2))

def calculate_ssim(img1, img2):
    return ssim(img1, img2, multichannel=True)

def evaluate_metrics(ground_truth_dir, generated_dir):
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    subject_metrics = defaultdict(lambda: {'psnr': [], 'mae': [], 'ssim': []})

    for gt_file in tqdm(ground_truth_files, desc="Processing ground truth images"):
        gt_path = os.path.join(ground_truth_dir, gt_file)
        gt_image = cv2.imread(gt_path)
        if gt_image is None:
            print(f"Failed to load ground truth image: {gt_path}")
            continue

        # Assuming generated images are in a folder named by the ground truth image (without extension)
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
            ssim_value = calculate_ssim(gt_image, gen_image)

            psnr_values.append(psnr)
            mae_values.append(mae)
            ssim_values.append(ssim_value)

        # Calculate the average metrics for this ground truth image
        avg_psnr = np.mean(psnr_values)
        avg_mae = np.mean(mae_values)
        avg_ssim = np.mean(ssim_values)

        # Extract subject ID and store the average metrics
        subject_id = gt_file.split('_')[2]
        subject_metrics[subject_id]['psnr'].append(avg_psnr)
        subject_metrics[subject_id]['mae'].append(avg_mae)
        subject_metrics[subject_id]['ssim'].append(avg_ssim)

    # Calculate and print averages and standard deviations per subject
    subject_averages = []
    subject_stddevs = []

    for subject_id, metrics in subject_metrics.items():
        avg_psnr = np.mean(metrics['psnr'])
        stddev_psnr = np.std(metrics['psnr'])
        avg_mae = np.mean(metrics['mae'])
        stddev_mae = np.std(metrics['mae'])
        avg_ssim = np.mean(metrics['ssim'])
        stddev_ssim = np.std(metrics['ssim'])

        print(f'Subject {subject_id} - Average PSNR: {avg_psnr}, Std Dev: {stddev_psnr}')
        print(f'Subject {subject_id} - Average MAE: {avg_mae}, Std Dev: {stddev_mae}')
        print(f'Subject {subject_id} - Average SSIM: {avg_ssim}, Std Dev: {stddev_ssim}')

        subject_averages.append([subject_id, avg_psnr, avg_mae, avg_ssim])
        subject_stddevs.append([subject_id, stddev_psnr, stddev_mae, stddev_ssim])

    # Save results to CSV file
    df_avg = pd.DataFrame(subject_averages, columns=['Subject', 'Avg_PSNR', 'Avg_MAE', 'Avg_SSIM'])
    df_stddev = pd.DataFrame(subject_stddevs, columns=['Subject', 'StdDev_PSNR', 'StdDev_MAE', 'StdDev_SSIM'])

    df_avg.to_csv('average_metrics_per_subject.csv', index=False)
    df_stddev.to_csv('stddev_metrics_per_subject.csv', index=False)

    return subject_averages, subject_stddevs

if __name__ == "__main__":
    ground_truth_dir = 'results/BBDM_n98_s256x256_e10/BrownianBridge/samples_to_eval/ground_truth'
    generated_dir = 'results/BBDM_n98_s256x256_e10/BrownianBridge/samples_to_eval/200'

    evaluate_metrics(ground_truth_dir, generated_dir)
