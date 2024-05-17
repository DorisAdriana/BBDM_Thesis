import cv2
import os
import numpy as np
import pandas as pd
from collections import defaultdict

def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def evaluate_psnr(ground_truth_dir, generated_dir):
    ground_truth_files = sorted(os.listdir(ground_truth_dir))
    psnr_results = []
    subject_psnr = defaultdict(list)

    for gt_file in ground_truth_files:
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

        for gen_file in generated_files:
            gen_path = os.path.join(generated_folder, gen_file)
            gen_image = cv2.imread(gen_path)
            if gen_image is None:
                print(f"Failed to load generated image: {gen_path}")
                continue

            psnr = calculate_psnr(gt_image, gen_image)
            psnr_values.append(psnr)
            print(f"PSNR for {gt_file} and {gen_file}: {psnr}")

        avg_psnr = np.mean(psnr_values)
        print(f'Average PSNR for {gt_file}: {avg_psnr}')
        psnr_results.append((gt_file, psnr_values, avg_psnr))

        # Extract subject ID and store PSNR values
        subject_id = gt_file.split('_')[2]
        subject_psnr[subject_id].extend(psnr_values)

    # Calculate and print averages and standard deviations per subject
    subject_averages = {}
    subject_stddevs = {}

    for subject_id, psnr_values in subject_psnr.items():
        avg_psnr = np.mean(psnr_values)
        stddev_psnr = np.std(psnr_values)
        subject_averages[subject_id] = avg_psnr
        subject_stddevs[subject_id] = stddev_psnr
        print(f'Subject {subject_id} - Average PSNR: {avg_psnr}, Std Dev: {stddev_psnr}')

    # Save results to CSV file
    results = []

    for gt_file, psnr_values, avg_psnr in psnr_results:
        subject_id = gt_file.split('_')[2]
        results.append([gt_file] + psnr_values + [avg_psnr, subject_averages[subject_id], subject_stddevs[subject_id]])

    df = pd.DataFrame(results, columns=['Ground Truth', 'PSNR_0', 'PSNR_1', 'PSNR_2', 'PSNR_3', 'PSNR_4', 'Avg_PSNR', 'Subject_Avg', 'Subject_StdDev'])
    df.to_csv('psnr_results.csv', index=False)

    return psnr_results

if __name__ == "__main__":
    ground_truth_dir = 'results/BBDM_n98_s256x256_e10/BrownianBridge/samples_to_eval/ground_truth'
    generated_dir = 'results/BBDM_n98_s256x256_e10/BrownianBridge/samples_to_eval/200'

    evaluate_psnr(ground_truth_dir, generated_dir)
