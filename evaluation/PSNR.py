import cv2
import os
import numpy as np

def calculate_psnr(img1, img2):
    return cv2.PSNR(img1, img2)

def evaluate_psnr(ground_truth_dir, generated_dir):
    ground_truth_files = os.listdir(ground_truth_dir)
    psnr_values = []

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

        generated_files = os.listdir(generated_folder)
        for gen_file in generated_files:
            gen_path = os.path.join(generated_folder, gen_file)
            gen_image = cv2.imread(gen_path)
            if gen_image is None:
                print(f"Failed to load generated image: {gen_path}")
                continue

            psnr = calculate_psnr(gt_image, gen_image)
            psnr_values.append(psnr)
            print(f"PSNR for {gt_file} and {gen_file}: {psnr}")

    average_psnr = np.mean(psnr_values)
    print(f'Average PSNR: {average_psnr}')
    return average_psnr

if __name__ == "__main__":
    ground_truth_dir = 'results/BBDM_n98_s256x256_e10/BrownianBridge/samples_to_eval/ground_truth'
    generated_dir = 'results/BBDM_n98_s256x256_e10/BrownianBridge/samples_to_eval/200'

    evaluate_psnr(ground_truth_dir, generated_dir)
