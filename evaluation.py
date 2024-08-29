import os
import numpy as np
import nibabel as nib
import pandas as pd
from medpy.metric.binary import hd, dc  # hd is Hausdorff distance, dc is Dice coefficient

def compute_metrics(vol1, vol2):
    """
    Compute both the Dice score and the Hausdorff Distance between two binary volumes.
    vol1 and vol2 are numpy arrays.
    """
    dice_score = dc(vol1, vol2)
    hausdorff_distance = hd(vol1, vol2)
    
    return dice_score, hausdorff_distance

def main(pred_folder, gt_folder, output_excel):
    pred_folder = os.path.expanduser(pred_folder)
    gt_folder = os.path.expanduser(gt_folder)

    metrics_list = []  # List to store all metrics for Excel export
    
    # Dictionaries to store metrics for calculating averages
    metrics_by_scan = {}
    metrics_by_phase = {}
    
    for filename in sorted(os.listdir(pred_folder)):
        if filename == 'scan_239_b01.nii.gz':
            break
        if filename.endswith('nii.gz'):
            pred_path = os.path.join(pred_folder, filename)
            gt_path = os.path.join(gt_folder, filename)
            
            pred_vol = nib.load(pred_path).get_fdata()
            gt_vol = nib.load(gt_path).get_fdata()
            
            dice_score, hausdorff_distance = compute_metrics(pred_vol > 0, gt_vol > 0)
            
            base_name = filename.rsplit('b', 1)[0]
            phase = filename.split('b')[-1].split('.')[0]  # Extract phase
            
            metrics_list.append({
                'Filename': filename,
                'Base Name': base_name,
                'Phase': phase,
                'Dice Score': dice_score,
                'Hausdorff Distance': hausdorff_distance
            })
            
            # Accumulate metrics for averages
            if base_name not in metrics_by_scan:
                metrics_by_scan[base_name] = {'dice': [], 'hausdorff': []}
            metrics_by_scan[base_name]['dice'].append(dice_score)
            metrics_by_scan[base_name]['hausdorff'].append(hausdorff_distance)
            
            if phase not in metrics_by_phase:
                metrics_by_phase[phase] = {'dice': [], 'hausdorff': []}
            metrics_by_phase[phase]['dice'].append(dice_score)
            metrics_by_phase[phase]['hausdorff'].append(hausdorff_distance)
    
    # Calculate averages
    averages_by_scan = [{
        'Base Name': k,
        'Average Dice': np.mean(v['dice']),
        'Average Hausdorff': np.mean(v['hausdorff'])
    } for k, v in metrics_by_scan.items()]
    
    averages_by_phase = [{
        'Phase': k,
        'Average Dice': np.mean(v['dice']),
        'Average Hausdorff': np.mean(v['hausdorff'])
    } for k, v in metrics_by_phase.items()]
    
    # Convert to DataFrames
    metrics_df = pd.DataFrame(metrics_list)
    averages_scan_df = pd.DataFrame(averages_by_scan)
    averages_phase_df = pd.DataFrame(averages_by_phase)
    
    # Sort DataFrames
    metrics_df.sort_values(by=['Base Name', 'Phase'], inplace=True)
    averages_scan_df.sort_values(by='Base Name', inplace=True)
    averages_phase_df.sort_values(by='Phase', inplace=True)
    
    # Save to Excel
    with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
        metrics_df.to_excel(writer, index=False, sheet_name='Metrics')
        averages_scan_df.to_excel(writer, index=False, sheet_name='Averages by Scan')
        averages_phase_df.to_excel(writer, index=False, sheet_name='Averages by Phase')
    
    print(f"Saved all metrics to {output_excel}")

if __name__ == "__main__":
    pred_folder = "~/my-scratch/outputs/pred_BBDM_mvar"
    gt_folder = "~/my-scratch/outputs/gt"
    output_excel = "computed_metrics_mvar.xlsx"  # Excel file to store the results
    main(pred_folder, gt_folder, output_excel)
