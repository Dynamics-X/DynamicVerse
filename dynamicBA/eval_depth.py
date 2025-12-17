import os
import json
import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from scipy.optimize import minimize

def depth2disparity(depth, return_mask=False):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity
    
def absolute_error_loss(params, predicted_depth, ground_truth_depth):
    s, t = params

    predicted_aligned = s * predicted_depth + t

    abs_error = np.abs(predicted_aligned - ground_truth_depth)
    return np.sum(abs_error)

def absolute_value_scaling(predicted_depth, ground_truth_depth, s=1, t=0):
    predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1)
    ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1)
    
    initial_params = [s, t]  # s = 1, t = 0
    
    result = minimize(absolute_error_loss, initial_params, args=(predicted_depth_np, ground_truth_depth_np))
    
    s, t = result.x  
    return s, t

def absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=1.0, t_init=0.0, lr=1e-4, max_iters=1000, tol=1e-6):
    # Initialize s and t as torch tensors with requires_grad=True
    s = torch.tensor([s_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)
    t = torch.tensor([t_init], requires_grad=True, device=predicted_depth.device, dtype=predicted_depth.dtype)

    optimizer = torch.optim.Adam([s, t], lr=lr)
    
    prev_loss = None

    for i in range(max_iters):
        optimizer.zero_grad()

        # Compute predicted aligned depth
        predicted_aligned = s * predicted_depth + t

        # Compute absolute error
        abs_error = torch.abs(predicted_aligned - ground_truth_depth)

        # Compute loss
        loss = torch.sum(abs_error)

        # Backpropagate
        loss.backward()

        # Update parameters
        optimizer.step()

        # Check convergence
        if prev_loss is not None and torch.abs(prev_loss - loss) < tol:
            break

        prev_loss = loss.item()

    return s.detach().item(), t.detach().item()

def depth_evaluation(predicted_depth_original, ground_truth_depth_original, max_depth=80, custom_mask=None, 
                    post_clip_min=None, post_clip_max=None, pre_clip_min=None, pre_clip_max=None,
                    align_with_lstsq=False, align_with_lad=False, align_with_lad2=False, 
                    lr=1e-4, max_iters=1000, use_gpu=False, align_with_scale=False, disp_input=False):
    """
    Evaluate the depth map using various metrics and return a depth error parity map, with an option for least squares alignment.
    
    Args:
        predicted_depth (numpy.ndarray or torch.Tensor): The predicted depth map.
        ground_truth_depth (numpy.ndarray or torch.Tensor): The ground truth depth map.
        max_depth (float): The maximum depth value to consider. Default is 80 meters.
        align_with_lstsq (bool): If True, perform least squares alignment of the predicted depth with ground truth.
    
    Returns:
        dict: A dictionary containing the evaluation metrics.
        torch.Tensor: The depth error parity map.
    """
    
    if isinstance(predicted_depth_original, np.ndarray):
        predicted_depth_original = torch.from_numpy(predicted_depth_original)
    if isinstance(ground_truth_depth_original, np.ndarray):
        ground_truth_depth_original = torch.from_numpy(ground_truth_depth_original)
    if custom_mask is not None and isinstance(custom_mask, np.ndarray):
        custom_mask = torch.from_numpy(custom_mask)

    # if the dimension is 3, flatten to 2d along the batch dimension
    if predicted_depth_original.dim() == 3:
        _, h, w = predicted_depth_original.shape
        predicted_depth_original = predicted_depth_original.view(-1, w)
        ground_truth_depth_original = ground_truth_depth_original.view(-1, w)
        if custom_mask is not None:
            custom_mask = custom_mask.view(-1, w)

    # put to device
    if use_gpu:
        predicted_depth_original = predicted_depth_original.cuda()
        ground_truth_depth_original = ground_truth_depth_original.cuda()
    
    # Filter out depths greater than max_depth
    if max_depth is not None:
        mask = (ground_truth_depth_original > 0) & (ground_truth_depth_original < max_depth)
    else:
        mask = (ground_truth_depth_original > 0)

    # lower_bound = torch.quantile(predicted_depth, 0.01)
    # upper_bound = torch.quantile(predicted_depth, 0.99)
    
    predicted_depth = predicted_depth_original[mask]
    ground_truth_depth = ground_truth_depth_original[mask]

    # Clip the depth values
    if pre_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=pre_clip_min)
    if pre_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=pre_clip_max)

    if disp_input: # align the pred to gt in the disparity space
        print("Aligning in disp space")
        real_gt = ground_truth_depth.clone()
        ground_truth_depth = 1 / (ground_truth_depth + 1e-8)

    # various alignment methods
    if align_with_lstsq:
        print("Using lstsq!")
        # Convert to numpy for lstsq
        predicted_depth_np = predicted_depth.cpu().numpy().reshape(-1, 1)
        ground_truth_depth_np = ground_truth_depth.cpu().numpy().reshape(-1, 1)
        
        # Add a column of ones for the shift term
        A = np.hstack([predicted_depth_np, np.ones_like(predicted_depth_np)])
        
        # Solve for scale (s) and shift (t) using least squares
        result = np.linalg.lstsq(A, ground_truth_depth_np, rcond=None)
        s, t = result[0][0], result[0][1]

        # convert to torch tensor
        s = torch.tensor(s, device=predicted_depth_original.device)
        t = torch.tensor(t, device=predicted_depth_original.device)
        
        # Apply scale and shift
        predicted_depth = s * predicted_depth + t
    elif align_with_lad:
        s, t = absolute_value_scaling(predicted_depth, ground_truth_depth, s=torch.median(ground_truth_depth) / torch.median(predicted_depth))
        predicted_depth = s * predicted_depth + t
    elif align_with_lad2:
        print("using lad2!")
        s_init = (torch.median(ground_truth_depth) / torch.median(predicted_depth)).item()
        s, t = absolute_value_scaling2(predicted_depth, ground_truth_depth, s_init=s_init, lr=lr, max_iters=max_iters)
        predicted_depth = s * predicted_depth + t
    elif align_with_scale:
        
        # Compute initial scale factor 's' using the closed-form solution (L2 norm)
        dot_pred_gt = torch.nanmean(ground_truth_depth)
        dot_pred_pred = torch.nanmean(predicted_depth)
        s = dot_pred_gt / dot_pred_pred

        # Iterative reweighted least squares using the Weiszfeld method
        for _ in range(10):
            # Compute residuals between scaled predictions and ground truth
            residuals = s * predicted_depth - ground_truth_depth
            abs_residuals = residuals.abs() + 1e-8  # Add small constant to avoid division by zero
            
            # Compute weights inversely proportional to the residuals
            weights = 1.0 / abs_residuals
            
            # Update 's' using weighted sums
            weighted_dot_pred_gt = torch.sum(weights * predicted_depth * ground_truth_depth)
            weighted_dot_pred_pred = torch.sum(weights * predicted_depth ** 2)
            s = weighted_dot_pred_gt / weighted_dot_pred_pred

        # Optionally clip 's' to prevent extreme scaling
        s = s.clamp(min=1e-3)
        
        # Detach 's' if you want to stop gradients from flowing through it
        s = s.detach()
        
        # Apply the scale factor to the predicted depth
        predicted_depth = s * predicted_depth

    else:
        print("using scale!")
        # Align the predicted depth with the ground truth using median scaling
        scale_factor = torch.median(ground_truth_depth) / torch.median(predicted_depth)
        predicted_depth *= scale_factor

    if disp_input:
        # convert back to depth
        ground_truth_depth = real_gt
        predicted_depth = depth2disparity(predicted_depth)

    # Clip the predicted depth values
    if post_clip_min is not None:
        predicted_depth = torch.clamp(predicted_depth, min=post_clip_min)
    if post_clip_max is not None:
        predicted_depth = torch.clamp(predicted_depth, max=post_clip_max)

    if custom_mask is not None:
        assert custom_mask.shape == ground_truth_depth_original.shape
        mask_within_mask = custom_mask.cpu()[mask]
        predicted_depth = predicted_depth[mask_within_mask]
        ground_truth_depth = ground_truth_depth[mask_within_mask]

    # Calculate the metrics
    abs_rel = torch.mean(torch.abs(predicted_depth - ground_truth_depth) / ground_truth_depth).item()
    sq_rel = torch.mean(((predicted_depth - ground_truth_depth) ** 2) / ground_truth_depth).item()
    
    # Correct RMSE calculation
    rmse = torch.sqrt(torch.mean((predicted_depth - ground_truth_depth) ** 2)).item()
    
    # Clip the depth values to avoid log(0)
    predicted_depth = torch.clamp(predicted_depth, min=1e-5)
    
    # Calculate the accuracy thresholds
    max_ratio = torch.maximum(predicted_depth / ground_truth_depth, ground_truth_depth / predicted_depth)
    threshold_1 = torch.mean((max_ratio < 1.25).float()).item()

    # Compute the depth error parity map
    if align_with_lstsq or align_with_lad or align_with_lad2:
        predicted_depth_original = predicted_depth_original * s + t
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    elif align_with_scale:
        predicted_depth_original = predicted_depth_original * s
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    else:
        predicted_depth_original = predicted_depth_original * scale_factor
        if disp_input: predicted_depth_original = depth2disparity(predicted_depth_original)
        depth_error_parity_map = torch.abs(predicted_depth_original - ground_truth_depth_original) / ground_truth_depth_original
    
    # Reshape the depth_error_parity_map back to the original image size
    depth_error_parity_map_full = torch.zeros_like(ground_truth_depth_original)
    depth_error_parity_map_full = torch.where(mask, depth_error_parity_map, depth_error_parity_map_full)

    predict_depth_map_full = predicted_depth_original

    gt_depth_map_full = torch.zeros_like(ground_truth_depth_original)
    gt_depth_map_full = torch.where(mask, ground_truth_depth_original, gt_depth_map_full)

    num_valid_pixels = torch.sum(mask).item() if custom_mask is None else torch.sum(mask_within_mask).item()
    if num_valid_pixels == 0:
        abs_rel, sq_rel, rmse, log_rmse, threshold_1, threshold_2, threshold_3 = 0, 0, 0, 0, 0, 0, 0

    results = np.array([abs_rel, threshold_1, num_valid_pixels])

    return results, depth_error_parity_map_full.detach().cpu().numpy(), predict_depth_map_full.detach().cpu().numpy(), gt_depth_map_full.detach().cpu().numpy(), mask.detach().cpu().numpy()
    
def get_preds(video_path, experiment_name, number_of_frames, w, h):
    """
    Reads in depth predictions
    """
    full_result_dir = os.path.join(video_path, "dynamicBA", experiment_name)
    
    disp_preds = [] 

    # Ensure per-frame files exist; stop early on missing to avoid index errors
    for i in range(number_of_frames):
        # npz_path = os.path.join(full_result_dir, f'{i:04d}.npz')
        # npz_path = os.path.join(full_result_dir, f'{i:04d}_dynamic_flow_fused_wcotracker_noweighted_20epochs_1e-2.npz')
        # npz_path = os.path.join(full_result_dir, f'{i:04d}_before_flow_opt.npz')
        npz_path = os.path.join(full_result_dir, f'{i:04d}_simple_flow_opt_1.0_1e-4.npz')

        if not os.path.exists(npz_path):
            # Stop at first missing file and use available predicted frames
            break

        npz = np.load(npz_path, allow_pickle=True)
        disp = npz["disp"]
        disp = cv2.resize(disp, (w, h), interpolation=cv2.INTER_CUBIC)
        disp = torch.tensor(disp)
        disp_preds.append(disp)

    # If fewer predicted frames than GT, use available frames and warn
    if len(disp_preds) == 0:
        raise FileNotFoundError(f"No prediction files found in {full_result_dir}")
    if len(disp_preds) < number_of_frames:
        print(f"Warning: Only found {len(disp_preds)} prediction frames, fewer than GT frames {number_of_frames}. Using {len(disp_preds)}.")

    disp_preds = np.array(disp_preds)

    return disp_preds
    
def get_gt(video_path, dataset):
    """
    Load ground truth depth maps
    """
    depth_gt = []

    depth_paths = sorted(os.listdir(os.path.join(video_path, "depth_gt")))
    depth_paths = [os.path.join(video_path, "depth_gt", x) for x in depth_paths]

    number_of_frames = len(depth_paths)
    h, w = None, None

    for i in range(number_of_frames):
        if dataset == "sintel":
            depth = np.load(depth_paths[i])
        elif dataset == "kitti":
            img_pil = Image.open(depth_paths[i])
            depth_png = np.array(img_pil, dtype=int)
            assert(np.max(depth_png) > 255)
            depth = depth_png.astype(float) / 256.
        elif dataset == "bonn":
            depth_png = np.asarray(Image.open(depth_paths[i]))
            assert np.max(depth_png) > 255
            depth = depth_png.astype(np.float64) / 5000.0
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        if h is None:
            h, w = depth.shape
                    
        depth_gt.append(depth)

    depth_gt = np.stack(depth_gt, axis=0)

    return depth_gt, number_of_frames, h, w

def eval_depth(dataset="sintel", base_path=None, experiment_name="base", 
               scale_only=False, max_depth=70):
    """Main function to evaluate depth predictions"""
    
    if base_path is None:
        base_path = f"./data/{dataset}/"
    
    videos = sorted(os.listdir(base_path))
    excluded = set()
    
    res_save_path = f"results/depth/{dataset}/{experiment_name}"
    os.makedirs(res_save_path, exist_ok=True)
    
    # Open a txt file to write results for each video with better formatting
    results_file = open(f"{res_save_path}/all_results.txt", "w")
    
    # Column headers with fixed width formatting
    results_file.write(f"{'Video':<15}{'Abs_Rel':<12}{'d1 < 1.25':<12}\n")
    results_file.write("-" * 40 + "\n")  # Separator line

    errs = {
        "total_videos": 0,
        "errors": torch.zeros(2),
        "total_weights": 0
    }
    
    # Add a dict to store per-video weights for later printing
    video_weights = {}
    video_metrics = {}  # Store per-video metrics directly to avoid rereading files

    for i, video in enumerate(videos):
        if video in excluded:
            continue
        
        video_path = os.path.join(base_path, video)
        print(f"Working on {i}:{video}", flush=True)
        
        gt_depths, number_of_frames, h, w = get_gt(video_path, dataset)
        pred_disp = get_preds(video_path, experiment_name, number_of_frames, w, h)
        # If fewer predicted frames than GT, truncate GT to match
        if pred_disp.shape[0] < number_of_frames:
            gt_depths = gt_depths[:pred_disp.shape[0]]
            number_of_frames = pred_disp.shape[0]

        results, _,_,_,_ = depth_evaluation(
            pred_disp, gt_depths, align_with_lstsq=not scale_only, 
            align_with_lad2=False, disp_input=True, max_depth=max_depth, 
            use_gpu=torch.cuda.is_available(), post_clip_max=max_depth
        )
    
        # Write results to txt file with fixed width columns for better alignment
        results_file.write(f"{video:<15}{results[0]:<12.6f}{results[1]:<12.6f}\n")

        # Save per-video weight info and metrics
        video_weights[video] = results[2]  # The third element is the number of valid pixels
        video_metrics[video] = {"abs_rel": results[0], "d1": results[1]}  # 保存指标
        
        errs["errors"] += (torch.tensor(results[:2]) * results[-1])
        errs["total_videos"] += 1
        errs["total_weights"] += results[-1]

    # 先关闭文件
    results_file.close()
        
    total_errors_video = errs["errors"] / errs["total_weights"]
    
    # Print per-video weight information
    print("\nPer-video weight information:")
    print(f"{'Video':<15}{'Abs_Rel':<12}{'d1 < 1.25':<12}{'ValidPixels':<15}{'Weight(%)':<12}")
    print("-" * 65)
    
    # Sort videos by weight (desc) and print
    for video, weight in sorted(video_weights.items(), key=lambda x: x[1], reverse=True):
        if video in video_metrics:
            abs_rel = video_metrics[video]["abs_rel"]
            d1 = video_metrics[video]["d1"]
            weight_ratio = (weight / errs["total_weights"]) * 100
            print(f"{video:<15}{abs_rel:<12.6f}{d1:<12.6f}{weight:<15.0f}{weight_ratio:<12.2f}")
    
    print("\nWeighted results:")
    print(f"Total valid pixels: {errs['total_weights']}")
    print(f"Weighted mean Abs_Rel: {total_errors_video[0].item():.6f}")
    print(f"Weighted mean d1 < 1.25: {total_errors_video[1].item():.6f}")
    
    # 重新打开文件添加总结信息
    with open(f"{res_save_path}/all_results.txt", "a") as results_file:
        results_file.write("\n" + "=" * 40 + "\n")
        results_file.write("SUMMARY\n")
        results_file.write("-" * 40 + "\n")
        results_file.write(f"{'Average Abs_Rel:':<20}{total_errors_video[0].item():.6f}\n")
        results_file.write(f"{'Average d1 < 1.25:':<20}{total_errors_video[1].item():.6f}\n") 
        results_file.write(f"{'Evaluated videos:':<20}{errs['total_videos']}\n")

    final_result = {
        "video: abs_rel": total_errors_video[0].item(),
        "video: d1 < 1.25": total_errors_video[1].item(),
        "total_videos": errs["total_videos"]
    }

    print(json.dumps(final_result, indent=4))

    # Still keep the overall result json for compatibility
    with open(f"{res_save_path}/result.json", "w") as f:
        f.write(json.dumps(final_result, indent=4))
        
    return final_result

def main():
    parser = argparse.ArgumentParser(description="Evaluate depth predictions")
    parser.add_argument("--dataset", type=str, default="sintel", help="Dataset name")
    parser.add_argument("--base_path", type=str, default="./data/sintel", help="Base path to dataset")
    parser.add_argument("--experiment_name", type=str, default="base", help="Experiment name")
    parser.add_argument("--scale_only", action="store_true", help="Use only scale alignment (no shift)")
    parser.add_argument("--max_depth", type=float, default=70, help="Maximum depth value to consider")
    
    args = parser.parse_args()
    
    eval_depth(
        dataset=args.dataset,
        base_path=args.base_path,
        experiment_name=args.experiment_name,
        scale_only=args.scale_only,
        max_depth=args.max_depth
    )

if __name__ == '__main__':
    main()