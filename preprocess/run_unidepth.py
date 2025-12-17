import torch
import imageio
import numpy as np
import glob
import os
from unidepth.models import UniDepthV2, UniDepthV2old
from unidepth.utils.camera import Pinhole, BatchCamera
import argparse
from tqdm import tqdm
import time
import cv2

parser = argparse.ArgumentParser(description='Running unidepth')
parser.add_argument('--workdir',
                    metavar='DIR',
                    help='path to dataset',
                    default="./data_preprocessed/sintel_test")
parser.add_argument('--v2',
                    action='store_true',
                    help='use UniDepthV2')
parser.add_argument('--use_gt_K',
                    action='store_true',
                    help='use ground truth intrinsics')

args = parser.parse_args()
BASE = args.workdir

videos = sorted([x for x in sorted(os.listdir(BASE))])

if args.v2:
    model = UniDepthV2.from_pretrained("./preprocess/pretrained/unidepth_model")
    # model = UniDepthV2.from_pretrained("lpiccinelli/unidepth-v2-vitl14")
    print("using v2")
else:
    model = UniDepthV2old.from_pretrained(f"./preprocess/pretrained/unidepth_model")
    # model = UniDepthV2old.from_pretrained(f"lpiccinelli/unidepth-v2old-vitl14")

# Move to CUDA, if any
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

start_time = time.time()

for i, video in enumerate(tqdm(videos, desc="Processing Unidepth")): # Use enumerate and tqdm

    # === Added: Check if output directory and PNG files exist ===
    OUTPUT_BASE = os.path.join(BASE, video, "unidepth")
    if args.use_gt_K:
        OUTPUT_BASE += "_gt_K"

    png_files_exist = False
    if os.path.isdir(OUTPUT_BASE):
        # Check 1: Does depth.npy exist? (Use os.path.exists for more direct check)
        has_npy = os.path.exists(os.path.join(OUTPUT_BASE, "depth.npy"))
        
        has_png = False
        # Check 2: Only continue checking png if npy exists
        if has_npy:
            try:
                # Traverse all files in the directory
                for filename in os.listdir(OUTPUT_BASE):
                    if filename.lower().endswith(".png"):
                        has_png = True
                        break # Stop immediately after finding the first .png file, no need to continue traversing
            except OSError:
                pass # If directory reading fails, has_png remains False

        # Comprehensive judgment: Both must be True
        if has_npy and has_png:
            png_files_exist = True

    if png_files_exist:
        print(f"\n⏭️  Scene '{video}' already processed, skipping (output directory '{OUTPUT_BASE}' exists and contains depth.npy file)")
        continue  # Skip current loop, process next video
    # === Check end ===

    # Only continue if processing is needed
    print(f"\n🔄 ({i + 1}/{len(videos)}) Processing scene: {video}")

    # Ensure output directory exists (created if it didn't exist above)
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # 2. Load images
    rgb_path = os.path.join(BASE, video, "rgb")
    if not os.path.isdir(rgb_path):
            print(f"⚠️  Skipping scene '{video}': 'rgb' subdirectory not found.")
            continue

    print(f"   Loading frames: {rgb_path}")
    image_paths = sorted(glob.glob(os.path.join(rgb_path, "*.png")) + glob.glob(os.path.join(rgb_path, "*.jpg")))

    if not image_paths:
        print(f"⚠️  Skipping scene '{video}': No PNG or JPG image files found in 'rgb' directory.")
        continue

    images = []
    for image_path in image_paths:
        try:
            im = imageio.imread(image_path)
            # Check image channel count, ensure RGB
            if im.ndim == 2: # Grayscale
                print(f"   Warning: Image {os.path.basename(image_path)} is grayscale, converting to RGB.")
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            elif im.ndim == 3 and im.shape[2] == 4: # RGBA
                print(f"   Warning: Image {os.path.basename(image_path)} is RGBA, converting to RGB.")
                im = cv2.cvtColor(im, cv2.COLOR_RGBA2RGB)
            elif im.ndim != 3 or im.shape[2] != 3:
                    print(f"   Warning: Image {os.path.basename(image_path)} has abnormal channel count ({im.shape}), skipping this frame.")
                    continue

            images.append(im)
        except Exception as e:
                print(f"   Warning: Failed to read image {image_path}: {e}, skipping this frame.")
                continue

    if not images:
        print(f"❌  Processing scene '{video}' failed: Could not successfully load any valid RGB frames.")
        continue

    try: # Add try-except block for potential errors in subsequent calculations
        images = torch.tensor(np.array(images)) # F x H x W x 3
        f, h, w, c = images.shape
        batch_size = 8 # Can be set as a parameter
        print(f"   Successfully loaded {f} frames, dimensions: {w}x{h}")

        # 3. Calculate or load camera intrinsics K
        K = None # Initialize K
        intrinsics_save_path = os.path.join(OUTPUT_BASE, 'intrinsics.npy')

        if not args.use_gt_K:
            print("   Calculating camera intrinsics...")
            with torch.no_grad():
                frame_range = list(range(f))
                chuncks = [frame_range[i:i+batch_size] for i in range(0, len(frame_range), batch_size)]
                initial_intrinsics = []
                for chunk in tqdm(chuncks, desc="     Calculating intrinsics batch", leave=False):
                    imgs_batch = images[chunk, ...].to(device) # Move to GPU
                    imgs_batch = torch.permute(imgs_batch, (0, 3, 1, 2)).float() / 255.0 # BxCxHxW, normalize
                    preds = model.infer(imgs_batch) # B x 4 (fx, fy, cx, cy) or B x 3x3
                    initial_intrinsics.append(preds['intrinsics']) # Get intrinsics

                if not initial_intrinsics:
                        raise ValueError("Could not calculate any intrinsics")

                initial_intrinsics = torch.cat(initial_intrinsics, dim=0)

                # Handle different output formats for K (B x 4 or B x 3x3)
                if initial_intrinsics.ndim == 2 and initial_intrinsics.shape[1] == 4: # B x 4
                    print("     Detected intrinsics format as B x 4 (fx, fy, cx, cy), converting to 3x3 matrix")
                    # Take average
                    mean_intrinsics_vec = torch.mean(initial_intrinsics, dim=0)
                    fx, fy, cx_offset, cy_offset = mean_intrinsics_vec
                    # Build K matrix
                    K = torch.zeros((3, 3), dtype=torch.float32)
                    K[0, 0] = fx * w # Multiply by width
                    K[1, 1] = fy * h # Multiply by height
                    K[0, 2] = w / 2 + cx_offset * w # Center point + offset
                    K[1, 2] = h / 2 + cy_offset * h # Center point + offset
                    K[2, 2] = 1.0
                elif initial_intrinsics.ndim == 3 and initial_intrinsics.shape[1:] == (3, 3): # B x 3x3
                    print("     Detected intrinsics format as B x 3x3")
                    K = torch.mean(initial_intrinsics, dim=0) # Take average matrix directly
                else:
                    raise ValueError(f"Unrecognized intrinsics output shape: {initial_intrinsics.shape}")

                K = K.cpu() # Move back to CPU for saving
                np.save(intrinsics_save_path, K.numpy())
                print(f"   Camera intrinsics calculated and saved: {intrinsics_save_path}")

        else:
            k_path = os.path.join(BASE, video, "K.npy")
            print(f"   Loading GT camera intrinsics: {k_path}")
            if not os.path.exists(k_path):
                print(f"❌ Error: GT intrinsics file {k_path} not found, but --use_gt_K specified.")
                continue
            K = torch.from_numpy(np.load(k_path)).float()
            # GT K is usually a single 3x3 matrix, no averaging needed
            np.save(intrinsics_save_path, K.numpy()) # Save a copy to output directory
            print(f"   GT camera intrinsics loaded and copied to: {intrinsics_save_path}")

        # Ensure K is a 3x3 matrix
        if K is None or K.shape != (3, 3):
                print(f"❌ Error: Could not obtain correct 3x3 camera intrinsics matrix.")
                continue

        # 5. Calculate and save depth maps
        print("   Calculating depth maps...")
        depth_save_path_disp = os.path.join(OUTPUT_BASE, 'depth.npy') # Path defined earlier

        # Extend K to all frames and move to GPU
        K_batch = K.unsqueeze(0).repeat(f, 1, 1).to(device)

        depths = []
        with torch.no_grad():
            frame_range = list(range(f))
            chuncks = [frame_range[i:i+batch_size] for i in range(0, len(frame_range), batch_size)]
            for chunk in tqdm(chuncks, desc="     Calculating depth batch", leave=False):
                imgs_batch = images[chunk, ...].to(device)
                Ks_batch = K_batch[chunk, ...]

                # Prepare Camera object for V2
                if isinstance(model, (UniDepthV2)):
                    # Pinhole needs K matrix list or (B, 3, 3) tensor
                    Ks_for_pinhole = Ks_batch # Already (B, 3, 3)
                    Ks_cam = Pinhole(K=Ks_for_pinhole)
                    # BatchCamera.from_camera no longer needed, Pinhole accepts tensor directly
                    # cam = BatchCamera.from_camera(Ks_cam) # This step may not be needed anymore
                    Ks_input = Ks_cam # Pass Pinhole object directly
                else:
                    Ks_input = Ks_batch # V2old may accept (B, 3, 3) tensor directly

                imgs_batch = torch.permute(imgs_batch, (0, 3, 1, 2)).float() / 255.0 # BxCxHxW, normalize
                preds = model.infer(imgs_batch, Ks_input)
                depth = preds['depth'] # B x 1 x H x W
                depths.append(depth)

        if not depths:
                raise ValueError("Could not calculate any depth maps")

        depths = torch.cat(depths, dim=0) # F x 1 x H x W
        depths_np = depths.cpu().numpy() # Move back to CPU for saving
        np.save(depth_save_path_disp, depths_np)
        print(f"   Depth map data saved: {depth_save_path_disp}")

        # 6. Save depth map visualizations
        print("   Generating and saving depth map visualizations...")
        depth_vis = depths_np[:,0,:,:] # F x H x W
        disp_vis = 1.0 / (depth_vis + 1e-12) # Calculate disparity

        # Normalize and color each frame individually to avoid detail loss from overall normalization
        for frame_idx in tqdm(range(len(disp_vis)), desc="     Saving visualizations", leave=False):
            disp_single = disp_vis[frame_idx]
            # Normalize single frame to avoid detail loss
            disp_normalized = cv2.normalize(disp_single, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            disp_color = cv2.applyColorMap(disp_normalized, cv2.COLORMAP_JET)
            # Save using imageio for consistency
            imageio.imwrite(os.path.join(OUTPUT_BASE, f"{frame_idx:04d}.png"), disp_color)

        print(f"   Depth map visualizations saved to: {OUTPUT_BASE}")
        print(f"✅  Scene '{video}' processing completed.")

    except RuntimeError as e: # Catch PyTorch runtime errors (e.g., out of memory)
            print(f"❌  Runtime error processing scene '{video}': {e}")
            continue # Skip to next scene
    except Exception as e: # Catch all other unexpected errors
            print(f"❌  Unknown error processing scene '{video}': {e}")
            import traceback
            traceback.print_exc() # Print detailed error stack
            continue # Skip to next scene