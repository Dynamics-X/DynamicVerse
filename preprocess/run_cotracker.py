import torch
import os
# add to path
import sys
import imageio
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
import argparse
from tqdm import tqdm

sys.path.insert(0,'./co-tracker')
from cotracker.predictor import CoTrackerPredictor 


def get_frames(path):
    
    frames = []
    files = sorted([x for x in os.listdir(path) if x.endswith(".png") or x.endswith(".jpg")])
    for file in files:
        # res = np.load(os.path.join(path, file))
        frames.append(imageio.imread(os.path.join(path, file)))
    frames = np.array(frames) # F x H x W x C

    return frames

def vis(pred_tracks, pred_visibility, video, output_dir):
    """
    Visualize tracklets, highlighting neighbors as red and others as blue if neighbors is not None
    """

    pred_tracks = pred_tracks              # F x N x 2
    pred_visibility = pred_visibility      # F x N
    video = np.array(video.cpu())[0].astype(np.uint8)       # F x C x H x W
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get video dimensions
    frame_height, frame_width = video.shape[2], video.shape[3]
    
    # Determine how many tracklets to sample (1% of total)
    num_tracklets = pred_tracks.shape[1]
    sample_size = max(1, int(num_tracklets * 0.01))  # Ensure at least 1 tracklet
    
    # Randomly sample tracklet indices (1% of all tracklets)
    sampled_indices = random.sample(range(num_tracklets), sample_size)
    
    # Generate a unique color for each sampled tracklet (consistent across frames)
    tracklet_colors = np.random.rand(sample_size, 3)  # RGB colors for each tracklet
    
    # Process each frame
    num_frames = video.shape[0]
    for frame_idx in range(num_frames):
        # Get current frame and convert from CxHxW to HxWxC
        frame = video[frame_idx].transpose(1, 2, 0)
        
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(frame)
        
        # Get coordinates of all sampled points for this frame
        x_coords = pred_tracks[frame_idx, sampled_indices, 0]
        y_coords = pred_tracks[frame_idx, sampled_indices, 1]
        visibility = pred_visibility[frame_idx, sampled_indices]
        
        # Create in-bounds mask to filter out points that go outside the frame
        in_bounds_mask = (
            (x_coords >= 0) & 
            (x_coords < frame_width) & 
            (y_coords >= 0) & 
            (y_coords < frame_height)
        )
        
        # Combine visibility and bounds check
        visible_mask = visibility > 0
        visible_and_in_bounds = visible_mask & in_bounds_mask
        invisible_and_in_bounds = (~visible_mask) & in_bounds_mask
        
        # Plot visible points (filled markers) that are in bounds
        if np.any(visible_and_in_bounds):
            indices_to_plot = np.where(visible_and_in_bounds)[0]
            ax.scatter(
                x_coords[indices_to_plot], 
                y_coords[indices_to_plot],
                c=tracklet_colors[indices_to_plot],
                s=32,  # size
                alpha=1.0
            )
        
        # Plot invisible points (hollow markers) that are in bounds
        if np.any(invisible_and_in_bounds):
            indices_to_plot = np.where(invisible_and_in_bounds)[0]
            for i in indices_to_plot:
                ax.scatter(
                    x_coords[i], 
                    y_coords[i],
                    facecolors='none',
                    edgecolors=[tracklet_colors[i]],  # Same color as when visible
                    s=32,  # size
                    linewidths=2
                )
        
        # Add a legend or text indicating these are 1% of tracklets
        ax.text(10, 20, f"Showing {sample_size}/{num_tracklets} tracklets (1%)", 
                color='white', backgroundcolor='black', fontsize=10)
        
        # Remove axis ticks and labels
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the 
        os.makedirs(os.path.join(output_dir, 'vis'), exist_ok=True)
        output_path = os.path.join(output_dir, 'vis', f'frame_{frame_idx:04d}.jpg')
        plt.savefig(output_path, bbox_inches='tight', dpi=100)
        plt.close()
    
    print(f"Visualization complete. {num_frames} frames saved to {output_dir}")
    print(f"Displayed {sample_size} tracklets out of {num_tracklets} total (1%)")

parser = argparse.ArgumentParser(description='Running cotrackers')
parser.add_argument('--workdir',
                    metavar='DIR',
                    help='path to dataset',
                    default="sintel")
parser.add_argument('--interval',
                    help='interval for cotracker',
                    default=10,
                    type=int)
parser.add_argument('--grid_size',
                    help='grid size for cotracker',
                    default=50,
                    type=int)

args = parser.parse_args()
BASE = args.workdir

INTERVAL = args.interval
GRID_SIZE = args.grid_size

videos = sorted(os.listdir(BASE))
if not videos:
    print(f"Warning: No scene subdirectories found in directory '{BASE}'.")
    sys.exit(0) # Normal exit, as there is nothing to process

print(f"Found {len(videos)} scenes in '{BASE}' directory, starting processing...")

# --- Model initialization moved outside the loop for efficiency ---
print("Initializing CoTracker model...")
model = CoTrackerPredictor(
    checkpoint=os.path.join(
        './preprocess/pretrained/scaled_offline.pth' # Ensure this path is correct
    )
)
if torch.cuda.is_available():
    print("CUDA detected, moving model to GPU.")
    model = model.cuda()
else:
    print("Warning: CUDA not detected, running model on CPU (may be very slow).")
print("Model initialization completed.")
# --- Model initialization end ---
for idx, name in enumerate(tqdm(videos, desc="Processing scenes")): # Use enumerate and tqdm

        # === Added: Check if output file exists ===
        # Build expected output directory name based on current INTERVAL and GRID_SIZE parameters
        cotracker_output_dir_name = f"cotrackerv3_{INTERVAL}_{GRID_SIZE}"
        feature_track_path = os.path.join(BASE, name, cotracker_output_dir_name)
        results_file_path = os.path.join(feature_track_path, "results.npz")

        if os.path.exists(results_file_path):
            print(f"\n⏭️  Scene '{name}' already processed, skipping (results file exists: {results_file_path})")
            continue  # Skip current loop, process next video
        # === Check end ===

        print(f"\n🔄 ({idx + 1}/{len(videos)}) Processing scene: {name}")

        # 1. Load video frames
        rgb_path = os.path.join(BASE, name, "rgb")
        if not os.path.isdir(rgb_path):
             print(f"⚠️  Skipping scene '{name}': 'rgb' subdirectory not found.")
             continue # Skip to next scene

        print(f"   Loading frames: {rgb_path}")
        video_frames_np = get_frames(rgb_path)

        # Check if get_frames loaded successfully
        if video_frames_np is None or video_frames_np.size == 0:
            print(f"❌  Processing scene '{name}' failed: Could not load any valid frames from 'rgb' directory.")
            continue # Skip to next scene

        video = torch.from_numpy(video_frames_np).permute(0, 3, 1, 2)[None].float() # B(=1) x F x C x H x W

        # Move to GPU (if available)
        if torch.cuda.is_available():
            video = video.cuda()

        B, F, C, H, W = video.shape # Get video dimensions

        # 3. Determine query frames
        QUERY_FRAMES = [x for x in range(0, F, INTERVAL)]
        if not QUERY_FRAMES: # At least one query frame is needed
             print(f"⚠️  Skipping scene '{name}': Frame count ({F}) too low to select query frames based on interval ({INTERVAL}).")
             continue

        print(f"   Query frame indices (interval {INTERVAL}): {QUERY_FRAMES}")

        # 4. Initialize result storage
        all_tracks = np.zeros((F, 0, 2))
        all_visibilities = np.zeros((F, 0))
        all_confidences = np.zeros((F, 0))
        all_init_frames = []

        # 5. Loop to perform CoTracker inference
        print(f"   Performing CoTracker inference (grid size: {GRID_SIZE})...")
        try: # Add try-except for errors during inference
            with torch.no_grad(): # No need to compute gradients during inference
                for f in tqdm(QUERY_FRAMES, desc=f"   Query frames ({name})", leave=False):

                    pred_tracks, pred_visibility, pred_confidence = model(
                        video, # Pass B x F x C x H x W
                        grid_size=GRID_SIZE,
                        grid_query_frame=f,
                        backward_tracking=True
                        )

                    # Ensure prediction result dimensions are correct B x F x N x 2/1
                    if pred_tracks.ndim != 4 or pred_visibility.ndim != 3 or pred_confidence.ndim != 3:
                         print(f"❌  Error processing query frame {f} for scene '{name}': Model output dimensions incorrect.")
                         # Can choose to skip this query frame or mark entire scene as failed
                         continue # Here, choose to skip this query frame

                    # Extract the first (and only) result from the batch
                    frame_tracks = pred_tracks[0].cpu().numpy() # F x N x 2
                    frame_visibilities = pred_visibility[0].cpu().numpy() # F x N
                    frame_confidences = pred_confidence[0].cpu().numpy() # F x N

                    # Check if any valid track points were generated
                    if frame_tracks.shape[1] == 0:
                         print(f"⚠️  Query frame {f} for scene '{name}' generated no track points.")
                         continue # Skip this empty query frame result

                    init_frames = np.repeat(f, frame_tracks.shape[1])

                    all_tracks = np.concatenate((all_tracks, frame_tracks), axis=1)
                    all_visibilities = np.concatenate((all_visibilities, frame_visibilities), axis=1)
                    all_confidences = np.concatenate((all_confidences, frame_confidences), axis=1)
                    all_init_frames.extend(init_frames)

            # Check if any track points were collected
            if all_tracks.shape[1] == 0:
                print(f"❌  Processing scene '{name}' failed: All query frames generated no valid track points.")
                continue # Skip to next scene

            print(f"   Inference completed, obtained {all_tracks.shape[1]} track points.")

            # 6. Create output directory (if it doesn't exist)
            os.makedirs(feature_track_path, exist_ok=True)

            # 7. Save results
            print(f"   Saving results to: {results_file_path}")
            np.savez(
                results_file_path,
                all_confidences=all_confidences,
                all_tracks=all_tracks,
                all_visibilities=all_visibilities,
                init_frames=np.array(all_init_frames), # Convert to NumPy array
                orig_shape=np.array(video.shape[-2:]) # H, W
            )

            # 8. Visualization
            print("   Starting visualization...")
            # Pass the original video tensor for visualization
            vis(all_tracks, all_visibilities, video, feature_track_path)
            print(f"✅  Scene '{name}' processing completed.")

        except RuntimeError as e: # Catch PyTorch runtime errors (e.g., out of memory)
             print(f"❌  Runtime error processing scene '{name}': {e}")
             # Can choose to delete partially created files or leave them
             continue # Skip to next scene
        except Exception as e: # Catch all other unexpected errors
             print(f"❌  Unknown error processing scene '{name}': {e}")
             import traceback
             traceback.print_exc() # Print detailed error stack
             continue # Skip to next scene

print("\n🎉 All scenes processing completed.")