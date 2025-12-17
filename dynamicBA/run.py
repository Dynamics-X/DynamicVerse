import configargparse
import os
import torch
import logging
import argparse
import json
from importlib import reload
import time
import numpy as np

from engine import Engine

def parse_args(input_string=None):

    parser = configargparse.ArgParser()

    parser.add_argument('--config', is_config_file=True, default="./config/config_static.yaml", help='config file path')
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--workdir', type=str, default='workdir', help='workdir')
    parser.add_argument('--intrinsics_lr', type=float, default=1e-3, help='intrinsics learning rate')
    parser.add_argument('--cp_translation_dyn_lr', type=float, default=1e-3, help='dyn points translation learning rate')
    parser.add_argument('--uncertainty_lr', type=float, default=1e-4, help='uncertainty learning rate')
    parser.add_argument('--ba_lr', type=float, default=1e-2, help='bundle adjustment learning rate')
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--num_init_epochs', type=int, default=100, help='number of init epochs')
    parser.add_argument('--num_BA_epochs', type=int, default=100, help='number of BA epochs')
    parser.add_argument('--num_dyn_epochs', type=int, default=100, help='number of dyn epochs')
    parser.add_argument('--num_flow_epochs', type=int, default=100, help='number of flow optimization epochs')
    parser.add_argument('--experiment_name', type=str, help='experiment name')
    parser.add_argument('--reproj_weight', type=float, help='weight for reproj error')
    parser.add_argument('--flow_weight', type=float, default=1.0, help='weight for flow loss')
    parser.add_argument('--pose_smooth_weight_t', type=float, help='weight for pose smoothness for t')
    parser.add_argument('--pose_smooth_weight_r', type=float, help='weight for pose smoothness for R')
    parser.add_argument('--dyn_smooth_weight_t', type=float, help='dyn smoothness weight')
    parser.add_argument('--dyn_laplacian_weight_t', type=float, help='dyn laplacian weight')
    parser.add_argument('--log', action='store_true', default=False, help='log to file or print to console')
    parser.add_argument('--opt_intrinsics', action='store_true', default=False, help='optimize intrinsics')
    parser.add_argument('--vis_4d', action='store_true', default=False, help='vis 4d')
    parser.add_argument('--depth_dir', type=str, default="unidepth", help='dir where predicted depth is stored')
    parser.add_argument('--deva_dir', type=str, default="deva", help='dir where deva is stored')
    parser.add_argument('--cotracker_path', type=str, default="cotracker", help='cotracker type')
    parser.add_argument('--dyn_mask_dir', type=str, help='dir where dynamic mask is stored')
    parser.add_argument('--mask_name', type=str, help='annotations name')
    parser.add_argument('--video', type=str, help='which video to work on')
    parser.add_argument('--loss_fn', type=str, help='which loss function to use')
    parser.add_argument('--print_every', type=int, default=100, help='print every')
    parser.add_argument('--deterministic', action='store_true', help='whether to use deterministic mode for consistent results')
    parser.add_argument('--seed', type=int, default=42, help='seed num')
    parser.add_argument('--debug_mode', action='store_true', help='Enable debug mode')
    parser.add_argument('--save_intermediate', action='store_true', default=False, help='Save intermediate state')
    parser.add_argument('--load_from_intermediate', action='store_true', help='Load from intermediate state')
    parser.add_argument('--flow_model', type=str, default='unimatch', help='Optical flow model type (raft/unimatch)')
    parser.add_argument('--static_points_only', action='store_true', help='Only optimize static points, not camera parameters and dynamic points')
    # [New] Add maximum frame limit parameter
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of image frames to process (default: no limit)')
    # --- Add flags for dense flow ablation ---
    parser.add_argument('--flow_opt_depth', action='store_true', default=False, help='Optimize depth map during dense flow phase')
    parser.add_argument('--flow_opt_pose', action='store_true', default=False, help='Optimize camera pose during dense flow phase')
    parser.add_argument('--flow_opt_intrinsics', action='store_true', default=False, help='Optimize camera intrinsics during dense flow phase (requires --opt_intrinsics)')
    parser.add_argument('--suffix', type=str, default='', help='suffix for the output file')
    parser.add_argument('--flow_lr', type=float, default=1e-3, help='Learning rate specifically for the dense flow optimization phase')
    parser.add_argument('--flow_loss_sample_fraction', type=float, default=1.0, help='Fraction of pixels to sample for dense flow loss (0.1 to 1.0). Default 1.0 (no sampling).')
    parser.add_argument('--depth_flow_weight_ratio', type=float, default=1.0, help='Ratio of depth flow loss to pose flow loss (0.1 to 1.0). Default 1.0 (equal weight).')
    parser.add_argument('--exclude_epi_from_flow', default=False, action='store_true', help='exclude epipolar constraint from flow')
    parser.add_argument('--force_regenerate_deva_masks', default=False, action='store_true', help='force regenerate deva masks')
    parser.add_argument('--separate_flow_ba_loss', action='store_true', help='Separate flow loss and BA dense loss for ablation.')
    parser.add_argument('--ba_loss_weight', type=float, default=1.0, help='Weight for the BA dense loss component when combined or separated.')
    parser.add_argument('--disable_flow_loss_calculation', default=False, action='store_true', help='Disable flow loss calculation.')
    parser.add_argument('--skip_flow_loss_calculation', default=False, action='store_true', help='Skip flow loss calculation.')
    parser.add_argument('--use_cotracker_points_in_flow', default=False, action='store_true', help='Use cotracker points in flow optimization instead of excluding them.')
    parser.add_argument('--use_weighted_flow', default=True, action='store_true', help='Use weighted flow optimization with consistency checks (default).')
    parser.add_argument('--no_weighted_flow', default=False, action='store_true', help='Disable weighted flow optimization, use simple unweighted flow calculation.')
    
    # --- Add missing flow interpolation and optimization parameters ---
    parser.add_argument('--allow_flow_interpolation', action='store_true', default=True, help='Allow flow interpolation')
    parser.add_argument('--exp_sigma', type=float, default=0.02, help='Exponential sigma value')
    parser.add_argument('--flow_consistency_cutoff', type=float, default=0.1, help='Flow consistency cutoff threshold')
    parser.add_argument('--min_weight', type=float, default=0.05, help='Minimum weight value')
    parser.add_argument('--normalize_weights', action='store_true', default=True, help='Normalize weights')
    parser.add_argument('--weighted_average', action='store_true', default=True, help='Use weighted average')
    parser.add_argument('--forward_flow_weight', type=float, default=0.5, help='Forward flow weight')
    parser.add_argument('--flow_sample_ratio', type=float, default=0.01, help='Flow sample ratio')
    parser.add_argument('--depth_preservation_weight', type=float, default=500.0, help='Depth preservation weight')
    parser.add_argument('--depth_smoothness_weight', type=float, default=0.1, help='Depth smoothness weight')
    parser.add_argument('--depth_stability_weight', type=float, default=200.0, help='Depth stability weight')
    parser.add_argument('--max_depth_change_ratio', type=float, default=0.01, help='Maximum depth change ratio')
    parser.add_argument('--max_grad_norm', type=float, default=0.1, help='Maximum gradient norm')
    parser.add_argument('--use_huber_loss', action='store_true', default=True, help='Use Huber loss')
    parser.add_argument('--huber_delta', type=float, default=1.0, help='Huber loss delta parameter')
    parser.add_argument('--weight_by_flow_magnitude', action='store_true', default=True, help='Weight by flow magnitude')
    parser.add_argument('--flow_magnitude_scale', type=float, default=15.0, help='Flow magnitude scale')
    parser.add_argument('--conservative_flow_threshold', type=float, default=0.01, help='Conservative flow threshold')
    parser.add_argument('--enable_flow_diagnosis', action='store_true', default=True, help='Enable flow diagnosis')
    parser.add_argument('--save_depth_changes', action='store_true', default=True, help='Save depth changes')
    parser.add_argument('--flow_optimization_analysis', action='store_true', default=True, help='Enable flow optimization analysis')
    
    # === New: Forward-backward wrap weight parameters ===
    parser.add_argument('--cycle_consistency_sigma', type=float, default=1.0, help='Gaussian weight standard deviation, controls weight decay speed')
    parser.add_argument('--min_cycle_weight', type=float, default=0.01, help='Minimum weight, prevents weight from being completely zero')
    parser.add_argument('--max_cycle_error', type=float, default=5.0, help='Maximum cycle error threshold (pixels)')
    parser.add_argument('--flow_quality_threshold', type=float, default=2.0, help='Optical flow quality threshold for filtering high-quality regions')
    
    # === New: Debug and optimization control parameters ===
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Early stopping patience parameter')
    parser.add_argument('--enable_mask_change_detection', action='store_true', default=False, help='Enable optimization mask change detection')
    parser.add_argument('--detailed_epoch_logging', action='store_true', default=False, help='Enable detailed epoch logging')
    parser.add_argument('--dynamic_gradient_scaling', action='store_true', default=False, help='Enable dynamic gradient scaling')
    parser.add_argument('--relaxed_optimization_mode', action='store_true', default=False, help='Enable relaxed optimization mode')
    
    if input_string is not None:
        opt = parser.parse_args(input_string)
    else:
        opt = parser.parse_args()

    return opt

def train_from_opt(opt):
    """
    Train from options (function docstring unchanged)
    """
    # --- Log configuration moved here ---
    log_level = logging.DEBUG  # Changed to DEBUG level to show detailed information
    log_format = '%(asctime)s | %(levelname)s | %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'

    root_logger = logging.getLogger()
    # Clear any existing old handlers, ensure each call reconfigures
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        handler.close()
    
    root_logger.setLevel(log_level)
    formatter = logging.Formatter(log_format, datefmt=date_format)

    handlers = []
    # Add console handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)
    print("Added console log handler")

    # If file logging is enabled
    if hasattr(opt, 'log') and opt.log:
        log_dir = os.path.join(opt.output_dir, "logs") 
        try:
            os.makedirs(log_dir, exist_ok=True)
            
            # --- Build log filename suffix ---
            log_suffix = ""
            if getattr(opt, 'flow_opt_depth', False):
                log_suffix += "_depth"
            if getattr(opt, 'flow_opt_pose', False):
                 log_suffix += "_pose"
            if getattr(opt, 'flow_opt_intrinsics', False) and getattr(opt, 'opt_intrinsics', False):
                 log_suffix += "_intrinsics"
            
            if not log_suffix: # If no flow_opt enabled, can add marker or keep original name
                log_suffix = "_no_flow_opt" # Or log_suffix = "" to keep original name

            log_filename = f'training_info{log_suffix}.log'
            log_file = os.path.join(log_dir, log_filename)
            # --- Log filename construction end ---

            print(f"Log file path: {log_file}")
            
            # Explicitly set UTF-8 encoding
            file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8') # Use 'w' mode to overwrite
            file_handler.setFormatter(formatter)
            handlers.append(file_handler)
            print("Added file log handler")
        except OSError as e:
             print(f"Failed to create log directory or file: {e}, will only use console logging")
             logging.error(f"Failed to create log directory or file: {e}")

    # Configure root logger
    root_logger.handlers = handlers # Directly set handler list
    
    logging.info("Log system configuration completed (train_from_opt)")
    # Record final configuration
    arg_dict = vars(opt)
    json_string = json.dumps(arg_dict, indent=4)
    logging.info(f"Final configuration used:\n{json_string}")
    # --- Log configuration end ---

    # --- Original train_from_opt code starts ---
    if not hasattr(opt, 'debug_mode'):
        opt.debug_mode = False
    if not hasattr(opt, 'save_intermediate'):
        opt.save_intermediate = True
    if not hasattr(opt, 'load_from_intermediate'):
        opt.load_from_intermediate = False
    if not hasattr(opt, 'static_points_only'):
        opt.static_points_only = False

    print(f"Debug info - debug_mode: {opt.debug_mode}, load_from_intermediate: {opt.load_from_intermediate}, static_points_only: {opt.static_points_only}, types: {type(opt.debug_mode)}, {type(opt.load_from_intermediate)}")

    engine = Engine(opt)
    
    load_success = False

    if opt.debug_mode and opt.load_from_intermediate:
        print("Entering debug mode and attempting to load from intermediate state (for optical flow optimization alignment)")
        
        # Specify state file for optical flow debugging
        flow_debug_state_file_to_load = os.path.join(opt.output_dir, "engine_state_for_flow_debug.pth")
        if not os.path.exists(flow_debug_state_file_to_load):
            logging.error(f"State file for optical flow debugging {flow_debug_state_file_to_load} does not exist! Please run once completely to generate this file.")
            print(f"Error: State file for optical flow debugging {flow_debug_state_file_to_load} does not exist!")
            return

        load_success = engine.load_state(state_file=flow_debug_state_file_to_load)
        
        if load_success:
            print("Successfully loaded intermediate state for optical flow optimization debugging...")
            logging.info("Successfully loaded intermediate state for optical flow optimization debugging...")
            
            try:
                # Save backup of loaded camera poses to ensure they won't be lost in subsequent operations
                backup_poses = None
                if hasattr(engine, 'poses'):
                    backup_poses = engine.poses
                    logging.info("Backed up camera poses object")
                    
                # New: Save K_gt backup
                backup_K_gt = None
                if not opt.opt_intrinsics and hasattr(engine, 'K_gt'):
                    backup_K_gt = engine.K_gt
                    logging.info("Backed up ground truth camera intrinsics K_gt object")
                
                # Only load necessary image data
                print("Manually loading image data...")
                engine.load_images()
                # Move images to device after loading
                if hasattr(engine, 'images'):
                    engine.images = engine.images.to(engine.device)
                
                # When opt.opt_intrinsics=False, reload ground truth intrinsics
                if not opt.opt_intrinsics:
                    print("Reloading ground truth intrinsics data...")
                    engine.load_gt()
                    logging.info("Reloaded ground truth intrinsics")
                
                # Initialize engine variables (will reset poses)
                print("Initializing engine variables...")
                engine.init_vars()  # This will create new poses, overwriting previously loaded poses
                
                # Restore original camera poses
                if backup_poses is not None:
                    engine.poses = backup_poses
                    print("Restored original camera pose parameters")
                    logging.info("Restored original camera pose parameters")
                
                # New: Restore original intrinsics (when not optimizing intrinsics)
                if not opt.opt_intrinsics and backup_K_gt is not None:
                    engine.K_gt = backup_K_gt
                    print("Restored original ground truth camera intrinsics")
                    logging.info("Restored original ground truth camera intrinsics")
                
                # Check if state file already contains mask data
                if not hasattr(engine, 'is_static_strict_tensor') or engine.is_static_strict_tensor is None:
                    print("State file does not contain valid mask data, need to reload...")
                    engine.get_masks()
                else:
                    print("Using mask data from state file, skipping reload")
                
                # Determine if depth interpolation is needed
                needs_depth_interpolation = False
                if not hasattr(engine, 'reproj_depths') or engine.reproj_depths is None:
                    needs_depth_interpolation = True
                    print("Need to perform depth interpolation to generate reproj_depths...")
                
                if not hasattr(engine, 'reproj_depths_param') or engine.reproj_depths_param is None:
                    if hasattr(engine, 'reproj_depths') and engine.reproj_depths is not None:
                        print("Creating reproj_depths_param from existing reproj_depths...")
                        engine.reproj_depths_param = torch.nn.Parameter(
                            engine.reproj_depths.clone().to(engine.device), requires_grad=True
                        )
                    else:
                        needs_depth_interpolation = True
                        print("Need to perform depth interpolation to generate reproj_depths_param...")
                
                # If needed, perform depth interpolation
                if needs_depth_interpolation:
                    print("Performing depth interpolation...")
                    engine.depth_interpolate(make_optimizable=True)
                
                # Initialize optimizer
                print("Initializing optimizer...")
                engine.init_optimization(lr=getattr(opt, 'ba_lr', 1e-3))
                
                print("Dependency data and variable initialization completed, ready for optical flow optimization.")
                logging.info("Dependency data and variable initialization completed, ready for optical flow optimization.")
                
                # Output key tensor states
                if hasattr(engine, 'depth') and engine.depth is not None:
                    print(f"Depth map parameters: shape={engine.depth.shape}, mean={engine.depth.mean().item():.4f}")
                if hasattr(engine, 'reproj_depths') and engine.reproj_depths is not None:
                    print(f"Reprojection depth: shape={engine.reproj_depths.shape}, mean={engine.reproj_depths.mean().item():.4f}")
                if hasattr(engine, 'reproj_depths_param') and engine.reproj_depths_param is not None:
                    print(f"Reprojection depth parameters: shape={engine.reproj_depths_param.shape}, mean={engine.reproj_depths_param.mean().item():.4f}")

            except Exception as e:
                print(f"Error loading dependency data or initializing variables (debug mode): {e}")
                logging.error(f"Error loading dependency data or initializing variables (debug mode): {e}", exc_info=True)
                return

            # --- Now can start dense optical flow optimization ---
            # print("Saving results before optical flow optimization for comparison...")
            # logging.info("Saving results before optical flow optimization for comparison...")
            # before_suffix = "_before_flow_opt"
            # engine.save_results(save_fused_points=opt.vis_4d, suffix=before_suffix)
            
            # Also save engine state to ensure recovery to state before optical flow optimization
            # before_flow_state_file = os.path.join(opt.output_dir, "engine_state_before_flow_opt.pth")
            # print(f"Saving engine state before optical flow optimization to: {before_flow_state_file}")
            # logging.info(f"Saving engine state before optical flow optimization to: {before_flow_state_file}")
            # engine.save_state(state_file=before_flow_state_file)

            # Ensure key parameters exist (debug mode)
            if not hasattr(opt, 'flow_lr'):
                opt.flow_lr = 1e-4
                logging.info(f"Set default flow_lr = {opt.flow_lr}")
            
            if not hasattr(opt, 'flow_weight'):
                opt.flow_weight = 0.1
                logging.info(f"Set default flow_weight = {opt.flow_weight}")
            
            if not hasattr(opt, 'flow_sample_ratio'):
                opt.flow_sample_ratio = 0.1
                logging.info(f"Set default flow_sample_ratio = {opt.flow_sample_ratio}")

            dense_flow_start_time = time.time()
            logging.info("Starting simplified optical flow optimization (optimize fused depth), recording start time...")
            print("Executing simplified optical flow optimization (optimize fused depth)...")
            logging.info(f"Optical flow optimization parameters: lr={opt.flow_lr}, weight={opt.flow_weight}, sample_ratio={opt.flow_sample_ratio}")
            engine.optimize_with_dense_flow() # Internally calls simplified optimize_simple_flow

            # Calculate and record optimization time
            dense_flow_time = time.time() - dense_flow_start_time
            print(f"Simplified optical flow optimization completed, total time: {dense_flow_time:.2f} seconds")
            logging.info(f"Simplified optical flow optimization completed, total time: {dense_flow_time:.2f} seconds")
            
            # Write optimization time to file
            time_log_path = os.path.join(opt.output_dir, "optimization_times.txt")
            with open(time_log_path, "a", encoding='utf-8') as f:
                f.write(f"Simplified optical flow optimization time: {dense_flow_time:.2f} seconds\n")
            
            # --- Save final results ---
            # save_results internally updates self.reproj_depths from reproj_depths_param
            print("Saving final results...")
            logging.info("Saving final results...")
            # Recommend using suffix that better reflects optimization objective
            final_suffix = opt.suffix if opt.suffix else "_dynamic_flow_fused"
            engine.save_results(save_fused_points=opt.vis_4d, suffix=final_suffix)

            return

        else:
            print("Failed to load intermediate state, will execute complete pipeline")
            logging.warning("Failed to load intermediate state, will execute complete pipeline")
    
    # Only execute full initialization and optimization pipeline when not loading state or loading failed
    if not load_success:
        # Check if file has already been processed
        final_suffix = opt.suffix if opt.suffix else "_dynamic_flow_fused"
        intermediate_suffix = "_before_flow_opt"
        
        # Get all files in output directory
        try:
            output_files = os.listdir(opt.output_dir)
        except:
            output_files = []
            
        # Check if any file contains final suffix (indicates optical flow optimization completed)
        has_final_result = any(final_suffix in f for f in output_files)
        if has_final_result:
            print(f"Skipping processing: found files with suffix {final_suffix}, indicates optical flow optimization completed")
            logging.info(f"Skipping processing: found files with suffix {final_suffix}, indicates optical flow optimization completed")
            return  # Exit current processing, move to next file
        
        # Check if any file contains intermediate suffix (indicates BA/Dyn optimization completed)
        has_intermediate_result = any(intermediate_suffix in f for f in output_files)
        engine_state_exists = os.path.exists(os.path.join(opt.output_dir, "engine_state.pth"))
        
        print(f"has_intermediate_result: {has_intermediate_result}, engine_state_exists: {engine_state_exists}")
        if has_intermediate_result:
            print(f"Found files with suffix {intermediate_suffix}, attempting to load from intermediate state and only execute optical flow optimization")
            logging.info(f"Found files with suffix {intermediate_suffix}, attempting to load from intermediate state and only execute optical flow optimization")
            
            # Force set to debug mode and load from intermediate state
            opt.debug_mode = True
            opt.load_from_intermediate = True
            
            # Recursively call itself, but now already in debug_mode and load_from_intermediate=True
            # Will directly enter the debug mode branch above
            return train_from_opt(opt)

        # If no intermediate result files found, execute full pipeline
        print("No intermediate or final result files found, executing full initialization pipeline...")
        logging.info("No intermediate or final result files found, executing full initialization pipeline...")
        engine.initialize() # -> load_data -> init_vars -> init_optimization (includes depth opt) -> to_device

        # --- Initial optimization phase ---
        engine.optimize_init()
        engine.log_timer("init")
        engine.optimize_BA()
        engine.reinitialize_static()
        engine.log_timer("BA")
        if engine.num_points_dyn > 0:
            engine.init_dyn_cp()
            engine.optimize_dyn()
            engine.filter_dyn()
            engine.log_timer("dyn")

        # --- Save intermediate state needed for optical flow optimization debugging ---
        # Key saving point: BA/Dyn optimization completed, save state after depth interpolation
        flow_debug_state_file = os.path.join(opt.output_dir, "engine_state_for_flow_debug.pth")

        # [Modified] First execute non-optimizable version of depth interpolation to ensure state contains reprojection depth
        print("Performing depth interpolation (preparing reproj_depths for state saving)...")
        logging.info("Performing depth interpolation (preparing reproj_depths for state saving)...")
        engine.depth_interpolate(make_optimizable=False)

        # [Modified] Save complete state including depth and reproj_depths
        logging.info(f"Saving intermediate state for optical flow optimization debugging to: {flow_debug_state_file}")
        print(f"Saving intermediate state for optical flow optimization debugging to: {flow_debug_state_file}")
        engine.save_state(state_file=flow_debug_state_file)


        # Save interpolation results (NPZ file)
        # print("Saving results after depth interpolation...")
        # logging.info("Saving results after depth interpolation...")
        # interpolation_suffix = "_before_flow_opt"
        # engine.save_results(save_fused_points=opt.vis_4d, suffix=interpolation_suffix)

        # Create optimizable version of parameters for optical flow optimization
        print("Creating optimizable reprojection depth parameters for optical flow optimization...")
        engine.depth_interpolate(make_optimizable=True)

        # [Modified] Clear previous checkpoint, ensure ready for new optimizer
        if "reproj_depth" not in engine.optimizers and hasattr(engine, 'reproj_depths_param'):
            reproj_depth_lr = getattr(opt, 'reproj_depth_lr', getattr(opt, 'depth_lr', 1e-4))
            engine.optimizers["reproj_depth"] = torch.optim.Adam([engine.reproj_depths_param], lr=reproj_depth_lr)
            logging.info(f"Created reproj_depth optimizer, lr={reproj_depth_lr}")
            engine.reset_schedulers(patience=getattr(opt, 'ba_lr_patience', 10))

        # --- Prepare dense optical flow optimization (input state is now explicit) ---
        print("Performing depth interpolation (generating reproj_depths_param for optical flow optimization)...")
        logging.info("Performing depth interpolation (generating reproj_depths_param for optical flow optimization)...")
        engine.depth_interpolate(make_optimizable=True) # Use original self.depth and optimized camera/control points

        if "reproj_depth" not in engine.optimizers and hasattr(engine, 'reproj_depths_param') and isinstance(engine.reproj_depths_param, torch.nn.Parameter):
             reproj_depth_lr = getattr(opt, 'reproj_depth_lr', getattr(opt, 'depth_lr', 1e-4))
             engine.optimizers["reproj_depth"] = torch.optim.Adam([engine.reproj_depths_param], lr=reproj_depth_lr)
             logging.info(f"Manually created reproj_depth optimizer, lr={reproj_depth_lr}")
             # If a new optimizer is created here, may also need to reset or update related scheduler
             engine.reset_schedulers(patience=getattr(opt, 'ba_lr_patience', 10)) # Example, adjust patience as needed


        # --- Simplified optical flow optimization ---
        # Ensure key parameters exist
        if not hasattr(opt, 'flow_lr'):
            opt.flow_lr = 1e-4
            logging.info(f"Set default flow_lr = {opt.flow_lr}")
        
        if not hasattr(opt, 'flow_weight'):
            opt.flow_weight = 1.0  # Increase default weight
            logging.info(f"Set default flow_weight = {opt.flow_weight}")
        elif opt.flow_weight < 0.5:
            opt.flow_weight = 1.0  # If weight too small, force increase
            logging.info(f"Optical flow weight too small, increased to flow_weight = {opt.flow_weight}")
        
        if not hasattr(opt, 'flow_sample_ratio'):
            opt.flow_sample_ratio = 0.1
            logging.info(f"Set default flow_sample_ratio = {opt.flow_sample_ratio}")
        
        dense_flow_start_time = time.time()
        logging.info("Starting simplified optical flow optimization (optimize fused depth), recording start time...")
        print("Executing simplified optical flow optimization (optimize fused depth)...")
        logging.info(f"Optical flow optimization parameters: lr={opt.flow_lr}, weight={opt.flow_weight}, sample_ratio={opt.flow_sample_ratio}")
        engine.optimize_with_dense_flow() # Internally calls simplified optimize_simple_flow

        # Calculate and record optimization time
        dense_flow_time = time.time() - dense_flow_start_time
        print(f"Simplified optical flow optimization completed, total time: {dense_flow_time:.2f} seconds")
        logging.info(f"Simplified optical flow optimization completed, total time: {dense_flow_time:.2f} seconds")
        
        # Write optimization time to file
        time_log_path = os.path.join(opt.output_dir, "optimization_times.txt")
        with open(time_log_path, "a", encoding='utf-8') as f:
            f.write(f"Simplified optical flow optimization time: {dense_flow_time:.2f} seconds\n")
            
        # --- Save final results ---
        logging.info("Saving final optical flow optimization results...")
        print("Saving final optical flow optimization results...")
        final_suffix = opt.suffix if opt.suffix else "_dynamic_flow_fused" # Use command line suffix or default
        engine.save_results(save_fused_points=opt.vis_4d, suffix=final_suffix)

    del engine

def main():
    opt = parse_args()

    # --- 1. Determine final path and add to opt --- 
    video_name_for_path = "default_video" # Default value
    if hasattr(opt, 'video') and opt.video is not None and opt.video != "None":
        # Single video mode
        video_name_for_path = opt.video
    elif hasattr(opt, 'workdir'):
        # Multi-video mode, try to get first video
        try:
            videos = sorted(os.listdir(f"{opt.workdir}"))
            if videos:
                video_name_for_path = videos[0]
                print(f"Multi-video mode, using first video found '{video_name_for_path}' to determine initial path")
            else:
                 print(f"Work directory {opt.workdir} is empty, using default video name '{video_name_for_path}'")
        except FileNotFoundError:
            print(f"Work directory {opt.workdir} not found, using default video name '{video_name_for_path}'")
            
    base_dir = os.path.abspath(os.path.join(opt.workdir if hasattr(opt, 'workdir') else '.', video_name_for_path))
    experiment_name = opt.experiment_name if hasattr(opt, 'experiment_name') and opt.experiment_name is not None else "default_exp"
    output_dir = os.path.abspath(os.path.join(base_dir, "dynamicBA", experiment_name))
    
    opt.BASE = base_dir 
    opt.output_dir = output_dir
    opt.video_name = video_name_for_path # Also store video name used for path calculation in opt
    print(f"Finally determined initial BASE: {opt.BASE}")
    print(f"Finally determined initial output_dir: {opt.output_dir}")
    os.makedirs(opt.output_dir, exist_ok=True) # Ensure directory exists

    # --- 2. *Immediately* configure logging system --- 
    # setup_logging(opt, opt.output_dir)

    # --- 3. Set environment and other parameters --- 
    # if hasattr(opt, 'gpu'):
    #     os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu

    # Control thread count for various numerical computation libraries
    os.environ['NUMEXPR_MAX_THREADS'] = '64'

    # PyTorch thread count
    torch.set_num_threads(1)

    # --- 4. Execute main logic --- 
    is_multi_video_mode = not hasattr(opt, 'video') or opt.video is None or opt.video.lower() == "none"

    if is_multi_video_mode:
        print("Entering multi-video processing mode...")
        logging.info("Starting multi-video processing mode")
        try:
            # Get video list again to ensure all videos are processed
            actual_workdir = opt.workdir if hasattr(opt, 'workdir') else '.'
            videos = sorted([d for d in os.listdir(actual_workdir) if os.path.isdir(os.path.join(actual_workdir, d))])
        except FileNotFoundError:
            logging.error(f"Work directory {actual_workdir} does not exist, cannot process multi-video mode")
            videos = []

        if not videos:
            logging.warning(f"Work directory '{actual_workdir}' is empty or contains no subdirectories, no videos found to process")

        for i, video in enumerate(videos):
            print(f"\n--- ({i+1}/{len(videos)}) Checking scene: {video} ---", flush=True)

            # --- Build check path for current video ---
            # Assume experiment_name is fixed as 'base'
            current_experiment_name = "base" # Keep consistent with check file path
            current_scene_base_path = os.path.abspath(os.path.join(actual_workdir, video))
            current_output_dir = os.path.abspath(os.path.join(current_scene_base_path, "dynamicBA", current_experiment_name))

            # === New: Check if specific result file exists ===
            check_file_name = "poses_simple_flow_opt_1.0_1e-4.npz"
            check_file_path = os.path.join(current_output_dir, check_file_name)

            if os.path.exists(check_file_path):
                print(f"⏭️  Scene '{video}' already processed, skipping (check file exists: {check_file_path})")
                logging.info(f"Skipping video {video}: Check file exists at {check_file_path}")
                continue # Skip processing this video
            else:
                 print(f"   Check file does not exist: {check_file_path}")
                 print(f"   Preparing to process scene '{video}'...")
            # === Check complete ===

            # === Only configure and call train_from_opt when processing is needed ===
            logging.info(f"---")
            logging.info(f"Starting processing for video: {video}")

            # Create video-specific opt object (keep unchanged)
            specific_opt = argparse.Namespace(**vars(opt)) # Copy general config
            specific_opt.video = video # Set current video name (may be needed inside train_from_opt)
            specific_opt.video_name = video # Keep consistency
            specific_opt.BASE = current_scene_base_path # Set BASE path for current scene
            specific_opt.output_dir = current_output_dir # Set output path for current scene
            specific_opt.experiment_name = current_experiment_name # Explicitly set experiment_name

            # Ensure output directory for current scene exists
            try:
                os.makedirs(specific_opt.output_dir, exist_ok=True)
                logging.info(f"  - Video BASE: {specific_opt.BASE}")
                logging.info(f"  - Video output_dir: {specific_opt.output_dir}")
            except OSError as e:
                 logging.error(f"Cannot create output directory '{specific_opt.output_dir}' for scene '{video}': {e}")
                 print(f"❌ Cannot create output directory for scene '{video}', skipping this scene.")
                 continue # Skip this scene that cannot create directory

            # Call processing function
            try:
                train_from_opt(specific_opt)
            except RuntimeError as e: # Catch runtime errors like out of memory
                 logging.error(f"Runtime error occurred while processing video {video}: {e}", exc_info=True)
                 print(f"❌ Runtime error occurred while processing video {video} (possibly out of memory), skipping this scene. Error: {e}")
                 # Choose continue instead of exit to continue processing other videos
            except Exception as e:
                logging.error(f"Unknown error occurred while processing video {video}: {e}", exc_info=True)
                print(f"❌ Unknown error occurred while processing video {video}, skipping this scene. Error: {e}")
            finally:
                # Try to clean GPU memory, prepare for next video
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logging.info(f"Finished processing attempt for video: {video}")
                print(f"--- Scene '{video}' processing attempt ended ---")

        logging.info("Multi-video mode processing completed")
    else:
        # Single video mode (logic remains unchanged, but can also add file check)
        print(f"Working on single video: {opt.video}", flush=True)

        # === (Optional) Also add check for single video mode ===
        # Use already calculated path in opt
        check_file_name = "poses_simple_flow_opt_1.0_1e-4.npz"
        # Assume experiment_name is fixed as 'base'
        current_experiment_name = "base"
        current_output_dir = os.path.abspath(os.path.join(opt.BASE, "dynamicBA", current_experiment_name))
        check_file_path = os.path.join(current_output_dir, check_file_name)

        if os.path.exists(check_file_path):
             print(f"⏭️  Scene '{opt.video}' already processed, skipping (check file exists: {check_file_path})")
             logging.info(f"Skipping single video {opt.video}: Check file exists at {check_file_path}")
             # If single video mode, can directly exit or do nothing
             sys.exit(0) # Or return
        else:
             print(f"   Check file does not exist: {check_file_path}")
             print(f"   Preparing to process scene '{opt.video}'...")
        # === Check complete ===

        logging.info(f"Starting processing for single video: {opt.video}")
        try:
            # Ensure output directory for single video mode also exists
            os.makedirs(current_output_dir, exist_ok=True) # Use directory calculated during check
            opt.output_dir = current_output_dir # Ensure output_dir in opt is correct
            train_from_opt(opt)
        except RuntimeError as e:
            logging.error(f"Runtime error occurred while processing video {opt.video}: {e}", exc_info=True)
            print(f"❌ Runtime error occurred while processing video {opt.video} (possibly out of memory). Error: {e}")
        except Exception as e:
            logging.error(f"Unknown error occurred while processing video {opt.video}: {e}", exc_info=True)
            print(f"❌ Unknown error occurred while processing video {opt.video}. Error: {e}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logging.info(f"Finished processing for single video: {opt.video}")

if __name__ == '__main__':
    main()