import numpy as np
from scipy.spatial.transform import Rotation

def convert_trajectory_to_extrinsic_matrices(positions, quaternions_xyzw):
    """
    Convert positions and quaternions to array of extrinsic matrices without loops.
    
    Input:
        positions: Nx3 array of positions
        quaternions_wxyz: Nx4 array of quaternions in the order [w, x, y, z]
    Output:
        Nx4x4 array of extrinsic matrices
    """
    n = positions.shape[0]

    # Create rotation matrices (Nx3x3) from quaternions
    rotations = Rotation.from_quat(quaternions_xyzw).as_matrix()
    
    # Initialize output array of 4x4 matrices
    extrinsics = np.zeros((n, 4, 4))
    
    # Set the bottom row to [0, 0, 0, 1] for all matrices
    extrinsics[:, 3, 3] = 1.0
    
    # Set the rotation part (top-left 3x3) for all matrices
    extrinsics[:, :3, :3] = rotations
    
    # Set the translation part (top-right 3x1) for all matrices
    extrinsics[:, :3, 3] = positions
    
    return extrinsics