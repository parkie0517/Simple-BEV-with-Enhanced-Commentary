import numpy as np
import torch

def apply_4x4(RT, xyz):
    """
        batch-wise 3D transformation of a large number of points using homogeneous coordinates.
    """
    B, N, _ = list(xyz.shape)
    ones = torch.ones_like(xyz[:,:,0:1])
    xyz1 = torch.cat([xyz, ones], 2)
    xyz1_t = torch.transpose(xyz1, 1, 2)
    # this is B x 4 x N
    xyz2_t = torch.matmul(RT, xyz1_t)
    xyz2 = torch.transpose(xyz2_t, 1, 2)
    # xyz2 = xyz2 / xyz2[:,:,3:4]
    xyz2 = xyz2[:,:,:3]
    return xyz2

RT = np.array([
    [0, -1, 0],
    [1,  0, 0],
    [0,  0, 1]
])

# Translation vector (1, 1, 0)
T = np.array([1, 3, -1])

# Construct the 4x4 extrinsic matrix
extrinsic_matrix = np.eye(4)  # Initialize as 4x4 identity matrix
extrinsic_matrix[:3, :3] = RT  # Set the rotation part
extrinsic_matrix[:3, 3] = T        # Set the translation part



extrinsic_matrix = extrinsic_matrix.reshape(1, 4, 4)


xyz = np.array([
    [3, 4, 5],  # Point 1
    [1, 4, 1],  # Point 2
    [1, 2, 1]   # Point 3
])

xyz = xyz.reshape(1, 3, 3)

xyz = torch.from_numpy(xyz).float()
extrinsic_matrix = torch.from_numpy(extrinsic_matrix).float()


xyz_2 = apply_4x4(extrinsic_matrix, xyz) # changes the 3d coordinates according to the corresponding extrinsic
print(extrinsic_matrix)
print(xyz_2)