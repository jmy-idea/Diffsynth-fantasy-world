"""
Fantasy World specific data operators for loading:
- Depth maps
- Point clouds  
- Camera parameters (9D: rotation, translation, fov)
"""

import torch
import numpy as np
import os
from PIL import Image
from .operators import DataProcessingOperator, DataProcessingPipeline, ToAbsolutePath


class LoadDepthMap(DataProcessingOperator):
    """Load depth map from various formats (.npy, .npz, .png, .exr)"""
    def __init__(self, normalize=True, max_depth=100.0):
        self.normalize = normalize
        self.max_depth = max_depth
    
    def __call__(self, data: str):
        if data.endswith('.npy'):
            depth = np.load(data)
        elif data.endswith('.npz'):
            depth_data = np.load(data)
            # Try common keys
            for key in ['depth', 'arr_0', 'data']:
                if key in depth_data:
                    depth = depth_data[key]
                    break
            else:
                depth = list(depth_data.values())[0]
        elif data.endswith('.png') or data.endswith('.jpg'):
            # Assume 16-bit PNG depth
            depth_img = Image.open(data)
            depth = np.array(depth_img).astype(np.float32)
            if depth.max() > 255:  # 16-bit
                depth = depth / 65535.0 * self.max_depth
            else:
                depth = depth / 255.0 * self.max_depth
        else:
            raise ValueError(f"Unsupported depth format: {data}")
        
        depth = torch.from_numpy(depth.astype(np.float32))
        if self.normalize:
            depth = depth / self.max_depth
        return depth


class LoadPointCloud(DataProcessingOperator):
    """Load point cloud from .npy or .npz files
    Expected shape: [H, W, 3] or [N, 3] for each frame
    """
    def __init__(self):
        pass
    
    def __call__(self, data: str):
        if data.endswith('.npy'):
            points = np.load(data)
        elif data.endswith('.npz'):
            points_data = np.load(data)
            for key in ['points', 'pts', 'pointmap', 'arr_0']:
                if key in points_data:
                    points = points_data[key]
                    break
            else:
                points = list(points_data.values())[0]
        else:
            raise ValueError(f"Unsupported point cloud format: {data}")
        
        points = torch.from_numpy(points.astype(np.float32))
        return points


class LoadCameraParams(DataProcessingOperator):
    """Load camera parameters file path.
    
    Accepts .txt files where each line contains camera pose data.
    The last 12 values of each line are the w2c matrix (3x4 flattened).
    Returns the file path as-is for downstream processing.
    """
    def __init__(self):
        pass
    
    def __call__(self, data):
        # Just return the file path as-is
        # The actual parsing (extracting last 12 values as w2c matrix) 
        # will be handled by downstream model code
        if isinstance(data, str):
            if not data.endswith('.txt'):
                raise ValueError(f"Camera params must be .txt files, got: {data}")
            return data
        else:
            raise ValueError(f"Camera params must be file path string, got: {type(data)}")


class LoadDepthSequence(DataProcessingOperator):
    """Load a sequence of depth maps for video"""
    def __init__(self, num_frames=81, normalize=True, max_depth=100.0):
        self.num_frames = num_frames
        self.depth_loader = LoadDepthMap(normalize=normalize, max_depth=max_depth)
    
    def __call__(self, data):
        if isinstance(data, str):
            if os.path.isdir(data):
                # Directory of depth files
                files = sorted([f for f in os.listdir(data) if f.endswith(('.npy', '.npz', '.png'))])
                files = files[:self.num_frames]
                depths = [self.depth_loader(os.path.join(data, f)) for f in files]
            elif data.endswith('.npy') or data.endswith('.npz'):
                # Single file with all frames
                depth_all = self.depth_loader(data)
                if depth_all.ndim == 3:  # [T, H, W]
                    depths = [depth_all[i] for i in range(min(len(depth_all), self.num_frames))]
                else:
                    depths = [depth_all]
            else:
                depths = [self.depth_loader(data)]
        elif isinstance(data, list):
            depths = [self.depth_loader(f) for f in data[:self.num_frames]]
        else:
            raise ValueError(f"Unsupported depth sequence format: {type(data)}")
        
        # Stack to [T, H, W]
        depths = torch.stack(depths, dim=0)
        return depths


class LoadPointCloudSequence(DataProcessingOperator):
    """Load a sequence of point clouds for video"""
    def __init__(self, num_frames=81):
        self.num_frames = num_frames
        self.points_loader = LoadPointCloud()
    
    def __call__(self, data):
        if isinstance(data, str):
            if os.path.isdir(data):
                files = sorted([f for f in os.listdir(data) if f.endswith(('.npy', '.npz'))])
                files = files[:self.num_frames]
                points = [self.points_loader(os.path.join(data, f)) for f in files]
            elif data.endswith('.npy') or data.endswith('.npz'):
                points_all = self.points_loader(data)
                if points_all.ndim == 4:  # [T, H, W, 3]
                    points = [points_all[i] for i in range(min(len(points_all), self.num_frames))]
                else:
                    points = [points_all]
            else:
                points = [self.points_loader(data)]
        elif isinstance(data, list):
            points = [self.points_loader(f) for f in data[:self.num_frames]]
        else:
            raise ValueError(f"Unsupported point cloud sequence format: {type(data)}")
        
        # Stack to [T, H, W, 3]
        points = torch.stack(points, dim=0)
        return points


class ResizeDepthToMatch(DataProcessingOperator):
    """Resize depth map to match video frame size"""
    def __init__(self):
        pass
    
    def __call__(self, data):
        depth, target_size = data  # depth: [T, H, W], target_size: (H, W)
        if depth.shape[-2:] != target_size:
            depth = torch.nn.functional.interpolate(
                depth.unsqueeze(1),  # [T, 1, H, W]
                size=target_size,
                mode='bilinear',
                align_corners=True
            ).squeeze(1)  # [T, H, W]
        return depth


class ResizePointsToMatch(DataProcessingOperator):
    """Resize point cloud to match video frame size"""
    def __init__(self):
        pass
    
    def __call__(self, data):
        points, target_size = data  # points: [T, H, W, 3], target_size: (H, W)
        if points.shape[1:3] != target_size:
            # Reshape for interpolation
            t, h, w, c = points.shape
            points = points.permute(0, 3, 1, 2)  # [T, 3, H, W]
            points = torch.nn.functional.interpolate(
                points,
                size=target_size,
                mode='bilinear',
                align_corners=True
            )
            points = points.permute(0, 2, 3, 1)  # [T, H', W', 3]
        return points


# Convenience functions for building dataset operators
def default_depth_operator(base_path="", num_frames=81, normalize=True, max_depth=100.0):
    """Default operator for loading depth sequences"""
    return ToAbsolutePath(base_path) >> LoadDepthSequence(num_frames, normalize, max_depth)


def default_points_operator(base_path="", num_frames=81):
    """Default operator for loading point cloud sequences"""
    return ToAbsolutePath(base_path) >> LoadPointCloudSequence(num_frames)


def default_camera_operator(base_path=""):
    """Default operator for loading camera parameters (returns file path).
    
    The returned path points to a .txt file where each line contains:
    [... other data ...] [w2c_00, w2c_01, ..., w2c_23]
    where the last 12 values are the 3x4 w2c matrix flattened.
    """
    return ToAbsolutePath(base_path) >> LoadCameraParams()
