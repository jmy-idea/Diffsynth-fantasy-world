"""
Fantasy World Dataset for training geometry-aware video generation.

Expected data format in metadata (JSON/JSONL/CSV):
{
    "video": "path/to/video.mp4",
    "prompt": "text description",
    "depth": "path/to/depth_sequence.npy",       # [T, H, W]
    "points": "path/to/point_cloud.npy",        # [T, H, W, 3]
    "camera_params": "path/to/camera.txt",      # Each line: [prefix...] [w2c_00, ..., w2c_23]
}

Camera parameters format (.txt):
- One line per frame
- Format: [prefix values...] [12 w2c matrix values]
- The last 12 values are the 3x4 w2c matrix flattened in row-major order
- Prefix values are ignored, only the last 12 are used by the model
"""

import torch
import os
import json
from .unified_dataset import UnifiedDataset
from .operators import (
    ToAbsolutePath, LoadVideo, ImageCropAndResize, 
    RouteByType, RouteByExtensionName, LoadImage, ToList, LoadGIF
)
from .fantasy_world_operators import (
    LoadDepthSequence, LoadPointCloudSequence, LoadCameraParams,
    default_depth_operator, default_points_operator, default_camera_operator
)


class FantasyWorldDataset(UnifiedDataset):
    """
    Dataset for Fantasy World training with geometry supervision.
    
    Loads:
    - Video frames (RGB)
    - Depth maps (optional)
    - Point clouds (optional)
    - Camera parameters (optional)
    """
    
    def __init__(
        self,
        base_path=None,
        metadata_path=None,
        repeat=1,
        # Video settings
        max_pixels=1920*1080,
        height=None,
        width=None,
        num_frames=81,
        # Geometry settings
        load_depth=True,
        load_points=True,
        load_camera=True,
        depth_normalize=True,
        max_depth=100.0,
        # Other
        max_data_items=None,
    ):
        self.load_depth = load_depth
        self.load_points = load_points
        self.load_camera = load_camera
        self.num_frames = num_frames
        self.height = height
        self.width = width
        
        # Build data file keys based on what we need to load
        data_file_keys = ["video"]
        if load_depth:
            data_file_keys.append("depth")
        if load_points:
            data_file_keys.append("points")
        if load_camera:
            data_file_keys.append("camera_params")
        
        # Build special operators for each data type
        special_operator_map = {}
        
        if load_depth:
            special_operator_map["depth"] = default_depth_operator(
                base_path=base_path or "",
                num_frames=num_frames,
                normalize=depth_normalize,
                max_depth=max_depth
            )
        
        if load_points:
            special_operator_map["points"] = default_points_operator(
                base_path=base_path or "",
                num_frames=num_frames
            )
        
        if load_camera:
            special_operator_map["camera_params"] = default_camera_operator(
                base_path=base_path or ""
            )
        
        # Main video operator
        main_data_operator = self.default_video_operator(
            base_path=base_path or "",
            max_pixels=max_pixels,
            height=height,
            width=width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        )
        
        super().__init__(
            base_path=base_path,
            metadata_path=metadata_path,
            repeat=repeat,
            data_file_keys=tuple(data_file_keys),
            main_data_operator=main_data_operator,
            special_operator_map=special_operator_map,
            max_data_items=max_data_items,
        )
    
    def __getitem__(self, data_id):
        data = super().__getitem__(data_id)
        
        # Post-process to ensure proper shapes and types
        if "video" in data:
            video_frames = data["video"]
            if len(video_frames) > 0:
                h, w = video_frames[0].size[1], video_frames[0].size[0]
                data["height"] = h
                data["width"] = w
                data["num_frames"] = len(video_frames)
        
        # Ensure depth has correct shape [T, H, W]
        if "depth" in data and data["depth"] is not None:
            depth = data["depth"]
            if isinstance(depth, torch.Tensor):
                if depth.ndim == 2:  # [H, W] -> [1, H, W]
                    depth = depth.unsqueeze(0)
                data["gt_depth"] = depth
            del data["depth"]
        
        # Ensure points has correct shape [T, H, W, 3]
        if "points" in data and data["points"] is not None:
            points = data["points"]
            if isinstance(points, torch.Tensor):
                if points.ndim == 3:  # [H, W, 3] -> [1, H, W, 3]
                    points = points.unsqueeze(0)
                data["gt_points"] = points
            del data["points"]
        
        # Ensure camera params - just return the path for downstream processing
        if "camera_params" in data and data["camera_params"] is not None:
            cam_path = data["camera_params"]
            # Verify it's a string path
            if isinstance(cam_path, str):
                data["pose_params"] = cam_path
            else:
                # Fallback: convert to string if needed
                data["pose_params"] = str(cam_path)
            del data["camera_params"]
        
        return data


def create_dummy_fantasy_world_data(
    output_dir: str,
    num_samples: int = 2,
    num_frames: int = 5,  # Small for testing
    height: int = 64,
    width: int = 64,
):
    """
    Create dummy data for testing the Fantasy World training pipeline.
    
    Creates:
    - Fake video files (MP4)
    - Fake depth maps
    - Fake point clouds
    - Fake camera parameters
    - Metadata file
    """
    import numpy as np
    from PIL import Image
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "videos"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "depths"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "points"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "cameras"), exist_ok=True)
    
    # Try to import imageio for MP4 creation
    try:
        import imageio
        use_imageio = True
    except ImportError:
        print("Warning: imageio not installed, creating frame directories instead")
        print("  Install with: pip install imageio imageio-ffmpeg")
        use_imageio = False
    
    metadata = []
    
    for sample_id in range(num_samples):
        sample_name = f"sample_{sample_id:04d}"
        
        if use_imageio:
            # Create video file (MP4)
            video_path = os.path.join(output_dir, "videos", f"{sample_name}.mp4")
            frames = []
            for frame_id in range(num_frames):
                # Create gradient image for variety
                img = np.zeros((height, width, 3), dtype=np.uint8)
                img[:, :, 0] = int(255 * frame_id / max(num_frames - 1, 1))  # R varies with time
                img[:, :, 1] = int(255 * sample_id / max(num_samples - 1, 1))  # G varies with sample
                img[:, :, 2] = 128  # B constant
                
                # Add some spatial variation
                for y in range(height):
                    for x in range(width):
                        img[y, x, 2] = int(128 + 64 * np.sin(x / 10) * np.cos(y / 10))
                
                frames.append(img)
            
            imageio.mimwrite(video_path, frames, fps=8, codec='libx264')
            video_rel_path = os.path.join("videos", f"{sample_name}.mp4")
        else:
            # Fallback to frame directory
            video_dir = os.path.join(output_dir, "videos", sample_name)
            os.makedirs(video_dir, exist_ok=True)
            
            for frame_id in range(num_frames):
                # Create gradient image for variety
                img = np.zeros((height, width, 3), dtype=np.uint8)
                img[:, :, 0] = int(255 * frame_id / max(num_frames - 1, 1))  # R varies with time
                img[:, :, 1] = int(255 * sample_id / max(num_samples - 1, 1))  # G varies with sample
                img[:, :, 2] = 128  # B constant
                
                # Add some spatial variation
                for y in range(height):
                    for x in range(width):
                        img[y, x, 2] = int(128 + 64 * np.sin(x / 10) * np.cos(y / 10))
                
                frame_path = os.path.join(video_dir, f"frame_{frame_id:04d}.png")
                Image.fromarray(img).save(frame_path)
            
            video_rel_path = os.path.join("videos", sample_name)
        
        # Generate fake depth maps [T, H, W]
        depths = np.random.rand(num_frames, height, width).astype(np.float32)
        # Add some structure
        for t in range(num_frames):
            xx, yy = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            depths[t] = 0.5 + 0.3 * np.sin(xx * 4 + t * 0.5) * np.cos(yy * 4)
        depth_path = os.path.join(output_dir, "depths", f"sample_{sample_id:04d}.npy")
        np.save(depth_path, depths)
        
        # Generate fake point clouds [T, H, W, 3]
        points = np.zeros((num_frames, height, width, 3), dtype=np.float32)
        xx, yy = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
        for t in range(num_frames):
            points[t, :, :, 0] = xx  # X
            points[t, :, :, 1] = yy  # Y
            points[t, :, :, 2] = depths[t] * 2 - 1  # Z from depth
        points_path = os.path.join(output_dir, "points", f"sample_{sample_id:04d}.npy")
        np.save(points_path, points)
        
        # Generate fake camera parameters as .txt file
        # Format: Each line has prefix (ignored) + last 12 values = w2c matrix (3x4 flattened)
        camera_path = os.path.join(output_dir, "cameras", f"sample_{sample_id:04d}.txt")
        with open(camera_path, 'w') as f:
            for t in range(num_frames):
                # Create a fake w2c matrix (3x4, 12 values)
                # Identity-like rotation with some variation
                angle = t * 0.1  # Slowly rotating camera
                
                # Simple rotation matrix (3x3) + translation (3x1)
                # R = rotation around Y axis
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                
                # w2c matrix (3x4) in row-major order
                w2c = np.array([
                    [cos_a, 0, sin_a, 0],
                    [0, 1, 0, 0],
                    [-sin_a, 0, cos_a, -2 + t * 0.1],
                ], dtype=np.float32)
                
                # Flatten to 12 values
                w2c_flat = w2c.reshape(-1)
                
                # Prefix (frame index + some dummy values)
                prefix = f"0 0.5 0.8 0.5 0.5 0.0 0.0"
                # Append the 12 w2c values
                line = prefix
                for val in w2c_flat:
                    line += f" {val:.6f}"
                f.write(line + "\n")
        
        # Add to metadata
        metadata.append({
            "video": video_rel_path,
            "prompt": f"A test scene with sample {sample_id}, showing geometric patterns.",
            "depth": os.path.join("depths", f"sample_{sample_id:04d}.npy"),
            "points": os.path.join("points", f"sample_{sample_id:04d}.npy"),
            "camera_params": os.path.join("cameras", f"sample_{sample_id:04d}.txt"),
        })
    
    # Save metadata
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Created dummy dataset at: {output_dir}")
    print(f"  - {num_samples} samples")
    print(f"  - {num_frames} frames per sample")
    print(f"  - Resolution: {height}x{width}")
    print(f"  - Metadata: {metadata_path}")
    
    return metadata_path


if __name__ == "__main__":
    # Test creating dummy data
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/tmp/fantasy_world_test_data")
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--num_frames", type=int, default=5)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    args = parser.parse_args()
    
    metadata_path = create_dummy_fantasy_world_data(
        args.output_dir,
        args.num_samples,
        args.num_frames,
        args.height,
        args.width,
    )
    
    # Test loading
    print("\nTesting dataset loading...")
    dataset = FantasyWorldDataset(
        base_path=args.output_dir,
        metadata_path=metadata_path,
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
    )
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {list(sample.keys())}")
    if "video" in sample:
        print(f"  video: {len(sample['video'])} frames")
    if "gt_depth" in sample:
        print(f"  gt_depth: {sample['gt_depth'].shape}")
    if "gt_points" in sample:
        print(f"  gt_points: {sample['gt_points'].shape}")
    if "pose_params" in sample:
        print(f"  pose_params: {sample['pose_params'].shape}")
