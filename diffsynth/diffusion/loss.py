from .base_pipeline import BasePipeline
import torch


def FlowMatchSFTLoss(pipe: BasePipeline, **inputs):
    max_timestep_boundary = int(inputs.get("max_timestep_boundary", 1) * len(pipe.scheduler.timesteps))
    min_timestep_boundary = int(inputs.get("min_timestep_boundary", 0) * len(pipe.scheduler.timesteps))

    timestep_id = torch.randint(min_timestep_boundary, max_timestep_boundary, (1,))
    timestep = pipe.scheduler.timesteps[timestep_id].to(dtype=pipe.torch_dtype, device=pipe.device)
    
    noise = torch.randn_like(inputs["input_latents"])
    inputs["latents"] = pipe.scheduler.add_noise(inputs["input_latents"], noise, timestep)
    training_target = pipe.scheduler.training_target(inputs["input_latents"], noise, timestep)
    
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep)
    
    loss = torch.nn.functional.mse_loss(noise_pred.float(), training_target.float())
    loss = loss * pipe.scheduler.training_weight(timestep)
    return loss


def DirectDistillLoss(pipe: BasePipeline, **inputs):
    pipe.scheduler.set_timesteps(inputs["num_inference_steps"])
    pipe.scheduler.training = True
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
    for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = pipe.model_fn(**models, **inputs, timestep=timestep, progress_id=progress_id)
        inputs["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred, **inputs)
    loss = torch.nn.functional.mse_loss(inputs["latents"].float(), inputs["input_latents"].float())
    return loss

# =============================================================================
# [Fantasy World] Loss Implementation

def parse_camera_txt(file_path, num_frames=None):
    """
    Parse camera txt file and extract w2c matrices.
    
    Format: Each line has [prefix...] [w2c_00, ..., w2c_23]
    The last 12 values are the 3x4 w2c matrix flattened.
    
    Returns: torch.Tensor of shape [T, 12] containing w2c matrices
    """
    import os
    if not os.path.exists(file_path):
        return None
    
    w2c_matrices = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        if num_frames:
            lines = lines[:num_frames]
        
        for line in lines:
            values = line.strip().split()
            if len(values) < 12:
                continue
            # Extract last 12 values as w2c matrix
            w2c = [float(v) for v in values[-12:]]
            w2c_matrices.append(w2c)
    
    if len(w2c_matrices) == 0:
        return None
    
    return torch.tensor(w2c_matrices, dtype=torch.float32)


def FantasyWorldLoss(pipe: BasePipeline, **inputs):
    """
    Combined loss for Fantasy World training.
    
    Total loss = L_diffusion + L_geo
    where L_geo = L_depth + L_pmap + 3 * L_camera
    
    Expected DPTHead3D outputs:
    - Depth head: [B, T, 1, H, W]
    - Point head: (pts [B, T, 3, H, W], conf [B, T, 1, H, W])
    - Camera head: [B, T, 9]
    """
    # 1. Video Diffusion Loss
    loss_diffusion = FlowMatchSFTLoss(pipe, **inputs)
    
    loss_geo = torch.tensor(0.0, device=pipe.device, dtype=pipe.torch_dtype)
    
    # 2. Geometry Branch Supervision
    
    # --- A. Depth Loss ---
    # L_depth = alpha * L_TGM + beta * L_frame (following Video Depth Anything)
    if hasattr(pipe.dit, 'last_depth_output') and pipe.dit.last_depth_output is not None:
        pred_depth = pipe.dit.last_depth_output  # [B, T, 1, H, W] from DPTHead3D
        pipe.dit.last_depth_output = None
        
        gt_depth = inputs.get("gt_depth", None)
        if gt_depth is not None:
            # Convert gt_depth to same dtype and device as pred_depth
            gt_depth = gt_depth.to(dtype=pred_depth.dtype, device=pred_depth.device)
            
            # Ensure shape: [B, T, 1, H, W]
            if gt_depth.ndim == 3:  # [T, H, W] - single sample without batch
                gt_depth = gt_depth.unsqueeze(0).unsqueeze(2)  # [1, T, 1, H, W]
            elif gt_depth.ndim == 4:  # [B, T, H, W]
                gt_depth = gt_depth.unsqueeze(2)  # [B, T, 1, H, W]
            
            # Ensure gt_depth is 5D before proceeding
            assert gt_depth.ndim == 5, f"Expected gt_depth to be 5D [B,T,C,H,W], got shape {gt_depth.shape}"
            
            # Interpolate spatial dimensions if needed (BEFORE permuting)
            if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                B, T, C, H_gt, W_gt = gt_depth.shape
                _, _, _, H_pred, W_pred = pred_depth.shape
                # Reshape to [B*T, C, H, W] for spatial interpolation (keep C separate)
                gt_depth = gt_depth.reshape(B * T, C, H_gt, W_gt)
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth, size=(H_pred, W_pred), mode='bilinear', align_corners=True
                )
                gt_depth = gt_depth.reshape(B, T, C, H_pred, W_pred)
            
            # Interpolate temporal dimension if needed
            if pred_depth.shape[1] != gt_depth.shape[1]:  # T mismatch
                B, T_gt, C, H, W = gt_depth.shape
                T_pred = pred_depth.shape[1]
                # Reshape to [B*C, T, H*W] for temporal interpolation
                gt_depth = gt_depth.permute(0, 2, 1, 3, 4).reshape(B * C, T_gt, H * W)
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth, size=T_pred, mode='linear', align_corners=True
                )
                gt_depth = gt_depth.reshape(B, C, T_pred, H, W).permute(0, 2, 1, 3, 4)
            
            # Now permute to [B, C, T, H, W] for temporal gradient computation
            pred_depth = pred_depth.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
            gt_depth = gt_depth.permute(0, 2, 1, 3, 4)      # [B, C, T, H, W]

            # L_frame (MAE) - per-frame spatial loss
            loss_frame = torch.abs(pred_depth - gt_depth).mean()
             
            # L_TGM (Temporal Gradient Matching) - enforces temporal consistency
            dt_pred = pred_depth[:, :, 1:, :, :] - pred_depth[:, :, :-1, :, :]
            dt_gt = gt_depth[:, :, 1:, :, :] - gt_depth[:, :, :-1, :, :]
            loss_tgm = torch.abs(dt_pred - dt_gt).mean()
             
            loss_geo = loss_geo + loss_tgm + loss_frame

    # --- B. Point Map Loss ---
    # L_pmap = |Conf * (P - G)| + |Conf * (GradP - GradG)| - gamma * log(Conf)
    # Following VGGT with uncertainty-weighted regression
    if hasattr(pipe.dit, 'last_point_output') and pipe.dit.last_point_output is not None:
        point_output = pipe.dit.last_point_output
        pipe.dit.last_point_output = None
        
        # DPTHead3D returns tuple (pts, conf) for point head
        if isinstance(point_output, tuple):
            pts_pred, pts_conf = point_output  # [B, T, 3, H, W], [B, T, 1, H, W]
        else:
            # Fallback if single tensor
            pts_pred = point_output[:, :, :3, :, :]
            pts_conf = point_output[:, :, 3:4, :, :]
        
        gt_points = inputs.get("gt_points", None)
        if gt_points is not None:
            # Convert gt_points to same dtype and device as pts_pred
            gt_points = gt_points.to(dtype=pts_pred.dtype, device=pts_pred.device)
            
            # Ensure shape: [B, T, 3, H, W]
            if gt_points.ndim == 4:  # [T, 3, H, W] - single sample
                gt_points = gt_points.unsqueeze(0)  # [1, T, 3, H, W]
            elif gt_points.ndim == 5:  # Already [B, T, 3, H, W]
                pass
            else:
                # Try to infer correct shape
                B = pts_pred.shape[0]
                T = pts_pred.shape[1]
                if gt_points.shape[0] == B * T:  # [B*T, 3, H, W]
                    gt_points = gt_points.view(B, T, 3, *gt_points.shape[-2:])
             
            # Resize spatial dimensions if needed
            if pts_pred.shape[-2:] != gt_points.shape[-2:]:
                B, T, C, H_gt, W_gt = gt_points.shape
                _, _, _, H_pred, W_pred = pts_pred.shape
                # Reshape to [B*T, C, H, W] for spatial interpolation (keep C=3 separate)
                gt_points = gt_points.reshape(B * T, C, H_gt, W_gt)
                gt_points = torch.nn.functional.interpolate(
                    gt_points, size=(H_pred, W_pred), mode='bilinear', align_corners=True
                )
                gt_points = gt_points.reshape(B, T, C, H_pred, W_pred)
            
            # Point Loss (uncertainty weighted)
            diff = (pts_pred - gt_points).abs()
            loss_pts = (pts_conf * diff).mean()
             
            # Spatial Gradient Loss (H and W dimensions)
            def spatial_grads(x):
                # x: [B, T, C, H, W]
                dx = x[..., :, 1:] - x[..., :, :-1]  # gradient along W
                dy = x[..., 1:, :] - x[..., :-1, :]  # gradient along H
                return dx, dy
             
            dx_p, dy_p = spatial_grads(pts_pred)
            dx_g, dy_g = spatial_grads(gt_points)
             
            conf_dx = pts_conf[..., :, 1:]
            conf_dy = pts_conf[..., 1:, :]
             
            loss_grad = (conf_dx * (dx_p - dx_g).abs()).mean() + \
                        (conf_dy * (dy_p - dy_g).abs()).mean()
             
            # Regularization: -gamma * log(Conf) to prevent confidence collapse
            loss_reg = -0.1 * torch.log(pts_conf.clamp(min=1e-6)).mean()
             
            loss_geo = loss_geo + loss_pts + loss_grad + loss_reg

    # --- C. Camera Loss ---
    # L_camera = Robust Huber Loss on camera parameters (w2c matrices)
    if hasattr(pipe.dit, 'last_camera_output') and pipe.dit.last_camera_output is not None:
        pred_cam = pipe.dit.last_camera_output  # [B, T, 9]
        pipe.dit.last_camera_output = None
        
        # Parse ground truth from txt file
        gt_camera_file = inputs.get("gt_camera_file", None)
        if gt_camera_file is not None:
            # Handle batch: gt_camera_file could be a list or single path
            if isinstance(gt_camera_file, (list, tuple)):
                gt_camera_file = gt_camera_file[0]
            
            # Parse txt file to get w2c matrices [T, 12]
            gt_w2c = parse_camera_txt(gt_camera_file, num_frames=pred_cam.shape[1])
            
            if gt_w2c is not None:
                # Convert to same dtype and device as pred_cam
                gt_w2c = gt_w2c.to(dtype=pred_cam.dtype, device=pred_cam.device)
                # Add batch dimension if needed: [T, 12] -> [1, T, 12]
                if gt_w2c.ndim == 2:
                    gt_w2c = gt_w2c.unsqueeze(0)
                
                # Interpolate to match lengths if needed
                if pred_cam.shape[1] != gt_w2c.shape[1]:
                    gt_w2c = torch.nn.functional.interpolate(
                        gt_w2c.permute(0, 2, 1),  # [B, 12, T]
                        size=pred_cam.shape[1],
                        mode='linear',
                        align_corners=True
                    ).permute(0, 2, 1)  # [B, T, 12]
                
                # Extract rotation (first 9 values) and translation (last 3 values)
                # from w2c matrix [3x4 flattened]
                gt_rot = gt_w2c[:, :, :9]   # [B, T, 9] - rotation matrix
                gt_trans = gt_w2c[:, :, 9:] # [B, T, 3] - translation vector
                
                # If pred_cam is [B, T, 9], assume it's [rot(3), trans(3), fov(3)]
                # We compare rotation (first 3) and translation (next 3)
                if pred_cam.shape[-1] == 9:
                    pred_rot = pred_cam[:, :, :3]   # Simplified rotation (e.g., euler angles)
                    pred_trans = pred_cam[:, :, 3:6]  # Translation
                    
                    # For rotation: we can't directly compare euler vs matrix
                    # So just use translation loss for now (more stable)
                    loss_cam = torch.nn.functional.huber_loss(pred_trans, gt_trans, delta=1.0)
                elif pred_cam.shape[-1] == 12:
                    # Direct comparison with w2c
                    loss_cam = torch.nn.functional.huber_loss(pred_cam, gt_w2c, delta=1.0)
                else:
                    # Fallback: just use what we have
                    min_dim = min(pred_cam.shape[-1], gt_w2c.shape[-1])
                    loss_cam = torch.nn.functional.huber_loss(
                        pred_cam[:, :, :min_dim], 
                        gt_w2c[:, :, :min_dim], 
                        delta=1.0
                    )
                
                # Paper uses 3x weight for camera loss
                loss_geo = loss_geo + 3.0 * loss_cam
                    
    return loss_diffusion + loss_geo
# =============================================================================

class TrajectoryImitationLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.initialized = False
    
    def initialize(self, device):
        import lpips # TODO: remove it
        self.loss_fn = lpips.LPIPS(net='alex').to(device)
        self.initialized = True

    def fetch_trajectory(self, pipe: BasePipeline, timesteps_student, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        trajectory = [inputs_shared["latents"].clone()]

        pipe.scheduler.set_timesteps(num_inference_steps, target_timesteps=timesteps_student)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

            trajectory.append(inputs_shared["latents"].clone())
        return pipe.scheduler.timesteps, trajectory
    
    def align_trajectory(self, pipe: BasePipeline, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        loss = 0
        pipe.scheduler.set_timesteps(num_inference_steps, training=True)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)

            progress_id_teacher = torch.argmin((timesteps_teacher - timestep).abs())
            inputs_shared["latents"] = trajectory_teacher[progress_id_teacher]

            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )

            sigma = pipe.scheduler.sigmas[progress_id]
            sigma_ = 0 if progress_id + 1 >= len(pipe.scheduler.timesteps) else pipe.scheduler.sigmas[progress_id + 1]
            if progress_id + 1 >= len(pipe.scheduler.timesteps):
                latents_ = trajectory_teacher[-1]
            else:
                progress_id_teacher = torch.argmin((timesteps_teacher - pipe.scheduler.timesteps[progress_id + 1]).abs())
                latents_ = trajectory_teacher[progress_id_teacher]
            
            target = (latents_ - inputs_shared["latents"]) / (sigma_ - sigma)
            loss = loss + torch.nn.functional.mse_loss(noise_pred.float(), target.float()) * pipe.scheduler.training_weight(timestep)
        return loss
    
    def compute_regularization(self, pipe: BasePipeline, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, num_inference_steps, cfg_scale):
        inputs_shared["latents"] = trajectory_teacher[0]
        pipe.scheduler.set_timesteps(num_inference_steps)
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}
        for progress_id, timestep in enumerate(pipe.scheduler.timesteps):
            timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            noise_pred = pipe.cfg_guided_model_fn(
                pipe.model_fn, cfg_scale,
                inputs_shared, inputs_posi, inputs_nega,
                **models, timestep=timestep, progress_id=progress_id
            )
            inputs_shared["latents"] = pipe.step(pipe.scheduler, progress_id=progress_id, noise_pred=noise_pred.detach(), **inputs_shared)

        image_pred = pipe.vae_decoder(inputs_shared["latents"])
        image_real = pipe.vae_decoder(trajectory_teacher[-1])
        loss = self.loss_fn(image_pred.float(), image_real.float())
        return loss

    def forward(self, pipe: BasePipeline, inputs_shared, inputs_posi, inputs_nega):
        if not self.initialized:
            self.initialize(pipe.device)
        with torch.no_grad():
            pipe.scheduler.set_timesteps(8)
            timesteps_teacher, trajectory_teacher = self.fetch_trajectory(inputs_shared["teacher"], pipe.scheduler.timesteps, inputs_shared, inputs_posi, inputs_nega, 50, 2)
            timesteps_teacher = timesteps_teacher.to(dtype=pipe.torch_dtype, device=pipe.device)
        loss_1 = self.align_trajectory(pipe, timesteps_teacher, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss_2 = self.compute_regularization(pipe, trajectory_teacher, inputs_shared, inputs_posi, inputs_nega, 8, 1)
        loss = loss_1 + loss_2
        return loss
