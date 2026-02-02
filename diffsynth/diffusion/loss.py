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
    
    loss_geo = 0.0
    
    # 2. Geometry Branch Supervision
    
    # --- A. Depth Loss ---
    # L_depth = alpha * L_TGM + beta * L_frame (following Video Depth Anything)
    if hasattr(pipe.dit, 'last_depth_output') and pipe.dit.last_depth_output is not None:
        pred_depth = pipe.dit.last_depth_output  # [B, T, 1, H, W] from DPTHead3D
        pipe.dit.last_depth_output = None
        
        gt_depth = inputs.get("gt_depth", None)
        if gt_depth is not None:
            # Ensure shape: both should be [B, T, 1, H, W]
            if gt_depth.ndim == 4:  # [B, T, H, W] -> [B, T, 1, H, W]
                gt_depth = gt_depth.unsqueeze(2)
            
            # Permute to [B, 1, T, H, W] for easier temporal gradient computation
            pred_depth = pred_depth.permute(0, 2, 1, 3, 4)  # [B, 1, T, H, W]
            gt_depth = gt_depth.permute(0, 2, 1, 3, 4)      # [B, 1, T, H, W]
             
            # Interpolate GT to Pred resolution if needed
            if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth.flatten(0, 2), size=pred_depth.shape[-2:], mode='bilinear', align_corners=True
                ).view(*gt_depth.shape[:3], *pred_depth.shape[-2:])
            if pred_depth.shape[2] != gt_depth.shape[2]:  # Temporal
                gt_depth = torch.nn.functional.interpolate(
                    gt_depth, size=pred_depth.shape[2:], mode='nearest'
                )

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
            # Ensure shape: [B, T, 3, H, W]
            if gt_points.ndim == 4:  # [B*T, 3, H, W]
                b = pts_pred.shape[0]
                t = pts_pred.shape[1]
                gt_points = gt_points.view(b, t, 3, *gt_points.shape[-2:])
             
            # Resize GT if needed
            if pts_pred.shape[-2:] != gt_points.shape[-2:]:
                b, t, c, h, w = gt_points.shape
                gt_points = torch.nn.functional.interpolate(
                    gt_points.flatten(0, 1), size=pts_pred.shape[-2:], mode='bilinear', align_corners=True
                ).view(b, t, c, *pts_pred.shape[-2:])
            
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
    # L_camera = Robust Huber Loss on 9D camera parameters
    if hasattr(pipe.dit, 'last_camera_output') and pipe.dit.last_camera_output is not None:
        pred_cam = pipe.dit.last_camera_output  # [B, T, 9]
        pipe.dit.last_camera_output = None
        
        gt_cam = inputs.get("pose_params", None)  # [B, T, 9] expected
        if gt_cam is not None:
            # Interpolate to match lengths if needed
            if pred_cam.shape[1] != gt_cam.shape[1]:
                pred_cam_tp = pred_cam.transpose(1, 2)  # [B, 9, T]
                gt_len = gt_cam.shape[1]
                pred_cam_tp = torch.nn.functional.interpolate(
                    pred_cam_tp, size=gt_len, mode='linear', align_corners=True
                )
                pred_cam = pred_cam_tp.transpose(1, 2)
            
            # Robust Huber Loss (less sensitive to outliers)
            loss_cam = torch.nn.functional.huber_loss(pred_cam, gt_cam, delta=1.0)
            
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
