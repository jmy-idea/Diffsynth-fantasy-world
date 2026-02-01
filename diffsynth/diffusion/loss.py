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
    # 1. Video Diffusion Loss
    # We ensure 'pose_params' is in inputs for FW injection
    loss_diffusion = FlowMatchSFTLoss(pipe, **inputs)
    
    # 2. Geometry Branch Supervision
    loss_geo = 0.0
    
    # Retrieve DPT outputs from the forward pass stored in the model
    # (Note: FlowMatchSFTLoss calls model_fn which triggers the forward)
    if hasattr(pipe.dit, 'last_dpt_output') and pipe.dit.last_dpt_output is not None:
        pts_pred, pts_conf = pipe.dit.last_dpt_output
        pipe.dit.last_dpt_output = None # Clear
        
        # pts_pred: [B, F, C=3, H, W]
        # gt_points: [B, F, 3, H, W] or [B, S, ...] depending on input
        gt_points = inputs.get("gt_points", None)
        
        if gt_points is not None:
            # Check resolution matching
            if pts_pred.shape[-2:] != gt_points.shape[-2:]:
                # Downsample GT or Upsample Pred? Usually Downsample GT to latents or Upsample Pred to pixels.
                # VGGT typically supervises at 2x latent resolution (from DPT).
                b, f, c, h, w = gt_points.shape
                # Flatten B, F for interpolation
                gt_points = torch.nn.functional.interpolate(
                    gt_points.view(b*f, c, h, w), 
                    size=pts_pred.shape[-2:], 
                    mode='nearest'
                ).view(b, f, c, *pts_pred.shape[-2:])
            
            # Point Map Loss (L_pmap)
            # sum(|Conf * (P - G)| + |Conf * (GradP - GradG)| - gamma * log(Conf))
            gamma = 0.1
            
            # Ensure proper broadcasting
            diff = (pts_pred - gt_points).abs()
            loss_pts = (pts_conf * diff).mean()
            
            # Gradient Loss
            def gradients(x):
                dy = x[..., 1:, :] - x[..., :-1, :]
                dx = x[..., :, 1:] - x[..., :, :-1]
                return dx, dy
                
            dx_p, dy_p = gradients(pts_pred)
            dx_g, dy_g = gradients(gt_points)
            
            # Align conf for gradients (slice to match size)
            conf_dx = pts_conf[..., :, 1:]
            conf_dy = pts_conf[..., 1:, :]
            
            loss_grad = (conf_dx * (dx_p - dx_g).abs()).mean() + \
                        (conf_dy * (dy_p - dy_g).abs()).mean()
            
            # Regularization
            loss_reg = -gamma * torch.log(pts_conf.clamp(min=1e-6)).mean()
            
            loss_geo += loss_pts + loss_grad + loss_reg

            # Depth Loss
            gt_depth = inputs.get("gt_depth", None)
            if gt_depth is not None:
                # pts_pred: [B, F, C=3, H, W] => Z is dim 2
                pred_depth = pts_pred[:, :, 2:3, :, :]
                
                if pred_depth.shape[-2:] != gt_depth.shape[-2:]:
                    b_d, f_d, c_d, h_d, w_d = gt_depth.shape
                    gt_depth = torch.nn.functional.interpolate(
                        gt_depth.view(b_d*f_d, c_d, h_d, w_d),
                        size=pred_depth.shape[-2:],
                        mode='nearest'
                    ).view(b_d, f_d, c_d, *pred_depth.shape[-2:])
                
                loss_depth = torch.nn.functional.l1_loss(pred_depth, gt_depth)
                loss_geo += loss_depth
        
        # Camera Loss
        # Retrieve camera head output
        if hasattr(pipe.dit, 'last_camera_output') and pipe.dit.last_camera_output is not None:
            pred_cam = pipe.dit.last_camera_output
            pipe.dit.last_camera_output = None # Clear
            
            gt_cam = inputs.get("pose_params", None) # Assuming input params are GT
            if gt_cam is not None:
                # Ensure dims match
                # pose_params might be (B, 9), pred_cam (B, 7) or (B, 6).
                # User set output_dim=7 in wan_video_dit.py
                # loss L2 or L1
                if pred_cam.shape == gt_cam.shape:
                    loss_cam = torch.nn.functional.mse_loss(pred_cam, gt_cam)
                    loss_geo += 3.0 * loss_cam
                elif gt_cam.shape[-1] > pred_cam.shape[-1]:
                     # Slice GT if needed, e.g. if GT is 9D but we predict 7D (quat+trans)
                     loss_cam = torch.nn.functional.mse_loss(pred_cam, gt_cam[:, :pred_cam.shape[-1]])
                     loss_geo += 3.0 * loss_cam
                    
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
