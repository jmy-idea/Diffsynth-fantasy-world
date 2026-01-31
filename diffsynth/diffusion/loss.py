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
# Fantasy World复现
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
        
        # pts_pred: [B, S, 3, H, W] (reshaped from [B*S])
        # gt_points: [B, S, 3, H, W]
        gt_points = inputs.get("gt_points", None)
        
        if gt_points is not None:
            # Resize GT to match prediction resolution if necessary
            if pts_pred.shape[-2:] != gt_points.shape[-2:]:
                b, s, c, h, w = gt_points.shape
                gt_points = torch.nn.functional.interpolate(
                    gt_points.view(b*s, c, h, w), 
                    size=pts_pred.shape[-2:], 
                    mode='nearest'
                ).view(b, s, c, *pts_pred.shape[-2:])
            
            # Mask valid points if needed (inputs['gt_valid_mask']?)
            
            # Point Map Loss (L_pmap)
            # sum(|Conf * (P - G)| + |Conf * (GradP - GradG)| - gamma * log(Conf))
            gamma = 0.1
            
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
            
            loss_geo = loss_pts + loss_grad + loss_reg
            
        # Depth Loss (Optional, using Z channel or separate GT)
        gt_depth = inputs.get("gt_depth", None)
        if gt_depth is not None:
             # Basic L1
             # Assuming Z is depth-like
             depth_pred = pts_pred[:, :, 2:3] 
             if depth_pred.shape[-2:] != gt_depth.shape[-2:]:
                  # Resize
                  pass
             # loss_depth = F.l1_loss(depth_pred, gt_depth)
             # loss_geo += loss_depth
             pass

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
