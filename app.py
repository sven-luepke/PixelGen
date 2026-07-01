#   vae:
#     class_path: src.models.vae.LatentVAE
#     init_args:
#       precompute: true
#       weight_path: /mnt/bn/wangshuai6/models/sd-vae-ft-ema/
#   denoiser:
#     class_path: src.models.denoiser.decoupled_improved_dit.DDT
#     init_args:
#       in_channels: 4
#       patch_size: 2
#       num_groups: 16
#       hidden_size: &hidden_dim 1152
#       num_blocks: 28
#       num_encoder_blocks: 22
#       num_classes: 1000
#   conditioner:
#     class_path: src.models.conditioner.LabelConditioner
#     init_args:
#       null_class: 1000
#   diffusion_sampler:
#     class_path: src.diffusion.stateful_flow_matching.sampling.EulerSampler
#     init_args:
#       num_steps: 250
#       guidance: 3.0
#       state_refresh_rate: 1
#       guidance_interval_min: 0.3
#       guidance_interval_max: 1.0
#       timeshift: 1.0
#       last_step: 0.04
#       scheduler: *scheduler
#       w_scheduler: src.diffusion.stateful_flow_matching.scheduling.LinearScheduler
#       guidance_fn: src.diffusion.base.guidance.simple_guidance_fn
#       step_fn: src.diffusion.stateful_flow_matching.sampling.ode_step_fn
import random
import os
import torch
import torch.nn.functional as F
import argparse
import numpy as np
import zipfile
from pathlib import Path
from omegaconf import OmegaConf
from src.models.autoencoder.base import fp2uint8, uint82fp
from src.diffusion.base.guidance import simple_guidance_fn
from src.diffusion.flow_matching.adam_sampling import AdamLMSamplerJiT
from src.diffusion.flow_matching.scheduling import LinearScheduler
from src.diffusion.pre_integral import lagrange_preint
from PIL import Image, ImageOps
import gradio as gr
import tempfile
from huggingface_hub import snapshot_download
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large


VIDEO_RESOLUTION = 512
DEFAULT_VIDEO_FPS = 24.0


def instantiate_class(config):
    kwargs = config.get("init_args", {})
    class_module, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)

def load_model(weight_dict, denoiser):
    prefix = "ema_denoiser."
    state_dict = weight_dict["state_dict"]
    failures = []
    for k, v in denoiser.state_dict().items():
        ckpt_key = prefix + k
        if ckpt_key not in state_dict:
            failures.append(f"missing {ckpt_key}")
            continue
        if state_dict[ckpt_key].shape != v.shape:
            failures.append(
                f"shape mismatch {ckpt_key}: checkpoint {tuple(state_dict[ckpt_key].shape)} != model {tuple(v.shape)}"
            )
            continue
        v.copy_(state_dict[ckpt_key])
    if failures:
        detail = "\n".join(failures[:10])
        if len(failures) > 10:
            detail += f"\n... and {len(failures) - 10} more"
        raise ValueError(f"Failed to load denoiser weights:\n{detail}")
    return denoiser


def resolve_ckpt_path(ckpt_path, model_id):
    path = Path(ckpt_path)
    if path.is_file():
        return str(path)
    if path.is_dir():
        model_ckpt = path / "model.ckpt"
        if model_ckpt.exists():
            return str(model_ckpt)
        ckpt_files = sorted(path.glob("*.ckpt"))
        if ckpt_files:
            return str(ckpt_files[0])
        raise FileNotFoundError(f"No .ckpt file found in {path}")

    local_dir = path.parent if path.suffix == ".ckpt" else path
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=model_id, local_dir=str(local_dir))
    if path.suffix == ".ckpt" and path.exists():
        return str(path)
    if (local_dir / "model.ckpt").exists():
        return str(local_dir / "model.ckpt")
    ckpt_files = sorted(local_dir.glob("*.ckpt"))
    if ckpt_files:
        return str(ckpt_files[0])
    raise FileNotFoundError(f"No .ckpt file found after downloading {model_id} to {local_dir}")


def get_imageio():
    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise gr.Error("Video support requires imageio[ffmpeg]. Install requirements.txt and restart the app.") from exc
    return imageio


class Pipeline:
    def __init__(self, vae, denoiser, conditioner, resolution):
        self.vae = vae.cuda()
        self.denoiser = denoiser.cuda()
        self.conditioner = conditioner.cuda()
        self.conditioner.compile()
        self.resolution = resolution
        self.tmp_dir = tempfile.TemporaryDirectory(prefix="traj_gifs_")
        self.flow_model = None
        self.flow_transforms = None
        # self.denoiser.compile()

    def __del__(self):
        if hasattr(self, "tmp_dir") and self.tmp_dir is not None:
            self.tmp_dir.cleanup()

    def _decode_images(self, samples):
        samples = self.vae.decode(samples)
        samples = fp2uint8(samples)
        samples = samples.permute(0, 2, 3, 1).cpu().numpy()
        images = []
        for i in range(len(samples)):
            image = Image.fromarray(samples[i])
            images.append(image)
        return images

    def _decode_trajs(self, trajs):
        cat_trajs = torch.stack(trajs, dim=0).permute(1, 0, 2, 3, 4)
        animations = []
        for i in range(cat_trajs.shape[0]):
            frames = self._decode_images(cat_trajs[i])
            gif_filename = f"{random.randint(0, 100000)}.gif"
            gif_path = os.path.join(self.tmp_dir.name, gif_filename)
            frames[0].save(
                gif_path,
                format="GIF",
                append_images=frames[1:],
                save_all=True,
                duration=200,
                loop=0
            )
            animations.append(gif_path)
        return animations

    def _save_images(self, images, run_id):
        image_paths = []
        for i, image in enumerate(images):
            image_path = os.path.join(self.tmp_dir.name, f"{run_id}_image_{i + 1}.png")
            image.save(image_path)
            image_paths.append(image_path)
        return image_paths

    def _write_zip(self, paths, run_id, name):
        zip_path = os.path.join(self.tmp_dir.name, f"{run_id}_{name}.zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, arcname=os.path.basename(path))
        return zip_path

    def _prepare_input_image(self, input_image, num_images, image_height, image_width):
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)
        image = input_image.convert("RGB")
        image = ImageOps.fit(image, (image_width, image_height), method=Image.Resampling.LANCZOS)
        image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        image = uint82fp(image).unsqueeze(0).repeat(num_images, 1, 1, 1)
        return self.vae.encode(image.to("cuda"))

    def _prepare_image_batch(self, images):
        tensors = []
        for image in images:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")
            image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
            tensors.append(uint82fp(image))
        batch = torch.stack(tensors, dim=0).to("cuda")
        return self.vae.encode(batch)

    def _coerce_video_path(self, input_video):
        if input_video is None:
            return None
        if isinstance(input_video, str):
            return input_video
        if isinstance(input_video, dict):
            for key in ("video", "name", "path"):
                value = input_video.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    nested_value = value.get("path") or value.get("name")
                    if isinstance(nested_value, str):
                        return nested_value
        return input_video

    def _read_video_frames(self, input_video, max_frames):
        video_path = self._coerce_video_path(input_video)
        if not video_path:
            raise gr.Error("Upload a video before generating.")

        imageio = get_imageio()
        frames = []
        reader = imageio.get_reader(video_path)
        try:
            try:
                metadata = reader.get_meta_data()
            except Exception:
                metadata = {}
            fps = float(metadata.get("fps") or DEFAULT_VIDEO_FPS)
            if fps <= 0:
                fps = DEFAULT_VIDEO_FPS
            for i, frame in enumerate(reader):
                if i >= max_frames:
                    break
                image = Image.fromarray(frame).convert("RGB")
                image = ImageOps.fit(
                    image,
                    (VIDEO_RESOLUTION, VIDEO_RESOLUTION),
                    method=Image.Resampling.LANCZOS,
                )
                frames.append(image)
        finally:
            reader.close()

        if not frames:
            raise gr.Error("Could not read any frames from the uploaded video.")
        return frames, fps

    def _write_video(self, frames, fps, run_id):
        imageio = get_imageio()
        video_path = os.path.join(self.tmp_dir.name, f"{run_id}_edited.mp4")
        writer = imageio.get_writer(
            video_path,
            fps=fps,
            codec="libx264",
            quality=8,
            macro_block_size=16,
            ffmpeg_params=["-pix_fmt", "yuv420p", "-an"],
        )
        try:
            for frame in frames:
                writer.append_data(np.array(frame.convert("RGB"), copy=True))
        finally:
            writer.close()
        return video_path

    def _get_flow_model(self):
        if self.flow_model is None:
            weights = Raft_Large_Weights.DEFAULT
            self.flow_transforms = weights.transforms()
            self.flow_model = raft_large(weights=weights, progress=True).cuda().eval()
        return self.flow_model, self.flow_transforms

    def _frames_to_flow_tensor(self, frames):
        tensors = []
        for image in frames:
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)
            image = image.convert("RGB")
            array = np.array(image, copy=True)
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append(tensor)
        return torch.stack(tensors, dim=0).to("cuda")

    def _estimate_forward_flows(self, frames, progress=None):
        if len(frames) < 2:
            return []

        flow_model, flow_transforms = self._get_flow_model()
        tensors = self._frames_to_flow_tensor(frames)
        forward_flows = []
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            for i in range(1, len(frames)):
                if progress is not None:
                    progress(
                        0.05 + 0.35 * i / max(1, len(frames) - 1),
                        desc=f"Estimating optical flow {i}/{len(frames) - 1}",
                    )
                previous, current = flow_transforms(tensors[i - 1:i], tensors[i:i + 1])
                forward = flow_model(previous.float(), current.float())[-1]
                forward_flows.append(forward[0].float())
        return forward_flows

    def _make_pixel_grid(self, height, width, device, dtype):
        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        return x, y

    def _coords_to_grid(self, coords):
        _, height, width = coords.shape
        grid_x = 2.0 * coords[0] / max(width - 1, 1) - 1.0
        grid_y = 2.0 * coords[1] / max(height - 1, 1) - 1.0
        return torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    def _sample_tensor_at_coords(self, tensor, coords, padding_mode="zeros"):
        grid = self._coords_to_grid(coords.to(device=tensor.device, dtype=tensor.dtype))
        return F.grid_sample(
            tensor.unsqueeze(0),
            grid,
            mode="bilinear",
            padding_mode=padding_mode,
            align_corners=True,
        )[0]

    def _resize_flow(self, flow, height, width, device, dtype):
        flow = flow.to(device=device, dtype=dtype)
        if flow.shape[-2:] != (height, width):
            scale_y = height / flow.shape[-2]
            scale_x = width / flow.shape[-1]
            flow = F.interpolate(
                flow.unsqueeze(0),
                size=(height, width),
                mode="bilinear",
                align_corners=True,
            )[0]
            flow[0] *= scale_x
            flow[1] *= scale_y
        return flow

    def _flow_sample_coords(self, flow, height, width, device, dtype):
        flow = self._resize_flow(flow, height, width, device, dtype)
        x, y = self._make_pixel_grid(height, width, device, dtype)
        return torch.stack((x + flow[0], y + flow[1]), dim=0)

    def _warp_tensor_with_flow(self, tensor, backward_flow, padding_mode="zeros"):
        _, height, width = tensor.shape
        flow = self._resize_flow(backward_flow, height, width, tensor.device, tensor.dtype)
        x, y = self._make_pixel_grid(height, width, tensor.device, tensor.dtype)
        coords = torch.stack((x + flow[0], y + flow[1]), dim=0)
        return self._sample_tensor_at_coords(tensor, coords, padding_mode=padding_mode)

    def _valid_coord_mask(self, coords):
        _, height, width = coords.shape
        return (
            (coords[0] >= 0.0)
            & (coords[0] <= width - 1)
            & (coords[1] >= 0.0)
            & (coords[1] <= height - 1)
        )

    def _regaussianize_noise(self, noise):
        mean = noise.mean(dim=(1, 2), keepdim=True)
        std = noise.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        return (noise - mean) / std

    def _noise_to_state(self, noise):
        _, height, width = noise.shape
        state = torch.zeros(
            (noise.shape[0] + 3, height, width),
            device=noise.device,
            dtype=noise.dtype,
        )
        state[2] = 1.0
        state[3:] = noise
        return state

    def _state_to_noise(self, state):
        return state[3:]

    def _scatter_add_flat(self, values, indices, height, width):
        channels = values.shape[0]
        output = torch.zeros(
            (channels, height * width),
            device=values.device,
            dtype=values.dtype,
        )
        output.index_add_(1, indices, values)
        return output.view(channels, height, width)

    def _nearest_remap_state(self, state, flow):
        _, height, width = state.shape
        flow = self._resize_flow(flow, height, width, state.device, state.dtype)
        x, y = self._make_pixel_grid(height, width, state.device, state.dtype)
        source_x = (x - flow[0]).round().long()
        source_y = (y - flow[1]).round().long()
        valid = (
            (source_x >= 0)
            & (source_x < width)
            & (source_y >= 0)
            & (source_y < height)
        )
        source_x = source_x.clamp(0, width - 1)
        source_y = source_y.clamp(0, height - 1)
        remapped = state[:, source_y, source_x]
        init = self._init_state_like(state)
        return torch.where(valid.unsqueeze(0), remapped, init)

    def _init_state_like(self, state):
        init = torch.zeros_like(state)
        init[2] = 1.0
        return init

    def _scatter_add_offsets(self, values, offsets):
        _, height, width = values.shape
        x, y = self._make_pixel_grid(height, width, values.device, values.dtype)
        target_x = (x + offsets[0]).long()
        target_y = (y + offsets[1]).long()
        valid = (
            (target_x >= 0)
            & (target_x < width)
            & (target_y >= 0)
            & (target_y < height)
        )
        target_x = target_x.clamp(0, width - 1)
        target_y = target_y.clamp(0, height - 1)
        flat_indices = (target_y * width + target_x).reshape(-1)
        valid_flat = valid.reshape(-1)
        if valid_flat.any():
            return self._scatter_add_flat(
                values.reshape(values.shape[0], -1)[:, valid_flat],
                flat_indices[valid_flat],
                height,
                width,
            )
        return torch.zeros_like(values)

    def _regaussianize_duplicate_samples(self, noise, generator):
        channels, height, width = noise.shape
        _, inverse, counts = torch.unique(
            noise[:1].reshape(-1),
            sorted=False,
            return_inverse=True,
            return_counts=True,
        )
        num_groups = counts.numel()
        counts = counts.to(device=noise.device, dtype=noise.dtype)
        counts_per_pixel = counts[inverse].view(1, height, width).clamp_min(1.0)

        foreign = torch.randn(
            noise.shape,
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        ).to(noise.device)
        foreign_flat = foreign.view(channels, -1)
        group_sums = torch.zeros(
            (channels, num_groups),
            device=noise.device,
            dtype=noise.dtype,
        )
        group_sums.index_add_(1, inverse, foreign_flat)
        group_means = group_sums[:, inverse].view(channels, height, width) / counts_per_pixel
        return noise / counts_per_pixel.sqrt() + foreign - group_means, counts_per_pixel

    def _warp_state_with_flow(self, state, flow, generator):
        _, height, width = state.shape
        init = self._init_state_like(state)
        flow = self._resize_flow(flow, height, width, state.device, state.dtype)

        pre_expand = self._nearest_remap_state(state, flow)
        pre_expand[2] = torch.where(
            pre_expand[2] == 0,
            torch.ones_like(pre_expand[2]),
            pre_expand[2],
        )

        pre_shrink = state.clone()
        pre_shrink[:2] += flow
        x, y = self._make_pixel_grid(height, width, state.device, state.dtype)
        pos_x = (x + pre_shrink[0]).round().long()
        pos_y = (y + pre_shrink[1]).round().long()
        in_bounds = (
            (pos_x >= 0)
            & (pos_x < width)
            & (pos_y >= 0)
            & (pos_y < height)
        )
        pre_shrink = torch.where(in_bounds.unsqueeze(0), pre_shrink, init)
        scatter_offsets = pre_shrink[:2].round()
        pre_shrink[:2] -= scatter_offsets

        # Algorithm 1 uses the contraction mask to choose shrink where any particle lands.
        shrink_mask = self._scatter_add_offsets(
            torch.ones((1, height, width), device=state.device, dtype=state.dtype),
            scatter_offsets,
        ) > 0
        pre_expand = torch.where(shrink_mask, init, pre_expand)

        concat = torch.cat((pre_shrink, pre_expand), dim=2)
        concat[3:], counts = self._regaussianize_duplicate_samples(concat[3:], generator)
        concat[2:3] = concat[2:3] / counts
        concat[2:3] = torch.nan_to_num(concat[2:3], nan=0.0, posinf=0.0, neginf=0.0)

        pre_shrink, expand = torch.chunk(concat, chunks=2, dim=2)
        shrink = torch.empty_like(pre_shrink)
        shrink[2:3] = self._scatter_add_offsets(pre_shrink[2:3], scatter_offsets)
        shrink[:2] = self._scatter_add_offsets(pre_shrink[:2] * pre_shrink[2:3], scatter_offsets)
        shrink[:2] = shrink[:2] / shrink[2:3].clamp_min(1e-6)
        density_sq = self._scatter_add_offsets(pre_shrink[2:3] ** 2, scatter_offsets)
        shrink[3:] = self._scatter_add_offsets(pre_shrink[3:] * pre_shrink[2:3], scatter_offsets)
        shrink[3:] = shrink[3:] / density_sq.sqrt().clamp_min(1e-6)

        output = torch.where(shrink_mask, shrink, expand)
        output[2] = output[2] / output[2].mean().clamp_min(1e-6)
        output[2] = output[2].clamp_min(1e-5).pow(0.9999)
        return output

    def _make_flow_warped_noise(self, frames, seed, progress=None):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        first_noise = torch.randn(
            (3, VIDEO_RESOLUTION, VIDEO_RESOLUTION),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        ).to("cuda")
        first_noise = self._regaussianize_noise(first_noise)

        if len(frames) == 1:
            return first_noise.unsqueeze(0), []

        forward_flows = self._estimate_forward_flows(frames, progress)
        noises = [first_noise]
        state = self._noise_to_state(first_noise)
        for i, forward_flow in enumerate(forward_flows, start=1):
            if progress is not None:
                progress(
                    0.40 + 0.20 * i / max(1, len(forward_flows)),
                    desc=f"Warping noise {i}/{len(forward_flows)}",
                )
            state = self._warp_state_with_flow(state, forward_flow, generator)
            noises.append(self._state_to_noise(state))
        return torch.stack(noises, dim=0), forward_flows

    def _noise_to_image(self, noise):
        noise = noise.detach().float().cpu()
        image = torch.clamp(noise[:3] / 4.0 + 0.5, 0.0, 1.0)
        image = (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        return Image.fromarray(image)

    def _write_noise_video(self, noises, fps, run_id):
        frames = [self._noise_to_image(noise) for noise in noises]
        return self._write_video(frames, fps, f"{run_id}_noise")

    def _hsv_to_rgb(self, hsv):
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]
        i = np.floor(h * 6.0).astype(np.int32)
        f = h * 6.0 - i
        p = v * (1.0 - s)
        q = v * (1.0 - f * s)
        t = v * (1.0 - (1.0 - f) * s)
        i = i % 6

        rgb = np.zeros_like(hsv)
        masks = [i == index for index in range(6)]
        rgb[masks[0]] = np.stack((v, t, p), axis=-1)[masks[0]]
        rgb[masks[1]] = np.stack((q, v, p), axis=-1)[masks[1]]
        rgb[masks[2]] = np.stack((p, v, t), axis=-1)[masks[2]]
        rgb[masks[3]] = np.stack((p, q, v), axis=-1)[masks[3]]
        rgb[masks[4]] = np.stack((t, p, v), axis=-1)[masks[4]]
        rgb[masks[5]] = np.stack((v, p, q), axis=-1)[masks[5]]
        return rgb

    def _flow_to_image(self, flow, max_magnitude):
        flow = flow.detach().float().cpu().numpy()
        dx = flow[0]
        dy = flow[1]
        angle = np.arctan2(dy, dx)
        magnitude = np.sqrt(dx * dx + dy * dy)

        hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.float32)
        hsv[..., 0] = (angle + np.pi) / (2.0 * np.pi)
        hsv[..., 1] = 1.0
        hsv[..., 2] = np.clip(magnitude / max(max_magnitude, 1e-6), 0.0, 1.0)
        rgb = (self._hsv_to_rgb(hsv) * 255.0).astype(np.uint8)
        return Image.fromarray(rgb)

    def _write_flow_video(self, flows, fps, run_id):
        if not flows:
            return None

        max_magnitude = 0.0
        for flow in flows:
            magnitude = torch.linalg.vector_norm(flow.float(), dim=0).max().item()
            max_magnitude = max(max_magnitude, magnitude)

        frames = [self._flow_to_image(flows[0], max_magnitude)]
        frames.extend(self._flow_to_image(flow, max_magnitude) for flow in flows)
        flow_path = self._write_video(frames, fps, f"{run_id}_flow")
        return flow_path

    def _build_truncated_solver_coeffs(self, timesteps, order, lms_transform_fn):
        solver_coeffs = [[] for _ in range(len(timesteps) - 1)]
        cpu_timesteps = timesteps.detach().cpu()
        for i in range(len(cpu_timesteps) - 1):
            pre_vs = [1.0] * (i + 1)
            pre_ts = lms_transform_fn(cpu_timesteps[:i + 1])
            int_t_start = lms_transform_fn(cpu_timesteps[i])
            int_t_end = lms_transform_fn(cpu_timesteps[i + 1])
            _, coeffs = lagrange_preint(min(order, i + 1), pre_vs, pre_ts, int_t_start, int_t_end)
            solver_coeffs[i] = coeffs
        return solver_coeffs

    def _sample_from_t(self, sampler, x, start_t, condition, uncondition):
        if start_t <= 1e-6:
            return sampler(self.denoiser, x, condition, uncondition, return_x_trajs=True)
        if start_t >= 1.0 - 1e-6:
            return x, [x]

        batch_size = x.shape[0]
        full_timesteps = sampler.timesteps.to(x.device, x.dtype)
        start_t = torch.tensor(start_t, device=x.device, dtype=x.dtype)
        remaining = full_timesteps[full_timesteps > start_t]
        if len(remaining) == 0 or remaining[-1] < 1.0:
            remaining = torch.cat([remaining, torch.ones(1, device=x.device, dtype=x.dtype)])
        timesteps = torch.cat([start_t.view(1), remaining])
        solver_coeffs = self._build_truncated_solver_coeffs(
            timesteps,
            sampler.order,
            sampler.lms_transform_fn,
        )

        cfg_condition = torch.cat([uncondition, condition], dim=0)
        pred_trajectory = []
        x_trajectory = [x]
        t_cur = timesteps[0].repeat(batch_size)

        for i, t_next in enumerate(timesteps[1:]):
            cfg_x = torch.cat([x, x], dim=0)
            cfg_t = t_cur.repeat(2)
            out = self.denoiser(cfg_x, cfg_t, cfg_condition)
            out = (out - cfg_x) / (1.0 - cfg_t.view(-1, 1, 1, 1)).clamp_min(5e-2)
            if t_cur[0] > sampler.guidance_interval_min and t_cur[0] < sampler.guidance_interval_max:
                out = sampler.guidance_fn(out, sampler.guidance)
            else:
                out = sampler.guidance_fn(out, 1.0)

            pred_trajectory.append(out)
            v = torch.zeros_like(out)
            order = len(solver_coeffs[i])
            for j in range(order):
                v += solver_coeffs[i][j] * pred_trajectory[-order:][j]
            x = sampler.step_fn(x, v, t_next - t_cur[0], s=0, w=0)
            t_cur = t_next.repeat(batch_size)
            x_trajectory.append(x)

        return x_trajectory[-1], x_trajectory

    def _build_sampler(self, num_steps, guidance, timeshift, order):
        return AdamLMSamplerJiT(
            order=int(order),
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=int(num_steps),
            guidance=guidance,
            timeshift=timeshift
        )

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(self, input_image, noise_level, y, neg_prompt, num_images, seed, image_height, image_width, num_steps, guidance, timeshift, order):
        num_images = int(num_images)
        seed = int(seed)
        image_height = int(image_height)
        image_width = int(image_width)
        num_steps = int(num_steps)
        order = int(order)
        diffusion_sampler = self._build_sampler(num_steps, guidance, timeshift, order)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image_height = image_height // 32 * 32
        image_width = image_width // 32 * 32
        self.denoiser.decoder_patch_scaling_h = image_height / 512
        self.denoiser.decoder_patch_scaling_w = image_width / 512
        xT = torch.randn((num_images, 3, image_height, image_width), device="cpu", dtype=torch.float32,
                         generator=generator)
        xT = xT.to("cuda")
        with torch.no_grad():
            condition, uncondition = self.conditioner([y,]*num_images, {"negative_prompt": neg_prompt})

        if input_image is not None:
            image_x0 = self._prepare_input_image(input_image, num_images, image_height, image_width)
            noise_level = max(0.0, min(1.0, noise_level))
            start_t = 1.0 - noise_level
            xT = start_t * image_x0 + noise_level * xT
            samples, trajs = self._sample_from_t(diffusion_sampler, xT, start_t, condition, uncondition)
        else:
            samples, trajs = diffusion_sampler(self.denoiser, xT, condition, uncondition, return_x_trajs=True)

        images = self._decode_images(samples)
        animations = self._decode_trajs(trajs)
        run_id = f"{seed}_{random.randint(0, 100000)}"
        image_paths = self._save_images(images, run_id)
        image_zip = self._write_zip(image_paths, run_id, "images")
        animation_zip = self._write_zip(animations, run_id, "trajs")

        return image_paths, animations, image_zip, animation_zip

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def edit_video(self, input_video, max_frames, video_batch_size, noise_level, y, neg_prompt, seed, num_steps, guidance, timeshift, order, progress=gr.Progress()):
        max_frames = int(max_frames)
        video_batch_size = max(1, int(video_batch_size))
        seed = int(seed)
        max_frames = max(1, max_frames)
        noise_level = max(0.0, min(1.0, noise_level))
        start_t = 1.0 - noise_level

        progress(0, desc="Reading video")
        frames, fps = self._read_video_frames(input_video, max_frames)
        total_frames = len(frames)

        diffusion_sampler = self._build_sampler(num_steps, guidance, timeshift, order)
        self.denoiser.decoder_patch_scaling_h = 1.0
        self.denoiser.decoder_patch_scaling_w = 1.0

        generator = torch.Generator(device="cpu").manual_seed(seed)
        shared_noise = torch.randn(
            (1, 3, VIDEO_RESOLUTION, VIDEO_RESOLUTION),
            device="cpu",
            dtype=torch.float32,
            generator=generator,
        ).to("cuda")

        edited_frames = []
        for start in range(0, total_frames, video_batch_size):
            end = min(start + video_batch_size, total_frames)
            batch_frames = frames[start:end]
            batch_size = len(batch_frames)
            progress(start / total_frames, desc=f"Editing frames {start + 1}-{end} of {total_frames}")

            image_x0 = self._prepare_image_batch(batch_frames)
            xT = start_t * image_x0 + noise_level * shared_noise.repeat(batch_size, 1, 1, 1)

            condition, uncondition = self.conditioner([y,] * batch_size, {"negative_prompt": neg_prompt})
            samples, _ = self._sample_from_t(diffusion_sampler, xT, start_t, condition, uncondition)
            edited_frames.extend(self._decode_images(samples))

        progress(0.98, desc="Writing video")
        run_id = f"{seed}_{random.randint(0, 100000)}"
        video_path = self._write_video(edited_frames, fps, run_id)
        progress(1.0, desc="Done")
        return video_path

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def edit_video_warped(self, input_video, max_frames, video_batch_size, noise_level, y, neg_prompt, seed, num_steps, guidance, timeshift, order, progress=gr.Progress()):
        max_frames = int(max_frames)
        video_batch_size = max(1, int(video_batch_size))
        seed = int(seed)
        max_frames = max(1, max_frames)
        noise_level = max(0.0, min(1.0, noise_level))
        start_t = 1.0 - noise_level

        progress(0, desc="Reading video")
        frames, fps = self._read_video_frames(input_video, max_frames)
        total_frames = len(frames)

        diffusion_sampler = self._build_sampler(num_steps, guidance, timeshift, order)
        self.denoiser.decoder_patch_scaling_h = 1.0
        self.denoiser.decoder_patch_scaling_w = 1.0

        with torch.autocast(device_type="cuda", enabled=False):
            warped_noises, flows = self._make_flow_warped_noise(frames, seed, progress)

        edited_frames = []
        for start in range(0, total_frames, video_batch_size):
            end = min(start + video_batch_size, total_frames)
            batch_frames = frames[start:end]
            batch_size = len(batch_frames)
            progress(0.60 + 0.35 * start / total_frames, desc=f"Editing frames {start + 1}-{end} of {total_frames}")

            image_x0 = self._prepare_image_batch(batch_frames)
            noise = warped_noises[start:end].to(image_x0.device, image_x0.dtype)
            xT = start_t * image_x0 + noise_level * noise

            condition, uncondition = self.conditioner([y,] * batch_size, {"negative_prompt": neg_prompt})
            samples, _ = self._sample_from_t(diffusion_sampler, xT, start_t, condition, uncondition)
            edited_frames.extend(self._decode_images(samples))

        progress(0.98, desc="Writing video")
        run_id = f"{seed}_{random.randint(0, 100000)}"
        video_path = self._write_video(edited_frames, fps, run_id)
        flow_path = self._write_flow_video(flows, fps, run_id)
        noise_path = self._write_noise_video(warped_noises, fps, run_id)
        progress(1.0, desc="Done")
        return video_path, flow_path, noise_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_t2i/sft_res512.yaml")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="zehongma/PixelGen")
    parser.add_argument("--ckpt_path", type=str, default="ckpts/PixelGen_XXL_T2I.ckpt")

    args = parser.parse_args()
    ckpt_path = resolve_ckpt_path(args.ckpt_path, args.model_id)

    config = OmegaConf.load(args.config)
    vae_config = config.model.vae
    denoiser_config = config.model.denoiser
    conditioner_config = config.model.conditioner

    vae = instantiate_class(vae_config)
    denoiser = instantiate_class(denoiser_config)
    conditioner = instantiate_class(conditioner_config)


    ckpt = torch.load(ckpt_path, map_location="cpu")
    denoiser = load_model(ckpt, denoiser)
    denoiser = denoiser.cuda()
    vae = vae.cuda()
    denoiser.eval()


    pipeline = Pipeline(vae, denoiser, conditioner, args.resolution)

    with gr.Blocks() as demo:
        # gr.Markdown(f"config:{args.config}\n\n ckpt_path:{args.ckpt_path}")
        with gr.Row():
            with gr.Column(scale=1):
                num_steps = gr.Slider(minimum=1, maximum=100, step=1, label="num steps", value=25)
                guidance = gr.Slider(minimum=0.1, maximum=10.0, step=0.1, label="CFG", value=4.0)
                noise_level = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SDEdit noise level", value=0.5)
                label = gr.Textbox(label="positive prompt", value="A beautiful woman.")
                neg_label = gr.Textbox(label="negative prompt", value="Unrealistic, JPEG artifacts.")
                seed = gr.Slider(minimum=0, maximum=1000000, step=1, label="seed", value=0)
                timeshift = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, label="timeshift", value=3.0)
                order = gr.Slider(minimum=1, maximum=4, step=1, label="order", value=2)
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("Image"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                image_height = gr.Slider(minimum=128, maximum=1024, step=32, label="image height", value=512)
                                image_width = gr.Slider(minimum=128, maximum=1024, step=32, label="image width", value=512)
                                num_images = gr.Slider(minimum=1, maximum=4, step=1, label="num images", value=4)
                                input_image = gr.Image(label="input image", type="pil")
                                btn = gr.Button("Generate")
                            with gr.Column(scale=2):
                                output_sample = gr.Gallery(label="Images", columns=2, rows=2)
                                output_images_zip = gr.File(label="Download images")
                            with gr.Column(scale=2):
                                output_trajs = gr.Gallery(label="Trajs of Diffusion", columns=2, rows=2)
                                output_trajs_zip = gr.File(label="Download diffusion trajs")
                    with gr.Tab("Video"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                input_video = gr.Video(label="input video")
                                max_video_frames = gr.Slider(minimum=1, maximum=240, step=1, label="max frames", value=48)
                                video_batch_size = gr.Slider(minimum=1, maximum=16, step=1, label="video batch size", value=4)
                                video_btn = gr.Button("Generate video")
                            with gr.Column(scale=2):
                                output_video = gr.Video(label="Edited video")
                    with gr.Tab("Warped Video"):
                        with gr.Row():
                            with gr.Column(scale=1):
                                warped_input_video = gr.Video(label="input video")
                                warped_max_video_frames = gr.Slider(minimum=1, maximum=240, step=1, label="max frames", value=48)
                                warped_video_batch_size = gr.Slider(minimum=1, maximum=16, step=1, label="video batch size", value=4)
                                warped_video_btn = gr.Button("Generate warped video")
                            with gr.Column(scale=2):
                                warped_output_video = gr.Video(label="Edited video")
                            with gr.Column(scale=2):
                                warped_flow_video = gr.Video(label="Optical flow")
                            with gr.Column(scale=2):
                                warped_noise_video = gr.Video(label="Warped noise")

        btn.click(fn=pipeline,
                  inputs=[
                      input_image,
                      noise_level,
                      label,
                      neg_label,
                      num_images,
                      seed,
                      image_height,
                      image_width,
                      num_steps,
                      guidance,
                      timeshift,
                      order
                  ], outputs=[output_sample, output_trajs, output_images_zip, output_trajs_zip])
        video_btn.click(fn=pipeline.edit_video,
                        inputs=[
                            input_video,
                            max_video_frames,
                            video_batch_size,
                            noise_level,
                            label,
                            neg_label,
                            seed,
                            num_steps,
                            guidance,
                            timeshift,
                            order
                        ], outputs=[output_video])
        warped_video_btn.click(fn=pipeline.edit_video_warped,
                               inputs=[
                                   warped_input_video,
                                   warped_max_video_frames,
                                   warped_video_batch_size,
                                   noise_level,
                                   label,
                                   neg_label,
                                   seed,
                                   num_steps,
                                   guidance,
                                   timeshift,
                                   order
                               ], outputs=[warped_output_video, warped_flow_video, warped_noise_video])
    # demo.launch(server_name="0.0.0.0", server_port=23231)
    demo.launch(share=True, server_name="0.0.0.0", server_port=23231)
