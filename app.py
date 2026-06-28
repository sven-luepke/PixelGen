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
        # self.denoiser.compile()

    def __del__(self):
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
    # demo.launch(server_name="0.0.0.0", server_port=23231)
    demo.launch(share=True, server_name="0.0.0.0", server_port=23231)
