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


def instantiate_class(config):
    kwargs = config.get("init_args", {})
    class_module, class_name = config["class_path"].rsplit(".", 1)
    module = __import__(class_module, fromlist=[class_name])
    args_class = getattr(module, class_name)
    return args_class(**kwargs)

def load_model(weight_dict, denoiser):
    prefix = "ema_denoiser."
    for k, v in denoiser.state_dict().items():
        try:
            v.copy_(weight_dict["state_dict"][prefix + k])
        except:
            print(f"Failed to copy {prefix + k} to denoiser weight")
    return denoiser


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

    def _prepare_input_image(self, input_image, num_images, image_height, image_width):
        if not isinstance(input_image, Image.Image):
            input_image = Image.fromarray(input_image)
        image = input_image.convert("RGB")
        image = ImageOps.fit(image, (image_width, image_height), method=Image.Resampling.LANCZOS)
        image = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1)
        image = uint82fp(image).unsqueeze(0).repeat(num_images, 1, 1, 1)
        return self.vae.encode(image.to("cuda"))

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

    @torch.no_grad()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def __call__(self, input_image, noise_level, y, neg_prompt, num_images, seed, image_height, image_width, num_steps, guidance, timeshift, order):
        num_images = int(num_images)
        seed = int(seed)
        image_height = int(image_height)
        image_width = int(image_width)
        num_steps = int(num_steps)
        order = int(order)
        diffusion_sampler = AdamLMSamplerJiT(
            order=order,
            scheduler=LinearScheduler(),
            guidance_fn=simple_guidance_fn,
            num_steps=num_steps,
            guidance=guidance,
            timeshift=timeshift
        )
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

        return images, animations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs_t2i/sft_res512.yaml")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--model_id", type=str, default="MCG-NJU/PixNerd-XXL-P16-T2I")
    parser.add_argument("--ckpt_path", type=str, default="models")

    args = parser.parse_args()
    if not os.path.exists(args.ckpt_path):
        snapshot_download(repo_id=args.model_id, local_dir=args.ckpt_path)
        ckpt_path = os.path.join(args.ckpt_path, "model.ckpt")
    else:
        ckpt_path = args.ckpt_path

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
                image_height = gr.Slider(minimum=128, maximum=1024, step=32, label="image height", value=512)
                image_width = gr.Slider(minimum=128, maximum=1024, step=32, label="image width", value=512)
                num_images = gr.Slider(minimum=1, maximum=4, step=1, label="num images", value=4)
                input_image = gr.Image(label="input image", type="pil")
                noise_level = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="SDEdit noise level", value=0.5)
                label = gr.Textbox(label="positive prompt", value="A beautiful woman.")
                neg_label = gr.Textbox(label="negative prompt", value="Unrealistic, JPEG artifacts.")
                seed = gr.Slider(minimum=0, maximum=1000000, step=1, label="seed", value=0)
                timeshift = gr.Slider(minimum=0.1, maximum=5.0, step=0.1, label="timeshift", value=3.0)
                order = gr.Slider(minimum=1, maximum=4, step=1, label="order", value=2)
            with gr.Column(scale=2):
                btn = gr.Button("Generate")
                output_sample = gr.Gallery(label="Images", columns=2, rows=2)
            with gr.Column(scale=2):
                output_trajs = gr.Gallery(label="Trajs of Diffusion", columns=2, rows=2)

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
                  ], outputs=[output_sample, output_trajs])
    # demo.launch(server_name="0.0.0.0", server_port=23231)
    demo.launch(share=True, server_name="0.0.0.0", server_port=23231)
