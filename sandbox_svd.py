import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import retrieve_timesteps
from diffusers.utils import load_image
from diffusers.utils.torch_utils import randn_tensor
import mediapy as media
import numpy as np
from PIL import Image

################################
################################
################################
################################
torch.set_grad_enabled(False)
################################
################################
################################
################################

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16"
)
pipe.enable_model_cpu_offload()

# Load the conditioning image
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/svd/rocket.png")
image = image.resize((1024, 576))

# Load conditioning video
video_path = '/nfs/turbo/coe-ahowens-nobackup/datasets/davis2017/DAVIS/Videos/480p/gold-fish.mp4'
video = media.read_video(video_path)
video = media.resize_video(video, (576, 1024))
video = video[:50:2]
video = torch.tensor(video)
image = video[0].permute(2,0,1)[None]
image = Image.fromarray(image.numpy()[0].transpose(1,2,0))

# Generate video
#generator = torch.manual_seed(42)
#frames = pipe(image, decode_chunk_size=8, generator=generator).frames[0]

# Convert to numpy and save
#frames_np = np.stack([np.array(frame) for frame in frames])
#media.write_video('test.mp4', frames_np, fps=7)





# Params
height = 576
width = 1024
batch_size = 1
device = 'cuda:0'
num_videos_per_prompt = 1
fps = 7
noise_aug_strength = 0.02
motion_bucket_id = 127
num_inference_steps = 25
sigmas=None
t_idx = 20

num_frames = pipe.unet.config.num_frames
decode_chunk_size = num_frames

# 3. Encode input image
image_embeddings = pipe._encode_image(image, device, num_videos_per_prompt, do_classifier_free_guidance=False)

# NOTE: Stable Video Diffusion was conditioned on fps - 1, which is why it is reduced here.
# See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
fps = fps - 1

# 4. Encode input image using VAE
image = pipe.video_processor.preprocess(image, height=height, width=width).to(device)
noise = randn_tensor(image.shape, generator=None, device=device, dtype=image.dtype)
image = image + noise_aug_strength * noise

needs_upcasting = pipe.vae.dtype == torch.float16 and pipe.vae.config.force_upcast
if needs_upcasting:
    pipe.vae.to(dtype=torch.float32)

image_latents = pipe._encode_vae_image(
    image,
    device=device,
    num_videos_per_prompt=num_videos_per_prompt,
    do_classifier_free_guidance=False,
)
image_latents = image_latents.to(image_embeddings.dtype)

# cast back to fp16 if needed
if needs_upcasting:
    pipe.vae.to(dtype=torch.float16)

# Repeat the image latents for each frame so we can concatenate them with the noise
# image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)

# 5. Get Added Time IDs
added_time_ids = pipe._get_add_time_ids(
    fps,
    motion_bucket_id,
    noise_aug_strength,
    image_embeddings.dtype,
    batch_size,
    num_videos_per_prompt,
    do_classifier_free_guidance=False,
)
added_time_ids = added_time_ids.to(device)

# 6. Prepare timesteps
timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, None, sigmas)

# 7. Prepare latent variables
num_channels_latents = pipe.unet.config.in_channels
latents = pipe.prepare_latents(
    batch_size * num_videos_per_prompt,
    num_frames,
    num_channels_latents,
    height,
    width,
    image_embeddings.dtype,
    device,
    None,
    None,
)
# Encode video to latents TODO: I HAVE NO IDEA WHAT THIS NORMALIZATION SHOULD BE
#latents = pipe._encode_vae_image((video / 255. * 2 - 1).permute(0,3,1,2).half(), device, 1, False)[None]

def get_clean(latents):
    # Add noise to get to timestep t_idx
    latents = latents + torch.randn_like(latents) * pipe.scheduler.sigmas[t_idx]

    # 9. Denoising loop
    pipe._num_timesteps = len(timesteps)

    # Denoise time t
    t = timesteps[t_idx]
    latent_model_input = latents
    latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)

    # Concatenate image_latents over channels dimension
    latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)

    # predict the noise residual
    noise_pred = pipe.unet(
        latent_model_input,
        t,
        encoder_hidden_states=image_embeddings,
        added_time_ids=added_time_ids,
        return_dict=False,
    )[0]

    # Upcast to avoid precision issues when computing prev_sample
    sample = latents.to(torch.float32)
    #sigma = pipe.scheduler.sigmas[pipe.scheduler.step_index]
    sigma = pipe.scheduler.sigmas[t_idx]

    s_churn: float = 0.0
    s_tmin: float = 0.0
    s_tmax: float = float("inf")
    s_noise: float = 1.0
    gamma = min(s_churn / (len(pipe.scheduler.sigmas) - 1), 2**0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0

    sigma_hat = sigma * (gamma + 1)

    if gamma > 0:
        noise = randn_tensor(
            noise_pred.shape, dtype=noise_pred.dtype, device=noise_pred.device, generator=generator
        )
        eps = noise * s_noise
        sample = sample + eps * (sigma_hat**2 - sigma**2) ** 0.5

    one_step_estimate = noise_pred * (-sigma / (sigma**2 + 1) ** 0.5) + (sample / (sigma**2 + 1))

    #pipe.vae.to(dtype=torch.float16)
    #one_step_frames = pipe.decode_latents(one_step_estimate.half(), num_frames, decode_chunk_size)
    #one_step_frames = pipe.video_processor.postprocess_video(video=one_step_frames, output_type='np')
    #media.write_video('test.mp4', one_step_frames[0], fps=7)

    return one_step_estimate

#one_step_estimate = get_clean(latents)

tangent = torch.zeros_like(latents)
h = 30
w = 45
s = 5
tangent[0, :, 0, h-s:h+s, w-s:w+s] = 1.0
output, jvp = torch.func.jvp(get_clean, (latents, ), (tangent, ))

_ = 1