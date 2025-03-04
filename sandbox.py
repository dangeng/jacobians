import torch
from diffusers import CogVideoXPipeline
from diffusers.pipelines.cogvideo.pipeline_cogvideox import retrieve_timesteps
from diffusers.utils import export_to_video
import numpy as np
import mediapy as media

################################
################################
################################
################################
torch.set_grad_enabled(False)
################################
################################
################################
################################

# Pipeline
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    torch_dtype=torch.bfloat16
)

prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()

video = pipe(
    prompt=prompt,
    num_videos_per_prompt=1,
    num_inference_steps=50,
    num_frames=49,
    guidance_scale=6,
    generator=torch.Generator(device="cuda").manual_seed(42),
).frames[0]

export_to_video(video, "output.mp4", fps=8)

# shape (t, h, w, c)
video_np = np.stack([np.array(frame) for frame in video])
media.write_video('output.mp4', video_np, fps=8)

# VAE takes (b,c,t,h,w), transformer takes (b,t,c,h,w)
with torch.no_grad():
    #x = torch.randn(1, 3, 8, 256, 256).bfloat16().cuda()
    #x = torch.tensor(video_np.transpose(3,0,1,2)[None, :, :, :240, :360] / 255.).bfloat16().cuda()
    x = torch.tensor(video_np.transpose(3,0,1,2)[None] / 255.).bfloat16().cuda()
    z = pipe.vae.encode(x)
    z = z['latent_dist'].mode()
    x_hat = pipe.vae.decode(z)
    media.write_video('input.mp4', x[0].float().cpu().numpy().transpose(1,2,3,0), fps=8)
    media.write_video('recon.mp4', x_hat.sample[0].float().cpu().numpy().transpose(1,2,3,0), fps=8)




height = 480
width = 720
num_frames = 49
num_inference_steps = 50

num_videos_per_prompt = 1
batch_size = 1

device = 0
guidance_scale = 1.0
prompt = "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."

# here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
# of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
# corresponds to doing no classifier free guidance.
do_classifier_free_guidance = guidance_scale > 1.0

# 3. Encode input prompt
prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
    prompt,
    '',
    do_classifier_free_guidance,
    num_videos_per_prompt=num_videos_per_prompt,
    prompt_embeds=None,
    negative_prompt_embeds=None,
    max_sequence_length=226,
    device=device,
)
if do_classifier_free_guidance:
    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

# 4. Prepare timesteps
timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, None)
pipe._num_timesteps = len(timesteps)

# 5. Prepare latents
#latent_frames = (num_frames - 1) // pipe.vae_scale_factor_temporal + 1

# For CogVideoX 1.5, the latent frames should be padded to make it divisible by patch_size_t
#patch_size_t = pipe.transformer.config.patch_size_t
#additional_frames = 0
#if patch_size_t is not None and latent_frames % patch_size_t != 0:
#    additional_frames = patch_size_t - latent_frames % patch_size_t
#    num_frames += additional_frames * pipe.vae_scale_factor_temporal
#
#latent_channels = pipe.transformer.config.in_channels
#latents = pipe.prepare_latents(
#    batch_size * num_videos_per_prompt,
#    latent_channels,
#    num_frames,
#    height,
#    width,
#    prompt_embeds.dtype,
#    device,
#    generator=None,
#    latents=None,
#)

# 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
extra_step_kwargs = pipe.prepare_extra_step_kwargs(generator=None, eta=0.0)

# 7. Create rotary embeds if required
image_rotary_emb = (
    pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
    if pipe.transformer.config.use_rotary_positional_embeddings
    else None
)

# 8. Denoising loop
num_warmup_steps = max(len(timesteps) - num_inference_steps * pipe.scheduler.order, 0)


# INSIDE DENOISING LOOP

t = timesteps[40]
pipe._current_timestep = t
#latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents   # should be no cfg
#latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, t)    # doesn't do anything
#latent_model_input = latent_model_input.cuda()

# Get alphas and betas
timestep = t.expand(1)
prev_timestep = timestep - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
alpha_prod_t = pipe.scheduler.alphas_cumprod.cuda()[timestep]
alpha_prod_t_prev = pipe.scheduler.alphas_cumprod.cuda()[prev_timestep] if prev_timestep >= 0 else pipe.scheduler.final_alpha_cumprod
beta_prod_t = 1 - alpha_prod_t

# Get latents and add noise
latents = z.permute(0,2,1,3,4)
latents = alpha_prod_t * latents + (1 - alpha_prod_t) * torch.randn_like(latents)
latents = latents.bfloat16()

# predict noise model_output
noise_pred = pipe.transformer(
    hidden_states=latents,
    encoder_hidden_states=prompt_embeds,
    timestep=timestep,
    image_rotary_emb=image_rotary_emb,
    attention_kwargs=None,
    return_dict=False,
)[0]
#noise_pred = noise_pred.float()

# DDIM STEP
pred_original_sample = (alpha_prod_t**0.5) * latents - (beta_prod_t**0.5) * noise_pred

# Decode one step estimate
frames = pipe.decode_latents(pred_original_sample.bfloat16())
one_step_video = pipe.video_processor.postprocess_video(video=frames, output_type='np')

media.write_video('test.mp4', one_step_video[0], fps=8)