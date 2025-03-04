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

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
################################
################################
################################
################################

# Params
height = 480
width = 720
num_frames = 49
num_inference_steps = 50
num_videos_per_prompt = 1
batch_size = 1
device = 0
#device = 'cpu'
guidance_scale = 1.0
prompt = "goldfish swimming in a blue aquarium"
video_path = '/nfs/turbo/coe-ahowens-nobackup/datasets/davis2017/DAVIS/Videos/480p/gold-fish.mp4'

# Load Pipeline
pipe = CogVideoXPipeline.from_pretrained(
    "THUDM/CogVideoX-5b",
    #torch_dtype=torch.bfloat16
    torch_dtype=torch.float
).to(device)
#pipe.enable_model_cpu_offload()
#pipe.vae.enable_tiling()

# Read video
video = media.read_video(video_path)
video = video[:49, :480, :720, :3]

# Encode in latent space
# VAE takes (b,c,t,h,w), transformer takes (b,t,c,h,w)
#with torch.no_grad():
#    x = torch.tensor(video.transpose(3,0,1,2)[None] / 255.).bfloat16().cuda()
#    latent = pipe.vae.encode(x)
#    latent_dist = latent.latent_dist
#    latent = latent_dist.sample()
#    latent_scaled = latent * pipe.vae_scaling_factor_image
#
#    # Sanity check encoding
#    x_hat = pipe.vae.decode(latent)
#    media.write_video('input.mp4', x[0].float().cpu().numpy().transpose(1,2,3,0), fps=8)
#    media.write_video('recon.mp4', x_hat.sample[0].float().cpu().numpy().transpose(1,2,3,0), fps=8)
latent_scaled = torch.randn(1,16,13,60,90).to(device) * 0.7

# Encode input prompt
with torch.no_grad():
    prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
        prompt,
        '',
        do_classifier_free_guidance=False,
        num_videos_per_prompt=num_videos_per_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=device,
    )


def get_clean(latents, timestep_idx=20, num_inference_steps=50, height=480, width=720):
    # Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(pipe.scheduler, num_inference_steps, device, None)
    pipe._num_timesteps = len(timesteps)

    # 7. Create rotary embeds if required
    image_rotary_emb = (
        #pipe._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
        pipe._prepare_rotary_positional_embeddings(height, width, latent_scaled.shape[2], device)
        if pipe.transformer.config.use_rotary_positional_embeddings
        else None
    )

    # Get timestep
    t = timesteps[timestep_idx]
    pipe._current_timestep = t

    # Get alphas and betas
    timestep = t.expand(1)
    prev_timestep = timestep - pipe.scheduler.config.num_train_timesteps // pipe.scheduler.num_inference_steps
    alpha_prod_t = pipe.scheduler.alphas_cumprod.to(device)[timestep]
    alpha_prod_t_prev = pipe.scheduler.alphas_cumprod.to(device)[prev_timestep] if prev_timestep >= 0 else pipe.scheduler.final_alpha_cumprod
    beta_prod_t = 1 - alpha_prod_t

    # Get latents and add noise
    #latents = latent_scaled.permute(0,2,1,3,4)
    latents = latents.permute(0,2,1,3,4)
    latents = torch.sqrt(alpha_prod_t) * latents + torch.sqrt(1 - alpha_prod_t) * torch.randn_like(latents)
    #latents = latents.bfloat16()
    latents = latents.float()

    print(latents.dtype)
    print(prompt_embeds.dtype)
    #image_rotary_emb = (image_rotary_emb[0].bfloat16(), image_rotary_emb[1].bfloat16())
    print(image_rotary_emb[0].dtype)
    print(image_rotary_emb[1].dtype)

    # predict noise model_output
    noise_pred = pipe.transformer(
        hidden_states=latents,
        encoder_hidden_states=prompt_embeds,
        timestep=timestep,
        image_rotary_emb=image_rotary_emb,
        attention_kwargs=None,
        return_dict=False,
    )[0]

    # DDIM STEP
    pred_original_sample = (alpha_prod_t**0.5) * latents - (beta_prod_t**0.5) * noise_pred

    return pred_original_sample

import torch.utils.checkpoint

def get_clean_checkpointed(*args):
    return torch.utils.checkpoint.checkpoint(get_clean, *args)


#with torch.no_grad():
#    clean_pred = get_clean(latent_scaled, timestep_idx=10)
#
#    # Decode one step estimate
#    frames = pipe.decode_latents(clean_pred.bfloat16())
#    media.write_video('test.mp4', frames.float().cpu().numpy()[0].transpose(1,2,3,0), fps=8)

tangent = torch.zeros_like(latent_scaled)
h = 30
w = 45
s = 10
tangent[0, :, 0, h-s:h+s, w-s:w+s] = 1.0
with torch.no_grad():
    #output, jvp = torch.func.jvp(get_clean, (latent_scaled, ), (tangent, ))
    output, jvp = torch.func.jvp(get_clean_checkpointed, (latent_scaled, ), (tangent, ))