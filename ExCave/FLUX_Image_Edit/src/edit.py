import os
# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import re
import time
from dataclasses import dataclass
from glob import iglob
import argparse
import torch
from einops import rearrange, repeat
from fire import Fire
from PIL import ExifTags, Image

from flux.sampling import denoise, get_schedule, prepare, unpack
from flux.util import (configs, embed_watermark, load_ae, load_clip,
                       load_flow_model, load_t5)
from transformers import pipeline
from PIL import Image
import numpy as np

import os

NSFW_THRESHOLD = 0.85

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str | list[str]
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    init_image = ae.encode(init_image.to()).to(torch.bfloat16)
    return init_image

@torch.inference_mode()
def main(
    args,
    seed: int | None = None,
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
    num_steps: int | None = None,
    loop: bool = False,
    offload: bool = False,
    add_sampling_metadata: bool = True,
):
    """
    Sample the flux model. Either interactively (set `--loop`) or run for a
    single image.

    Args:
        name: Name of the model to load
        height: height of the sample in pixels (should be a multiple of 16)
        width: width of the sample in pixels (should be a multiple of 16)
        seed: Set a seed for sampling
        output_name: where to save the output image, `{idx}` will be replaced
            by the index of the sample
        prompt: Prompt used for sampling
        device: Pytorch device
        num_steps: number of sampling steps (default 4 for schnell, 50 for guidance distilled)
        loop: start an interactive session and sample multiple times
        guidance: guidance value used for guidance distillation
        add_sampling_metadata: Add the prompt to the image Exif metadata
    """
    # import pdb
    # pdb.set_trace()
    torch.set_grad_enabled(False)
    name = args.name
    source_prompt = args.source_prompt
    target_prompt = args.target_prompt
    guidance = args.guidance
    output_dir = args.output_dir
    num_steps = args.num_steps
    offload = args.offload
    memory_stream = torch.cuda.Stream()
    kvload_stream = torch.cuda.Stream()
    kvstore_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    prealloc_stream = torch.cuda.Stream()
    maskgen_stream = torch.cuda.Stream()
    crossattn_stream = torch.cuda.Stream()
    knorm_stream = torch.cuda.Stream()
    vnorm_stream = torch.cuda.Stream()
    info_event = torch.cuda.Event()
    
    t5 = args.t5 
    clip = args.clip
    model = args.model
    ae = args.ae

    info = {}
    info['feature_path'] = args.feature_path
    info['feature'] = {}
    info['blockres'] = {}
    info['inject_step'] = args.inject
    
    with torch.cuda.stream(prealloc_stream):
        info_event.wait(stream=prealloc_stream)
        timestep_list = [1.0, 0.9778246283531189, 0.9534292817115784, 0.9264630675315857, 0.8964968323707581, 0.8630005717277527, 0.825311541557312, 0.782589852809906, 0.7337554097175598, 0.677395224571228, 0.6116241812705994, 0.5338707566261292, 0.4405321776866913, 0.3264005184173584, 0.18365685641765594]
        order_num = 2
        for i in range(len(timestep_list)):
            for j in range(order_num):
                for k in range(38):
                    if j == 0:
                        kv_feature_name = str(timestep_list[i]) + '_' + 'True' + '_' + str(k) + '_' + 'single' + '_' + 'KV'
                    else:
                        kv_feature_name = str(timestep_list[i]) + '_' + 'False' + '_' + str(k) + '_' + 'single' + '_' + 'KV'
                    # print('pre store kv: ', kv_feature_name)
                    info['feature'][kv_feature_name] = torch.zeros((2, 24, 4080, 128), dtype=torch.bfloat16, device='cuda:0').to('cpu', non_blocking=True)
                if timestep_list[i] < 0.97:
                    if j == 0:
                        x_feature_name = str(timestep_list[i]) + '_' + 'True' + '_' + str(0) + '_' + 'single' + '_' + 'x'
                        res_feature_name = str(timestep_list[i]) + '_' + 'True' + '_' + str(37) + '_' + 'single' + '_' + 'res'
                        info['feature'][x_feature_name] = torch.zeros((1, 4080, 3072), dtype=torch.bfloat16, device='cuda:0').to('cpu', non_blocking=True)
                        info['feature'][res_feature_name] = torch.zeros((1, 4080, 3072), dtype=torch.bfloat16, device='cuda:0').to('cpu', non_blocking=True)
                    else:
                        x_feature_name = str(timestep_list[i]) + '_' + 'False' + '_' + str(0) + '_' + 'single' + '_' + 'x'
                        res_feature_name = str(timestep_list[i]) + '_' + 'False' + '_' + str(37) + '_' + 'single' + '_' + 'res'
                        info['feature'][x_feature_name] = torch.zeros((1, 4080, 3072), dtype=torch.bfloat16, device='cuda:0').to('cpu', non_blocking=True)
                        info['feature'][res_feature_name] = torch.zeros((1, 4080, 3072), dtype=torch.bfloat16, device='cuda:0').to('cpu', non_blocking=True)

    
    info_event.record(torch.cuda.current_stream())
    nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection", device=device)

    if name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Got unknown model name: {name}, chose from {available}")

    torch_device = torch.device(device)
    if num_steps is None:
        num_steps = 4 if name == "flux-schnell" else 25

    
    init_image = None
    init_image = np.array(Image.open(args.source_img_dir).convert('RGB'))
    
    shape = init_image.shape

    new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
    new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

    init_image = init_image[:new_h, :new_w, :]
    # print('init image before ae shape: ', init_image.shape)

    width, height = init_image.shape[0], init_image.shape[1]
    init_image = encode(init_image, torch_device, ae)
    # print('init image after ae shape: ', init_image.shape)

    rng = torch.Generator(device="cpu")
    opts = SamplingOptions(
        source_prompt=source_prompt,
        target_prompt=target_prompt,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
    )

    if loop:
        opts = parse_prompt(opts)

    # while opts is not None:
    for i in range(1):
        if opts.seed is None:
            opts.seed = rng.seed()
        print(f"Generating with seed {opts.seed}:\n{opts.source_prompt}")
        t0 = time.perf_counter()

        opts.seed = None
        if offload:
            ae = ae.cpu()
            torch.cuda.empty_cache()
            t5, clip = t5.to(torch_device), clip.to(torch_device)

        if not os.path.exists(args.feature_path):
            os.mkdir(args.feature_path)

        inp, h_tmp, w_tmp = prepare(t5, clip, init_image, prompt=opts.source_prompt)
        inp_target, h_tmp, w_tmp = prepare(t5, clip, init_image, prompt=opts.target_prompt)
        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(name != "flux-schnell"))

        source_txt = inp["txt"]
        target_txt = inp_target["txt"]
        source_txt_tmp = repeat(source_txt, "1 ... -> bs ...", bs=target_txt.shape[0])##############
        source_txt_tmp_s = source_txt_tmp.clone()
        target_txt_s = target_txt.clone()
        source_txt_tmp_s[1:, ...] = target_txt_s[:-1, ...]
        source_txt_tmp = source_txt_tmp_s
        prompt_delta = (source_txt_tmp - target_txt).abs().sum(-1).float()
        value, index = prompt_delta.sort(dim=-1)
        txt_index = index[:, -20:]

        # offload TEs to CPU, load model to gpu
        if offload:
            t5, clip = t5.cpu(), clip.cpu()
            torch.cuda.empty_cache()
            model = model.to(torch_device)

        z, info = denoise(model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info, memory_stream=memory_stream, knorm_stream=knorm_stream, vnorm_stream=vnorm_stream, kvload_stream=kvload_stream, kvstore_stream=kvstore_stream, compute_stream=compute_stream, maskgen_stream=maskgen_stream, crossattn_stream=crossattn_stream, txt_index=-1, h_tmp=h_tmp, w_tmp=w_tmp)
        t_tmp = time.perf_counter()
        print(f"Done inversion in {t_tmp - t0:.1f}s.")

        inp_target["img"] = z

        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(name != "flux-schnell"))

        # denoise initial noise
        for i in range(len(target_prompt)):
            img_idx = i
            if offload:
                model.to(torch_device)
                torch.cuda.empty_cache()
                ae.decoder.cpu()
            inp_target_tmp = inp_target.copy()
            inp_target_tmp['img'] = inp_target['img']#########shape[bs, ...], bs=1?
            inp_target_tmp['img_ids'] = inp_target['img_ids'][i:i+1, ...]
            inp_target_tmp['txt'] = inp_target['txt'][i:i+1, ...]
            inp_target_tmp['txt_ids'] = inp_target['txt_ids'][i:i+1, ...]
            inp_target_tmp['vec'] = inp_target['vec'][i:i+1, ...]

            x, info = denoise(model, **inp_target_tmp, timesteps=timesteps, guidance=guidance, inverse=False, info=info, memory_stream=memory_stream, knorm_stream=knorm_stream, vnorm_stream=vnorm_stream, kvload_stream=kvload_stream, kvstore_stream=kvstore_stream, compute_stream=compute_stream, maskgen_stream=maskgen_stream, crossattn_stream=crossattn_stream, txt_index=txt_index[i, ...], h_tmp=h_tmp, w_tmp=w_tmp)
        
            if offload:
                model.cpu()
                torch.cuda.empty_cache()
                ae.decoder.to(x.device)

            # decode latents to pixel space
            batch_x = unpack(x.float(), opts.width, opts.height)

            for x in batch_x:
                x = x.unsqueeze(0)
                output_dir_step = os.path.join(output_dir, "step")
                output_name = os.path.join(output_dir_step, "img_{idx}_{img_idx}.jpg")
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                if not os.path.exists(output_dir_step):
                    os.makedirs(output_dir_step)

                idx = args.idx
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                fn = output_name.format(idx=idx,img_idx=img_idx)
                print(f"Done generation in {t1 - t0:.1f}s. Saving {fn}")
                # bring into PIL format and save
                x = x.clamp(-1, 1)
                x = embed_watermark(x.float())
                x = rearrange(x[0], "c h w -> h w c")

                img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
                # nsfw_score = [x["score"] for x in nsfw_classifier(img) if x["label"] == "nsfw"][0]
                nsfw_score = 0
                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = name
                    if add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = source_prompt
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)
                    if img_idx ==4:
                        output_dir_final = os.path.join(output_dir, "final")
                        if not os.path.exists(output_dir_final):
                            os.makedirs(output_dir_final)
                        fn_final = os.path.join(output_dir_final, "img_{idx}.jpg".format(idx=idx))
                        img.save(fn_final, exif=exif_data, quality=200, subsampling=0)
                    # idx += 1
                else:
                    print("Your generated image may contain NSFW content.")


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='RF-Edit')

    parser.add_argument('--name', default='flux-dev', type=str,
                        help='flux model')
    # parser.add_argument('--source_img_dir', default='', type=str,
    #                     help='The path of the source image')
    # parser.add_argument('--source_prompt', type=str,
    #                     help='describe the content of the source image (or leaves it as null)')
    # parser.add_argument('--target_prompt', type=None,
    #                     help='describe the requirement of editing')
    parser.add_argument('--feature_path', type=str, default='feature',
                        help='the path to save the feature ')
    parser.add_argument('--guidance', type=float, default=5,
                        help='guidance scale')
    parser.add_argument('--num_steps', type=int, default=25,
                        help='the number of timesteps for inversion and denoising')
    parser.add_argument('--inject', type=int, default=20,
                        help='the number of timesteps which apply the feature sharing')
    parser.add_argument('--output_dir', default='output', type=str,
                        help='the path of the edited image')
    parser.add_argument('--offload', action='store_true', help='set it to True if the memory of GPU is not enough')
    parser.add_argument('--img_config', default='img_config', type=str,
                        help='the path of the image config')
    
    args = parser.parse_args()

    import json
    with open(args.img_config, 'r') as f:
        if args.img_config.endswith('.jsonl'):
            img_configs = [json.loads(line) for line in f]
        else:
            img_configs = json.load(f)
            
    # init all components
    name = args.name
    offload = args.offload
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)
    args.t5 = load_t5(torch_device, max_length=256 if name == "flux-schnell" else 512)
    args.clip = load_clip(torch_device)
    args.model = load_flow_model(name, device="cpu" if offload else torch_device)
    args.ae = load_ae(name, device="cpu" if offload else torch_device)
    # print('model: ', model)

    if offload:
        args.model.cpu()
        torch.cuda.empty_cache()
        args.ae.encoder.to(torch_device)
        
    for idx, img_config in enumerate(img_configs['imgs']):
        args.source_prompt = img_config["source_prompt"]
        args.source_img_dir = img_config["image"]
        args.target_prompt = img_config["prompts"]
        args.idx = idx
        main(args)
