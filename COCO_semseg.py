import cv2
import os
import sys
import numpy as np
from PIL import Image
import torch.nn as nn
from fourm.models.fm import FM
from fourm.vq.vqvae import VQVAE, DiVAE
from fourm.models.generate import GenerationSampler, build_chained_generation_schedules, init_empty_target_modality, init_full_input_modality, custom_text
from fourm.demo_4M_sampler import Demo4MSampler, img_from_url, load_model, DEFAULT_ORDER, DEFAULTS_RGB2X, MODALITY_PLOTTING_NAME_MAP, MODALITY_PLOTTING_ORDER
from torchvision.transforms.functional import center_crop
from fourm.data.modality_transforms import RGBTransform
from tokenizers import Tokenizer
from fourm.data.modality_info import MODALITY_INFO
from fourm.data.modality_transforms import get_transform_key, get_transform_resolution, MetadataTransform
from einops import rearrange
from fourm.utils.generation import unbatch
import shutil
try:
    from detectron2.utils.visualizer import ColorMode, Visualizer
    from detectron2.data import MetadataCatalog
    coco_metadata = MetadataCatalog.get("coco_2017_val_panoptic")
    USE_DETECTRON = True
except Exception as e:
    print(e)
    print("Detectron2 can be used for semseg visualizations. Please install detectron2 to use this feature, or plotting will fall back to matplotlib.")
    USE_DETECTRON = False

def load_image(path):
    rgb_transform = RGBTransform(imagenet_default_mean_and_std=True)
    img_pil = rgb_transform.load(path)
    img_pil = rgb_transform.preprocess(img_pil)
    img_pil = center_crop(img_pil, (min(img_pil.size), min(img_pil.size))).resize((224,224))
    img = rgb_transform.postprocess(img_pil).unsqueeze(0)
    return img

def resize_to_square(image, ismask=False):
        # Get the original image size
        original_width, original_height = image.size
        new_size = max(original_width, original_height)
        new_image = (
            Image.new("RGB", (new_size, new_size), (255, 255, 255))
            if not ismask
            else Image.new("RGBA", (new_size, new_size), (0, 0, 0, 1))
        )
        paste_x = (new_size - original_width) // 2
        paste_y = (new_size - original_height) // 2
        new_image.paste(image, (paste_x, paste_y))

        return new_image, paste_x, paste_y, original_width, original_height, new_size

def crop_back_image(
        square_image, paste_x, paste_y, original_width, original_height, new_size
    ):
        square_image = square_image.resize(
            (new_size, new_size), Image.LANCZOS
        )
        # Calculate the bounding box of the original image
        left = paste_x
        upper = paste_y
        right = paste_x + original_width
        lower = paste_y + original_height
        cropped_image = square_image.crop((left, upper, right, lower))

        return cropped_image
    
def get_value(defaults_dict, domain, key):
    """Look up a default value belonging to a given domain and key."""
    for domains, defaults in defaults_dict.items():
        if domain in domains:
            return defaults[key]
    
def decode_semseg(rgb_img, mod_dict, tokenizers, key='tok_semseg', image_size=224, patch_size=16, use_detectron=True, return_logits=False):
    """
    Decodes a sequence of semantic segmentation tokens from a model dictionary into an RGB image.

    Args:
        rgb_img (torch.Tensor): RGB image to overlay the semantic segmentation on.
        mod_dict (dict): Model output dictionary.
        tokenizers (dict): Dictionary of tokenizers.
        key (str): Key of the tokenized semantic segmentation modality to decode.
        image_size (int): Size of the image.
        patch_size (int): Size of the patches.
        use_detectron (bool): Uses detectron2's visualization for the semseg output.
    """
    tokens = mod_dict[key]['tensor']
    tokens = tokens.unsqueeze(0) if tokens.ndim == 1 else tokens
    img_tok = rearrange(tokens, "b (nh nw) -> b nh nw", nh=image_size//patch_size, nw=image_size//patch_size)
    rec = tokenizers[get_transform_key(key)].decode_tokens(img_tok).detach().cpu()
    if return_logits:
        return rec
    semsegs = rec.argmax(1)
    target_mask =  mod_dict[key]["target_mask"].cpu().numpy()
    # print(target_mask)
    print(target_mask.shape)
    print(mod_dict[key]["ids"].cpu().numpy().shape)
    # target_mask = cv2.merge([target_mask, target_mask, target_mask])
    cv2.imwrite("/tmp/target_mask.png", target_mask)
    B, H, W = semsegs.shape

    if not use_detectron:
    
        return semsegs if B > 1 else semsegs[0]
    
    else:
        rgb_imgs = [rgb_img] * B
        imgs = []
        for rgb, semseg in zip(rgb_imgs, semsegs):
            if USE_DETECTRON:
                v = Visualizer(255*rgb, coco_metadata, scale=1.2, instance_mode=ColorMode.IMAGE_BW)
                img = v.draw_sem_seg((semseg-1).cpu()).get_image() / 255.0
            else:
                colormap = plt.get_cmap('viridis')
                img = colormap(semseg.cpu())[..., :3]
            imgs.append(img)
        imgs = np_squeeze(np.stack(imgs), axis=0)
        return imgs

def __setup_sample_and_schedule(sample, cond_domains, target_domains, cfg_grow_conditioning=True):
    # 1 - Setup generation schedule
    
    defaults = DEFAULTS_RGB2X if ('rgb@224' in cond_domains or 'tok_rgb@224' in cond_domains) else DEFAULTS_X2RGB

    tokens_per_target = [get_value(defaults, domain, 'tokens_per_target') for domain in target_domains]
    autoregression_schemes = [get_value(defaults, domain, 'autoregression_scheme') for domain in target_domains]
    decoding_steps = [get_value(defaults, domain, 'decoding_steps') for domain in target_domains]
    token_decoding_schedules = [get_value(defaults, domain, 'token_decoding_schedule') for domain in target_domains]
    temps = [get_value(defaults, domain, 'temp') for domain in target_domains]
    temp_schedules = [get_value(defaults, domain, 'temp_schedule') for domain in target_domains]
    cfg_scales = [get_value(defaults, domain, 'cfg_scale') for domain in target_domains]
    cfg_schedules = [get_value(defaults, domain, 'cfg_schedule') for domain in target_domains]
    
    schedule = build_chained_generation_schedules(
        cond_domains=cond_domains, target_domains=target_domains, tokens_per_target=tokens_per_target, 
        autoregression_schemes=autoregression_schemes, decoding_steps=decoding_steps, 
        token_decoding_schedules=token_decoding_schedules, temps=temps, temp_schedules=temp_schedules,
        cfg_scales=cfg_scales, cfg_schedules=cfg_schedules, cfg_grow_conditioning=cfg_grow_conditioning, 
    )

    # 2 - Setup sample
    
    sample_dict = {}

    # Handle special cases
    if 'caption' in sample:
        caption = sample.pop('caption')
        sample_dict = custom_text(
            sample_dict, input_text=caption, eos_token='[EOS]', 
            key='caption', device=device, text_tokenizer=tok_text
        )
    if 'det' in sample:
        caption = sample.pop('det')
        sample_dict = custom_text(
            sample_dict, input_text=caption, eos_token='[EOS]', 
            key='det', device=device, text_tokenizer=tok_text
        )
    # Add remaining modalities
    sample_dict.update({domain: {'tensor': tensor} for domain, tensor in sample.items()})
    
    # Initialize these remaining input modalities (caption and det are already initialized by custom_text)
    for cond_mod in sample.keys():
        sample_dict = init_full_input_modality(sample_dict, MODALITY_INFO, cond_mod, device, eos_id=tok_text.token_to_id("[EOS]"))
    
    # Initialize target modalities
    for target_mod, ntoks in zip(target_domains, tokens_per_target):
        sample_dict = init_empty_target_modality(sample_dict, MODALITY_INFO, target_mod, 1, ntoks, device)

    return sample_dict, schedule

# Use Demo
if __name__ == "__main__":
    path_fm = "F:/ml-4m/checkpoints/4M-21_XL/model.safetensors"
    path_fm_sr = "EPFL-VILAB/4M-7-SR_B_CC12M"
    path_COCO_sem_seg = "EPFL-VILAB/4M_tokenizers_semseg_4k_224-448"     #"/home/ubuntu/ml-4m/checkpoints/COCO_seg_sam/"
    image_input = "F:\\ml-4m\\test\\drinkminna-1000w.jpg"                         #F:/ml-4m/test/allbirds/   #squareface/ F:\ml-4m\test\squareface\beauty-2-500w.jpg
    sampler = Demo4MSampler(fm=path_fm,
                            fm_sr=None, 
                            tok_rgb=None,
                            tok_depth=None,
                            tok_normal=None,
                            tok_edge=None,
                            tok_semseg=path_COCO_sem_seg,
                            tok_clip=None,
                            tok_dinov2=None,
                            tok_imagebind=None,
                            tok_sam_instance=None,
                            tok_human_poses=None,).cuda()
    # sampler = Demo4MSampler(fm_sr='EPFL-VILAB/4M-7-SR_L_CC12M').cuda()
    if os.path.isfile(image_input):
        list_image = [image_input]
    else:
        list_image = [os.path.join(image_input, image)
                      for image in os.listdir(image_input)]
    num_threads = 1
    path_tmp = "C:/Users/admin/AppData/Local/Temp/ml-4m_tmp.png"
    target_modalities_tmp = ['tok_semseg@224']
    for filename in list_image:
        # count = 1
        # filename = "781490520120333.mp4"
        # text_img_folder = f"{os.path.dirname(os.path.dirname(filename))}/results"

        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):
            # path
            
            
            
            image_name = os.path.basename(filename).split(".")[0]
            
            text_img_folder = f"{os.path.dirname(filename)}\\{os.path.basename(filename).split('.')[0]}"
            asset_folder    = f"{os.path.dirname(filename)}\\{os.path.basename(filename).split('.')[0]}"
            asset_output_folder = f"{os.path.dirname(filename)}\\4M_output\\modelL\\{image_name}"
            if not os.path.exists(asset_output_folder):
                os.mkdir(asset_output_folder, mode=0o777)
            
            list_crop = [os.path.join(asset_folder, image)
                      for image in os.listdir(asset_folder)]
            for filename_crop in list_crop:
                if filename_crop.lower().endswith(('crop.jpg', 'crop.JPG', 'crop.png', "crop.PNG", 'crop.jpeg', 'crop.JPEG')):
                    image_name_crop = os.path.basename(filename_crop).split("_crop")[0]
                    image = Image.open(filename_crop)
                    ori_size = max(image.size[0],image.size[1])
                    new_image, paste_x, paste_y, original_width, original_height, new_size = resize_to_square(image)
                    new_image.save(path_tmp)
                    img = load_image(path_tmp)

                    preds = sampler({'rgb@224': img.cuda()}, seed=0,
                                    target_modalities= target_modalities_tmp) 
                    # print(preds)
                    # for i, (mod_name, mod) in enumerate(preds.items()):
                    #     img_pil = Image.fromarray((255*mod).astype(np.uint8))
                    #     img_pil.save(f"/tmp/{mod_name}.png")
                    # # sampler.plot_modalities(preds, save_path=None)
                    list_output = sampler.modalities_to_pil(preds, False, ori_size)
                    for image_tmp, name in list_output:
                        cropped_image = crop_back_image(image_tmp,paste_x, paste_y, original_width, original_height, new_size )
                        cropped_image.save(f"{asset_output_folder}/{image_name_crop}_visualize.png")
                        # print(filename_crop)
                        # print(f"{asset_output_folder}/{image_name_crop}_crop.png")
                        shutil.copy(filename_crop, f"{asset_output_folder}/{image_name_crop}_crop.png")
                        # image_tmp.save(f"/home/ubuntu/allbirds/4M_output/modelB/{image_name}_{name}_s224.png")


#Use GennerationSampler
# if __name__ == "__main__":
#     device = 'cuda'
#     path_fm = "/home/ubuntu/ml-4m/checkpoints/4M-21_L/model.safetensors"
#     path_fm_sr = "EPFL-VILAB/4M-7-SR_B_CC12M"
#     path_COCO_sem_seg = "EPFL-VILAB/4M_tokenizers_semseg_4k_224-448"#"/home/ubuntu/ml-4m/checkpoints/COCO_seg_sam/"
#     image_input = "/home/ubuntu/allbirds/allbirds1.png"
#     tok_text = './fourm/utils/tokenizer/trained/text_tokenizer_4m_wordpiece_30k.json'
#     # Load 4M model and initialize sample
#     toks = {}
#     fm = load_model(path_fm, FM)
#     sampler_fm = GenerationSampler(fm).cuda()
#     toks['tok_semseg'] = load_model(path_COCO_sem_seg, VQVAE)
#     toks = nn.ModuleDict(toks).cuda()
#     tok_text = Tokenizer.from_file(tok_text)
#     # sampler = Demo4MSampler(fm_sr='EPFL-VILAB/4M-7-SR_L_CC12M').cuda()
#     if os.path.isfile(image_input):
#         list_image = [image_input]
#     else:
#         list_image = [os.path.join(image_input, image)
#                       for image in os.listdir(image_input)]
#     num_threads = 1
#     path_tmp = "/tmp/ml-4m_tmp.png"
#     target_modalities_tmp = ['tok_semseg@224']
#     defaults = DEFAULTS_RGB2X 
#     cond_domains = ['rgb@224']
#     target_domains = ['tok_semseg@224']
#     # tokens_per_target = [get_value(defaults, domain, 'tokens_per_target') for domain in target_domains]
#     # autoregression_schemes = [get_value(defaults, domain, 'autoregression_scheme') for domain in target_domains]
#     # decoding_steps = [get_value(defaults, domain, 'decoding_steps') for domain in target_domains]
#     # token_decoding_schedules = [get_value(defaults, domain, 'token_decoding_schedule') for domain in target_domains]
#     # temps = [get_value(defaults, domain, 'temp') for domain in target_domains]
#     # temp_schedules = [get_value(defaults, domain, 'temp_schedule') for domain in target_domains]
#     # cfg_scales = [get_value(defaults, domain, 'cfg_scale') for domain in target_domains]
#     # cfg_schedules = [get_value(defaults, domain, 'cfg_schedule') for domain in target_domains]
    
#     # generation_schedule = build_chained_generation_schedules(
#     #         cond_domains=cond_domains, target_domains=target_domains, tokens_per_target=tokens_per_target, 
#     #         autoregression_schemes=autoregression_schemes, decoding_steps=decoding_steps, 
#     #         token_decoding_schedules=token_decoding_schedules, temps=temps, temp_schedules=temp_schedules,
#     #         cfg_scales=cfg_scales, cfg_schedules=cfg_schedules, cfg_grow_conditioning=True, 
#     #     )
#     for filename in list_image:
#         # count = 1
#         # filename = "781490520120333.mp4"
#         # text_img_folder = f"{os.path.dirname(os.path.dirname(filename))}/results"

#         if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):
#             image = Image.open(filename)
#             new_image, paste_x, paste_y, original_width, original_height, new_size = resize_to_square(image)
#             new_image.save(path_tmp)
            
            
#             image_name = os.path.basename(filename).split(".")[0]
#             ori_size = max(image.size[0],image.size[1])
#             text_img_folder = f"{os.path.dirname(filename)}/{os.path.basename(filename).split('.')[0]}"
#             img = load_image(filename)
#             sample = {'rgb@224': img.cuda()}
#             sample, generation_schedule = __setup_sample_and_schedule(sample, cond_domains, target_domains)
#             out_dict = sampler_fm.generate(
#                 sample, generation_schedule, text_tokenizer=tok_text, 
#                 verbose=True, seed=None, top_p=0.8, top_k=0.0,
#             )
#             image_size = 224
#             print(out_dict)
#             for key in out_dict:
#                 k, res = get_transform_key(key), get_transform_resolution(key, image_size, to_tuple=False)
#                 if k == 'tok_semseg':
#                     decoded = decode_semseg(
#                         np.ones((res, res, 3)), out_dict, toks, key=key, 
#                         image_size=res,  return_logits=False
#                     )
                    
                
#             # preds = sampler({'rgb@224': img.cuda()}, seed=None,
#             #                 target_modalities= target_modalities_tmp) 
#             # # print(preds)
#             # for i, (mod_name, mod) in enumerate(preds.items()):
#             #     img_pil = Image.fromarray((255*mod).astype(np.uint8))
#             #     img_pil.save(f"/tmp/{mod_name}.png")
#             # # sampler.plot_modalities(preds, save_path=None)
#             # list_output = sampler.modalities_to_pil(preds, False, ori_size)
#             # for image_tmp, name in list_output:
#             #     # cropped_image = crop_back_image(image_tmp,paste_x, paste_y, original_width, original_height, new_size )
#             #     # cropped_image.save(f"/home/ubuntu/allbirds/4M_output/modelL/{image_name}_{name}_s224.png")
#             #     image_tmp.save(f"/home/ubuntu/allbirds/4M_output/modelB/{image_name}_{name}_s224.png")
            
