import cv2 as cv
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

from rembg import remove, new_session


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
    
def segmen(img_path):
    img = cv.imread(img_path)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 85, 255, cv.THRESH_BINARY_INV)
    cv.erode( thresh, (3,3), iterations=3)
    cv.dilate(thresh, (3,3), iterations=2)
    return thresh

def transparent(img):
    color = (0,0,0)
    mask = np.where((img==color).all(axis=2), 0, 255).astype(np.uint8)

    result = img.copy()
    result = cv.cvtColor(result, cv.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result


def apply_mask(img_path, mask):
    image = cv.imread(img_path)
    res = np.zeros_like(image)

    idx = (mask!=0)
    res[idx] = image[idx]
    return transparent(res)


# Use Demo
if __name__ == "__main__":
    path_fm = "F:/ml-4m/checkpoints/4M-21_XL/model.safetensors"
    path_fm_sr = "EPFL-VILAB/4M-7-SR_B_CC12M"
    path_COCO_sem_seg = "EPFL-VILAB/4M_tokenizers_semseg_4k_224-448"     
    image_input = "F:\\ml-4m\\test\\drinkminna-1000w.jpg"                         
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
    
    session = new_session("u2net")

    if os.path.isfile(image_input):
        list_image = [image_input]
    else:
        list_image = [os.path.join(image_input, image)
                      for image in os.listdir(image_input)]
    num_threads = 1
    path_tmp = "C:/Users/admin/AppData/Local/Temp/ml-4m_tmp.png"
    target_modalities_tmp = ['tok_semseg@224']
    for filename in list_image:


        if filename.lower().endswith(('.jpg', '.JPG', '.png', ".PNG", '.jpeg', ',JPEG')):
            
            
            
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
                    
                    u2mask = remove(image, session=session, only_mask=True, post_process_mask=True)
 
                    list_output = sampler.modalities_to_pil(preds, False, ori_size)
                    for image_tmp, name in list_output:
                        cropped_image = crop_back_image(image_tmp,paste_x, paste_y, original_width, original_height, new_size )
                        cropped_image.save(f"{asset_output_folder}/{image_name_crop}_visualize.png")
                        u2mask.save(f"{asset_output_folder}/{image_name_crop}_u2mask.png")
                        shutil.copy(filename_crop, f"{asset_output_folder}/{image_name_crop}_crop.png")

            for image in os.listdir(asset_output_folder):
                if image.lower().endswith(('visualize.jpg', 'visualize.JPG', 'visualize.png', "visualize.PNG", 'visualize.jpeg', ',visualize.JPEG')):
                    sub_img_name = image.split("_visualize")[0]
                    absolute_mask_path = f"{asset_output_folder}\\{image}"
                    absolute_img_path = f"{asset_output_folder}\\{sub_img_name}_crop.png"
                    mask = segmen(img_path= absolute_mask_path)
                    # masked = apply_mask(img_path=absolute_img_path, mask=mask) #create transparent image
                    cv.imwrite(f"{asset_output_folder}\\{sub_img_name}_4mmask.png", mask)
                    #cv.imwrite(f"{asset_output_folder}\\{sub_img_name}_transparent.png", masked) #save transparent image 
                 



