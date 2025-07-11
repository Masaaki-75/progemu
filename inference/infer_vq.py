import os
import sys
sys.path.append('..')

import torch
import argparse
from PIL import Image
from tqdm import tqdm

from emu3.visionvq import Emu3VisionVQModel, Emu3VisionVQImageProcessor
from utilities.helpers import find_files_recursively


def smart_resize(image, image_area=512*512):
    w, h = image.size
    current_area = h * w
    if current_area != image_area:
        target_ratio = (image_area / current_area) ** 0.5
        th = int(round(h * target_ratio))
        tw = int(round(w * target_ratio))
        image = image.resize((tw, th))
    return image


def tokenize_one_image(image_path, tokenizer, processor,  image_area=512*512):
    pil_image = Image.open(image_path).convert("RGB")
    pil_image = smart_resize(pil_image, image_area)
    image = processor(images=pil_image, return_tensors="pt")["pixel_values"]
    
    token_ids = tokenizer.encode(image.to(tokenizer.device))
    token_ids = token_ids.squeeze().cpu()
    return token_ids


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--ext", type=str, default="png")
    args = parser.parse_args()
    
    image_dir = args.image_dir
    ext = args.ext
    if not ext.startswith('.'):
        ext = '.' + ext
    
    # Find all PNG files within the directory
    image_paths = find_files_recursively(image_dir, ext=ext)
    
    # Prepare VQ-VAE as image tokenizer
    image_processor = Emu3VisionVQImageProcessor.from_pretrained(args.model_path)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(args.model_path, device_map="cuda").eval()
    image_area = 512 * 512
    
    with torch.no_grad():
        for image_path in tqdm(image_paths, desc="image tokenization"):
            image_name = os.path.basename(image_path)
            parent_dir = os.path.dirname(image_path)
            token_name = image_name.replace(ext, '.pth')
            token_path = os.path.join(parent_dir, token_name)
            
            token_ids = tokenize_one_image(image_path, image_tokenizer, image_processor, image_area=image_area)
            torch.save(token_ids, token_path)
    
