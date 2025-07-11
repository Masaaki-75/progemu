import os
import json
import math
import random
import pynvml
import logging
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from io import BytesIO
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm


################################################################################################
# NUMBER PROCESSING
################################################################################################

def true_round(value):
    """
    Perform rounding as taught in school, i.e., rounding to nearest integer 
    without considering it's odd or even.
    """
    return np.floor(value + 0.5).astype(int) if value >= 0 else np.ceil(value - 0.5).astype(int)


def nearest_divisible(num, n=4, output_upper=True):
    lower = num - (num % n)
    upper = lower + n
    if output_upper:
        return upper if lower != num else lower
    else:
        return lower


def nearest_power(n, p=2, output_type='upper'):
    assert n > 0, "Input number must be positive."

    lower_power = p ** (int(n).bit_length() - 1)
    upper_power = lower_power * p

    if output_type == 'upper':
        return upper_power if lower_power != n else lower_power
    elif output_type == 'lower':
        return lower_power
    else:
        return lower_power if abs(n - lower_power) <= abs(n - upper_power) else upper_power



################################################################################################
# FILE READING, WRITING AND SEARCH
################################################################################################

def exist(path):
    return (path is not None) and os.path.exists(path)

def read_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    return [l.strip() for l in lines]

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, json_path, write_mode='w', encoding='utf-8'):
    with open(json_path, write_mode, encoding=encoding) as f:
        json.dump(data, f, indent=4)

def write_jsonl(dicts, jsonl_path, write_mode='w'):
    if not isinstance(dicts, (list, tuple)):
        dicts = [dicts,]
        
    with open(jsonl_path, write_mode, encoding='utf-8') as f:
        for adict in dicts:
            f.write(json.dumps(adict) + '\n')
                    
        

def find_files_recursively(directory, ext='', inclusions='', exclusions=None):
    matched_paths = []
    cond3 = True
    if not isinstance(inclusions, (list, tuple)):
        inclusions = [inclusions,]  # make it a list by default
    if (exclusions is not None) and (not isinstance(exclusions, (list, tuple))):
        exclusions = [exclusions,]
    
    if not os.path.exists(directory):
        raise ValueError(f'Path does not exist: {directory}.')

    for root, dirs, files in os.walk(directory):
        for d in dirs[:]: 
            # Deal with extensions like `.nii.gz`. These files will be seen as directories by `os.walk`
            if d.endswith('.nii.gz'):  
                dirs.remove(d)  # Remove it from dirs, so we don’t recurse into it
                files.append(d)  # Treat it as a file

        for file in files:
            cond1 = all(s in file for s in inclusions)
            cond2 = file.endswith(ext)
            if exclusions is not None:
                cond3 = all(s not in file for s in exclusions)
            
            if cond1 and cond2 and cond3:
                matched_paths.append(os.path.join(root, file))

    return sorted(matched_paths)


def find_folders(directory, inclusions=''):
    if not isinstance(inclusions, (list, tuple)):
        inclusions = [inclusions]  # make it a list by default
        
    names = os.listdir(directory)
    folders = []
    for n in names:
        folder = os.path.join(directory, n)
        if os.path.isdir(folder):
            if all(s in n for s in inclusions):
                folders.append(folder)
    
    return sorted(folders)


def check_multiple_exist(paths):
    if not isinstance(paths, (tuple, list)):
        paths = [paths]
    
    for p in paths:
        assert os.path.exists(p), f'Path does not exist: {p}'


################################################################################################
# STRING PROCESSING
################################################################################################






################################################################################################
# LIST and DICT PROCESSING
################################################################################################

def interleave_lists(a, b):
    if len(a) != len(b):
        raise ValueError("The two lists must have the same length.")
    
    interleaved = [item for pair in zip(a, b) for item in pair]
    return interleaved

def concat_lists(sublists):
    concatenated = [item for sublist in sublists for item in sublist]
    return concatenated

def divide_list_with_overlap(a, group_len, overlap=0):
    if group_len <= 0 or overlap < 0 or group_len <= overlap:
        raise ValueError("group_len must be positive and greater than overlap.")

    groups = []
    start = 0

    while start < len(a):
        end = start + group_len
        groups.append(a[start:end])
        start += group_len - overlap
    return groups


def get_consecutive_pairs(lst):
    return [(lst[i], lst[i+1]) for i in range(len(lst) - 1)]


def dynamic_counter():
    seen_elems = {}

    def count_elements(s):
        if s not in seen_elems:
            seen_elems[s] = 0
        seen_elems[s] += 1
        return seen_elems[s]
    
    return count_elements
    

class DotDict:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                value = DotDict(value)  # Recursively convert nested dictionaries
            setattr(self, key, value)


def dict_to_args(dictionary):
    return DotDict(dictionary)


def len_dict_of_lists(d):
    return sum(len(v) for v in d.values() if isinstance(v, list))


def update_dict_of_lists(d, key, value, only_unique_value=False):
    if key not in d.keys():
        d[key] = []
    
    do_append = value not in d[key] if only_unique_value else True

    if do_append:
        d[key].append(value)


################################################################################################
# ARRAY PROCESSING
################################################################################################

def print_array_info(arr_list, name_list=None):
    if not isinstance(arr_list, (list, tuple)):
        arr_list = [arr_list]
    
    name_list = [i for i in range(len(arr_list))] if name_list is None else name_list
    for i in range(len(arr_list)):
        arr = arr_list[i]
        if hasattr(arr, 'min') and hasattr(arr, 'max') and hasattr(arr, 'shape'):
            min_val, max_val = arr.min(), arr.max()
            if hasattr(min_val, 'item'):
                min_val = min_val.item()
            if hasattr(max_val, 'item'):
                max_val = max_val.item()
            shape = tuple(arr.shape)
            print(f'ARR {name_list[i]} INFO: {shape}, [min,max]=[{min_val},{max_val}], range={max_val-min_val}.')
        else:
            print(f'ARR {name_list[i]} INFO: Not an array, type {type(arr)}.')


def unique(data, output_type='list'):
    out = set(data)
    if output_type == 'tuple':
        return tuple(out)
    elif output_type == 'numpy':
        return np.array(list(out))
    elif output_type == 'tensor':
        return torch.tensor(list(out))
    elif output_type == 'set':
        return out
    else:
        return list(out)
    

def stable_linear_transform(x, y_min=0, y_max=1, x_min=None, x_max=None, do_clip=True):
    x_min = x.min() if x_min is None else x_min
    x_max = x.max() if x_max is None else x_max
    if hasattr(x, 'clip'):
        x = x.clip(x_min, x_max)
    else:  # a single number
        x = max(min(x, x_max), x_min)
    x_normalized = (x - x_min) / (x_max - x_min)
    y = y_min + (y_max - y_min) * x_normalized
    return y


def is_integer_after_conv(n, num_convs, kernel_size=3, stride=2, padding=1, dilation=1):
    def get_conv_out_size(n):
        return ((n + 2 * padding - dilation * (kernel_size - 1) - 1) / stride) + 1

    for _ in range(num_convs):
        n = get_conv_out_size(n)
        if n != int(n):  # Check if it's not an integer during intermediate steps
            return False
    return True


################################################################################################
# IMAGE PROCESSING
################################################################################################


def arrange_corner_points(points):
    if len(points) != 4:
        raise ValueError("The input must contain exactly four points.")

    # Sort points by their y-coordinates first (top to bottom), and then by x-coordinates (left to right)
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))

    # The first two points in the sorted list will be the upper points
    upper_points = sorted(sorted_points[:2], key=lambda p: p[0])  # Sort by x to get upper-left and upper-right
    # The last two points in the sorted list will be the lower points
    lower_points = sorted(sorted_points[2:], key=lambda p: p[0])  # Sort by x to get lower-left and lower-right

    # Combine the points in the desired order
    arranged_points = [upper_points[0], upper_points[1], lower_points[1], lower_points[0]]

    return arranged_points


def pad_images_to_max_size(images, pad_value=0):
    max_height = max(img.shape[0] for img in images)
    max_width = max(img.shape[1] for img in images)
    padded_images = []
    
    for img in images:
        pad_height = max_height - img.shape[0]
        pad_width = max_width - img.shape[1]
        padded_img = np.pad(
            img,
            pad_width=((0, pad_height), (0, pad_width)),
            mode='constant',
            constant_values=pad_value)
        padded_images.append(padded_img)
    
    return padded_images


def pseudo_color_array(arr, cmap='seismic'):
    assert len(arr.shape) == 2, f'Input should be 2-dim (x,y), got {arr.shape}.'
    assert (arr.min() >= 0) and (arr.max() <= 1), f'Input should be normalized to [0, 1].'
    colormap = plt.get_cmap(cmap)
    pc_arr = colormap(arr)  # Returns an RGBA (4-channel) image
    pc_arr = (pc_arr[:, :, :3] * 255).astype(np.uint8)
    return pc_arr


def get_volume_preview(
    volume, do_pad=True, is_depth_first=True, 
    input_min=-1000, input_max=1000, 
    concat_output=True, concat_axis=1, to_uint8=False,
    cmap=None):
    
    xval_dict = dict(x_min=input_min, x_max=input_max)
    
    if is_depth_first:
        d, h, w = volume.shape
        slices = [volume[d//2], volume[:,h//2], volume[:,:,w//2]]
    else:
        h, w, d = volume.shape
        slices = [volume[h//2], volume[:,w//2], volume[:,:,d//2]]
        
    if do_pad:
        slices = pad_images_to_max_size(slices, pad_value=input_min)
    
    if cmap is not None:
        slices = [pseudo_color_array(
            stable_linear_transform(
                v, y_min=0, y_max=1, **xval_dict
            ), cmap=cmap) for v in slices]  # [H,W,C], [0, 255]
    else:
        slices = [stable_linear_transform(
            v, y_min=0, y_max=255, **xval_dict
        ) for v in slices]
    
    if to_uint8:
        slices = [v.astype(np.uint8) for v in slices]
    
    if concat_output:
        slices = np.concatenate(slices, axis=concat_axis)
    
    return slices


def concat_pil_images(images, save_path=None):
    """
    Concatenate a list of PIL images along the width dimension.
    """
    if len(images) < 2:
        return images

    # Ensure all images have the same height and mode
    heights = [img.size[1] for img in images]
    modes = [img.mode for img in images]
    if len(set(heights)) > 1:
        raise ValueError("All images must have the same height.")
    if len(set(modes)) > 1:
        images = [im.convert('L') for im in images]
        #raise ValueError("All images must have the same mode (e.g., RGB, L).")

    # Compute the dimensions of the new image
    total_width = sum(img.size[0] for img in images)
    height = images[0].size[1]  # All images have the same height

    # Create a new blank image with the concatenated size
    new_image = Image.new(images[0].mode, (total_width, height))

    # Paste each image into the new image
    current_x = 0
    for img in images:
        new_image.paste(img, (current_x, 0))
        current_x += img.size[0]

    # Save the result if a path is provided
    if save_path is not None:
        new_image.save(save_path)

    return new_image


def show_multiple_images(
    images, 
    nrows=1, 
    ncols=None, 
    titles=None, 
    suptitle=None, 
    tight=True, 
    cmaps='gray', 
    figsize=None, 
    dpi=None, 
    masks=None,
    set_axis_off=True,
    return_pil=False,
):

    num_imgs = len(images)
    ncols = true_round(num_imgs / nrows) if ncols is None else ncols
    num_plots = int(nrows * ncols)
    cmaps = [cmaps] * num_imgs if not isinstance(cmaps, (tuple, list)) else cmaps
    masks = [masks] * num_imgs if not isinstance(masks, (tuple, list)) else masks
    titles = [titles] * num_imgs if not isinstance(titles, (tuple, list)) else titles
    assert num_imgs <= num_plots, f'num_imgs = {num_imgs}, nrows = {nrows}, ncols = {ncols}.'
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, figsize=figsize, dpi=dpi)
    axes = axes.flatten()
    for i in range(num_imgs):
        if isinstance(images[i], torch.Tensor):
            img = images[i].cpu().squeeze()
        elif isinstance(images[i], np.ndarray):
            img = images[i].squeeze()
        elif isinstance(images[i], Image.Image):
            img = np.array(images[i]).squeeze()
        else:
            img = images[i]
        
        mask = masks[i]
        if mask is not None:
            img[mask == 0] = img.min()
        axes[i].imshow(img, cmap=cmaps[i])
        if set_axis_off:
            axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(titles[i])
    
    if suptitle is not None:
        fig.suptitle(suptitle)
    
    if num_imgs < num_plots:
        for i in range(num_imgs, num_plots):
            axes[i].axis('off')
            
    if tight:
        fig.tight_layout()
    
    if return_pil:
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=dpi)
        buf.seek(0)
        pil_image = Image.open(buf).convert('RGB')
        buf.close()
        plt.close(fig)  # Close the figure to free memory
        return pil_image
    
    return None



def grayscale_to_cmap(x, cmap_name='viridis'):
    assert (0 <= x.min() <= 1) and (0 <= x.max() <= 1)
    cmap = cm.get_cmap(cmap_name)
    orig_shape = list(x.shape)
    new_shape = orig_shape + [4,]
    colored_x = cmap(x.flatten()).reshape(*new_shape)[..., :3]  # [*orig_shape, 3]
    # The resulting array has shape (height, width, 4) where the last dimension is RGBA
    # We can drop the alpha channel to get an RGB image
    return colored_x


################################################################################################
# DATAFRAME PROCESSING
################################################################################################


def concat_dataframes(dataframes, fill_value="", ignore_index=True):
    """
    Concatenates multiple DataFrames, aligning columns and padding missing columns with empty values.
    """
    if len(dataframes) < 2:
        raise ValueError("At least two DataFrames are required for concatenation.")
    
    all_columns = set().union(*[df.columns for df in dataframes])
    
    # Align all DataFrames to have the same columns
    aligned_dfs = [df.reindex(columns=all_columns, fill_value=fill_value) for df in dataframes]
    result = pd.concat(aligned_dfs, ignore_index=ignore_index)
    return result


################################################################################################
# DEEP LEARNING THINGS
################################################################################################

def fix_seed(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False



################################################
# DISTRIBUTED DATA PARALLEL
################################################

def get_nvml_info():
    handle = pynvml.nvmlDeviceGetHandleByIndex(0) # 0表示显卡标号
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    total_mem = meminfo.total/1024**3
    free_mem = meminfo.free/1024**3
    return total_mem, free_mem


def dprint(msg, local_rank=0):
    if dist.is_initialized():
        if dist.get_rank() == local_rank:
            print(msg)
    else:
        print(msg)


def only_on_rank0(func):
    '''wrapper for only log on the first rank'''
    def wrapper(self, *args, **kwargs):
        if self.rank != 0:
            return
        return func(self, *args, **kwargs)
    return wrapper



################################################################################################
# OTHERS
################################################################################################

def get_now(num=19):
    return str(datetime.now())[:num]

def print_dict(d, keys=None, indent=2):
    orig_keys = list(d.keys())
    d_ = d
    if keys is not None:
        keys = [keys] if isinstance(keys, str) else keys
        keys = [k for k in keys if k in orig_keys]
        d_ = {k:d[k] for k in keys}
    print(json.dumps(d_, indent=indent))


def print_and_log(message, log_type='info', logger=None):
    print(message)
    if log_type == 'error':
        if logger is None:
            logging.error(message)
        else:
            logger.error(message)
    elif log_type == 'warn':
        if logger is None:
            logging.warning(message)
        else:
            logger.warning(message)
    else:
        if logger is None:
            logging.info(message)
        else:
            logger.info(message)

