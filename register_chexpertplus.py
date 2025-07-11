import os
import cv2  # pip install opencv-python-headless
import json
import shutil
import numpy as np
import SimpleITK as sitk

from PIL import Image
from tqdm import tqdm
from recon_metrics import SSIM
#from skimage.restoration.inpaint import inpaint_biharmonic

SSIM_FUNC = SSIM(data_range=255, channel=1)

def read_json(json_path, encoding='utf-8'):
    with open(json_path, 'r', encoding=encoding) as f:
        data = json.load(f)
    return data

def write_json(data, json_path, write_mode='w', encoding='utf-8', ensure_ascii=False):
    with open(json_path, write_mode, encoding=encoding) as f:
        json.dump(data, f, indent=4, ensure_ascii=ensure_ascii)


def find_files_recursively(directory, ext='', inclusions='', exclusions=None):
    matched_paths = []
    
    # Ensure inclusions and exclusions are lists
    if not isinstance(inclusions, (list, tuple)):
        inclusions = [inclusions]
    if exclusions is not None and not isinstance(exclusions, (list, tuple)):
        exclusions = [exclusions]

    if not os.path.exists(directory):
        raise ValueError(f'Path does not exist: {directory}.')

    for root, dirs, files in os.walk(directory):
        filtered_dirs = []
        
        for d in dirs:
            full_path = os.path.join(root, d)
            # Only treat it as a file if it's actually a file
            if d.endswith('.nii.gz') and os.path.isfile(full_path):
                files.append(d)  # Treat it as a file
            else:
                filtered_dirs.append(d)  # Keep directories that are actually directories

        dirs[:] = filtered_dirs  # Modify dirs in place to control recursion

        for file in files:
            cond1 = all(s in file for s in inclusions)
            cond2 = file.endswith(ext)
            cond3 = all(s not in file for s in exclusions) if exclusions else True

            if cond1 and cond2 and cond3:
                matched_paths.append(os.path.join(root, file))

    return sorted(matched_paths)


def extract_foreground(img, threshold=10, border_foreground_ratio=0.3):
    """
    Some CXR images look like "rotated" or "resized", having a "border" of 
    black pixels (which we call "background") around the real CXR content 
    (which we call "foreground"). 
    """
    
    # Check if image needs processing (has black background)
    border_pixels = np.concatenate([img[0, :], img[-1, :], img[:, 0], img[:, -1]])
    if np.mean(border_pixels > threshold) > border_foreground_ratio:  # non-black pixel ratio >
        return img  # No processing needed
    
    # Create a binary mask and find countours
    _, binary = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return img
    
    # Find the largest contour (assumed to be the main content)
    main_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(main_contour)  # Get the minimum area rectangle
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    height, width = img.shape
    
    # The transform requires the positions of points in `src_pts` align with those in `dst_pts`,
    # so let's make them [upper left, upper right, lower right, lower left]. 
    # Note that each point in `box` is arranged as [x, y] or [w, h], not [h, w].
    src_pts = np.array(arrange_corner_points(box.tolist()), dtype="float32")
    dst_pts = np.array([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (width, height))
    return warped


def arrange_corner_points(points):
    
    if len(points) != 4:
        raise ValueError("The input must contain exactly four points.")
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))

    # The first two points in the sorted list will be the upper points
    upper_points = sorted(sorted_points[:2], key=lambda p: p[0])  # Sort by x to get upper-left and upper-right
    # The last two points in the sorted list will be the lower points
    lower_points = sorted(sorted_points[2:], key=lambda p: p[0])  # Sort by x to get lower-left and lower-right
    # Final order: upper left, upper right, lower right, lower left (clockwise)
    return (upper_points[0], upper_points[1], lower_points[1], lower_points[0])


def get_black_pixel_ratio(arr: np.ndarray, return_str=False):
    """Compute the ratio of black pixels in an image"""
    r = np.sum(arr==0) / arr.size
    return str(r)[:6] if return_str else r


def register_arrays(fixed_arr, moving_arr, use_affine=True, use_mse=True, brute_force=False, return_metric=False, **kwargs):
    """Register two images stored in numpy arrays, within the range of [0, 255]"""
    min_val = float(moving_arr.min())
    fixed_image = sitk.GetImageFromArray(fixed_arr)
    moving_image = sitk.GetImageFromArray(moving_arr)
    transform = sitk.AffineTransform(2) if use_affine else sitk.ScaleTransform(2)
    initial_transform = sitk.CenteredTransformInitializer(
        sitk.Cast(fixed_image, moving_image. GetPixelID()), 
        moving_image, 
        transform, 
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    ff_img = sitk.Cast(fixed_image, sitk.sitkFloat32)
    mv_img = sitk.Cast(moving_image, sitk.sitkFloat32)
    registration_method = sitk.ImageRegistrationMethod()
    if use_mse:
        registration_method.SetMetricAsMeanSquares()
    else:
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    
    if brute_force:
        sample_per_axis = 12
        registration_method.SetOptimizerAsExhaustive([sample_per_axis//2,0,0])
        # Utilize the scale to set the step size for each dimension
        registration_method.SetOptimizerScales([2.0*3.14/sample_per_axis, 1.0,1.0])
    else:
        registration_method.SetMetricSamplingStrategy(registration_method.RANDOM)
        registration_method.SetMetricSamplingPercentage(0.25)

    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0, 
        numberOfIterations=200, 
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10)
    # Scale the step size differently for each parameter, this is critical!!!
    registration_method.SetOptimizerScalesFromPhysicalShift() 

    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    final_transform_v1 = registration_method.Execute(ff_img, mv_img)
    metric_value = registration_method.GetMetricValue()
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(fixed_image)
    resample.SetDefaultPixelValue(min_val)
    resample.SetInterpolator(sitk.sitkBSpline)  
    resample.SetTransform(final_transform_v1)
    reg_img = sitk.GetArrayFromImage(resample.Execute(moving_image))
    return (reg_img, metric_value) if return_metric else reg_img



def complex_register_images(input_img1, input_img2, ssim_func=SSIM_FUNC, thres_ratio=1.2, lowthres_ssim=0.2, upthres_ssim=0.9):    
    def check_if_reregister(fixed_img, moving_img, reg_img, have_exchanged, reg_metric=-100):
        ratio3 = get_black_pixel_ratio(reg_img)
        # Step 2.1: Registration passes
        if ratio3 < thres_ratio:
            if have_exchanged:  
                # meaning fixed_img is actually img2 (followup), while moving_img/reg_img are img1 (reference)
                return {"reference": reg_img, "followup": fixed_img, "flag": 1, "reg_metric": reg_metric}
            else:
                return {"reference": fixed_img, "followup": reg_img, "flag": 1, "reg_metric": reg_metric}

        # Step 2.2: Consider re-registration
        fixed_img_tmp, moving_img_tmp = moving_img, fixed_img  # [NOTE] Here we do another exchange!
        reg_img_tmp, reg_metric_tmp = register_arrays(fixed_img_tmp, moving_img_tmp, return_metric=True)
        ratio3_tmp = get_black_pixel_ratio(reg_img_tmp)
        if ratio3_tmp < thres_ratio: 
            have_exchanged = (not have_exchanged)  # the exchange is applied
            if have_exchanged:
                return {"reference": reg_img_tmp, "followup": fixed_img_tmp, "flag": 1, "reg_metric": reg_metric_tmp}
            else:
                return {"reference": fixed_img_tmp, "followup": reg_img_tmp, "flag": 1, "reg_metric": reg_metric_tmp}

        if ratio3_tmp < ratio3:
            have_exchanged = (not have_exchanged)  # the exchange is applied
            if have_exchanged:
                return {"reference": reg_img_tmp, "followup": fixed_img_tmp, "flag": 0, "reg_metric": reg_metric_tmp}
            else:
                return {"reference": fixed_img_tmp, "followup": reg_img_tmp, "flag": 0, "reg_metric": reg_metric_tmp}
        else: # the exchange is NOT applied
            if have_exchanged:
                return {"reference": reg_img, "followup": fixed_img, "flag": 0, "reg_metric": reg_metric}
            else:
                return {"reference": fixed_img, "followup": reg_img, "flag": 0, "reg_metric": reg_metric}

    # Step 1: Compute black pixel ratios and handle initial interpolation
    have_exchanged = False
    img1 = input_img1 
    img2 = input_img2 

    ratio1 = get_black_pixel_ratio(img1)
    ratio2 = get_black_pixel_ratio(img2)
    ssim_before = ssim_func(img1, img2)

    # Step 1.1: Check if registration is unnecessary due to low quality
    if ratio1 > thres_ratio and ratio2 > thres_ratio or ssim_before < lowthres_ssim:
        return {"reference": img1, "followup": img2, "flag": -1, "reg_metric": -100}

    # Step 1.2: Both images have low black pixel ratios
    if ratio1 <= thres_ratio and ratio2 <= thres_ratio:
        if ssim_before < upthres_ssim:
            have_exchanged = (ratio1 > ratio2)
            fixed_img, moving_img = (img2, img1) if have_exchanged else (img1, img2)
            reg_img, reg_metric = register_arrays(fixed_img, moving_img, return_metric=True)
            return check_if_reregister(fixed_img, moving_img, reg_img, have_exchanged, reg_metric=reg_metric)
        else:
            return {"reference": img1, "followup": img2, "flag": 2, "reg_metric": 0}

    # Step 1.3: Only one image has a low black pixel ratio
    have_exchanged = (ratio1 > thres_ratio)
    fixed_img, moving_img = (img2, img1) if have_exchanged else (img1, img2)
    reg_img, reg_metric = register_arrays(fixed_img, moving_img, return_metric=True)
    return check_if_reregister(fixed_img, moving_img, reg_img, have_exchanged, reg_metric=reg_metric)



if __name__ == '__main__':
    from tqdm.contrib.concurrent import process_map
    
    icgcxr_chexp_dir = './chexpertplus'
    # Recursively find all JSON files in the ICG-CXR (CheXpertPlus Ext.) directory
    icgcxr_chexp_meta_paths = find_files_recursively(icgcxr_chexp_dir, ext='.json')
    
    def run_one(meta_path):
        img_path1 = meta_path.replace('.json', '-ref-init.png')
        img_path2 = meta_path.replace('.json', '-flu-init.png')
        
        if not (os.path.exists(img_path1) and os.path.exists(img_path2)):
            raise ValueError(f'Source image files not found for {meta_path}. Please run `process_chexpertplus.ipynb` first to retrieve the CXR images from the CheXpertPlus dataset.')
        
        img1 = Image.open(img_path1).convert('L')
        img2 = Image.open(img_path2).convert('L')
        
        arr1 = extract_foreground(np.array(img1))
        arr1 = cv2.resize(arr1, (512, 512))
        arr2 = extract_foreground(np.array(img2))
        arr2 = cv2.resize(arr2, (512, 512))
        
        data = complex_register_images(arr1, arr2)
        
        reg_flag = int(data['flag'])
        ## reg_flag == 1 is for good registration, while -1 indicates bad registration.
        ## reg_flag == 0 means that the image pair is already registered, so we don't need to care about it.
        ## There might be a few image pairs ending up with reg_flag == -1. In that case, we need to manually
        ## perform the registration again by fine-tuning some registration parameters.
        
        ref_img = Image.fromarray(arr1)
        flu_img = Image.fromarray(arr2)
        reg_ref_img = Image.fromarray(np.uint8(data["reference"]))
        reg_flu_img = Image.fromarray(np.uint8(data["followup"]))
        
        ref_img_path = img_path1.replace('-ref-init.png', '-ref.png')
        flu_img_path = img_path2.replace('-flu-init.png', '-flu.png')
        reg_ref_img_path = img_path1.replace('-ref-init.png', '-ref-reg.png')
        reg_flu_img_path = img_path2.replace('-flu-init.png', '-flu-reg.png')
        
        ref_img.save(ref_img_path)
        flu_img.save(flu_img_path)
        reg_ref_img.save(reg_ref_img_path)
        reg_flu_img.save(reg_flu_img_path)
        
        return meta_path, reg_flag
    
    # Process all files and collect results
    results = process_map(run_one, icgcxr_chexp_meta_paths, max_workers=16, chunksize=1)
    # Filter and save meta paths with reg_flag == -1
    bad_reg_paths = [meta_path for meta_path, reg_flag in results if reg_flag == -1]    

    write_json(bad_reg_paths, 'bad_reg_paths.json')
    print(f"Found {len(bad_reg_paths)} cases with bad registration (reg_flag == -1)")
    