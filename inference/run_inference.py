# -*- coding: utf-8 -*-
import os
import sys
sys.path.append('..')

import time
import torch
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from utilities.helpers import read_json, write_json, concat_pil_images, dict_to_args, fix_seed

fix_seed(1234)


from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation import LogitsProcessorList, PrefixConstrainedLogitsProcessor, UnbatchedClassifierFreeGuidanceLogitsProcessor
from emu3.mllm.processing_ti2ti import Emu3Processor
from emu3.visionvq import Emu3VisionVQModel, Emu3VisionVQImageProcessor
from emu3.mllm import Emu3Config, Emu3Tokenizer
from emu3.train.datasets_ti2ti import ProgEmuDataset
from emu3.mllm.modeling_ti2ti import Emu3ForCausalLM


WORKSPACE="path/to/progemu/workspace"
DATA_ROOT="root/directory/of/data"
TOK_HUB = os.path.join(WORKSPACE, 'weights/stage1')
VQ_HUB = os.path.join(WORKSPACE, 'weights/visiontokenizer')
DEFAULT_EMU_HUB = os.path.join(WORKSPACE, 'weights/stage1')
NEGATIVE_PROMPT = ""


parser = argparse.ArgumentParser(description="Control the script behavior with a command-line argument.")
parser.add_argument("--model_path", type=str, default=DEFAULT_EMU_HUB, help="Emu3 weights")
parser.add_argument("--meta_path", type=str, default=None, help="Path to metadata (in a JSON file)")
parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the output")
parser.add_argument("--do_sample", default=False, action='store_true', help="Whether to do sampling")
parser.add_argument("--use_cfg_proc", default=False, action='store_true', help="Whether to use classifier-free guidance")
parser.add_argument("--use_pc_proc", default=False, action='store_true', help="Whether to use prefix constrained processor")
parser.add_argument("--use_template", default=False, action='store_true', help="Whether to use template for input prompt")
parser.add_argument("--only_output_vision", default=False, action='store_true', help="Only output vision tokens")
parser.add_argument("--only_output_text", default=False, action='store_true', help="Only output textual tokens")
parser.add_argument("--cfg", default=3, type=float, help="Classifier-free guidance scale")
parser.add_argument("--start_idx", default=0, type=int)

args = parser.parse_args()

META_PATH = args.meta_path
assert META_PATH is not None, "Please specify metadata path."

MODEL_PATH = args.model_path
START_IDX = args.start_idx
OUTPUT_DIR = args.output_dir
if OUTPUT_DIR is None:
    exp_name = '/'.join(MODEL_PATH.split('/')[-2:])
    OUTPUT_DIR = os.path.join(WORKSPACE, 'outputs', exp_name)
os.makedirs(OUTPUT_DIR, exist_ok=True)


DO_SAMPLE = args.do_sample
USE_CFG_PROC = args.use_cfg_proc
USE_PC_PROC = args.use_pc_proc
USE_TEMPLATE = args.use_template
CFG_SCALE = args.cfg
ONLY_OUTPUT_VISION = args.only_output_vision
ONLY_OUTPUT_TEXT = args.only_output_text


def prepare_models(use_template=False, only_output_vision=True, only_output_text=False):
    model_config = Emu3Config.from_pretrained(MODEL_PATH)
    model = Emu3ForCausalLM.from_pretrained(
        MODEL_PATH,
        config=model_config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2")
    model.eval()

    tokenizer = Emu3Tokenizer.from_pretrained(
        TOK_HUB, 
        model_max_length=model.config.max_position_embeddings, 
        padding_side="left")
    image_processor = Emu3VisionVQImageProcessor.from_pretrained(VQ_HUB, trust_remote_code=True)
    image_tokenizer = Emu3VisionVQModel.from_pretrained(VQ_HUB, device_map="cuda").eval()
    processor = Emu3Processor(image_processor, image_tokenizer, tokenizer, only_output_vision=only_output_vision)

    visual_token_folder = "emu3_token_ids"
    path_prefix = None
    dataset_class = ProgEmuDataset

    data_args = dict_to_args(dict(
        data_path=META_PATH,
        path_prefix=path_prefix,
        visual_token_folder=visual_token_folder,
        null_prompt_prob=0,
        ignore_index=-100,
        visual_token_pattern="<|visual token {token_id:0>6d}|>",
        codebook_size=32768,
        use_template=use_template,
        apply_loss_on_only_vision=only_output_vision,
        apply_loss_on_only_text=only_output_text,
        shuffle_sentence_prob=0,
        dropneg_sentence_prob=0
    ))
    dataset = dataset_class(data_args, tokenizer)
    return model, processor, dataset


def prepare_input(dataset, processor, index, force_input_text=None):
    image_tokenizer = processor.vision_tokenizer
    image_processor = processor.image_processor
    meta_path = index if isinstance(index, str) else dataset.filelist[index]

    (
        input_image_name, 
        input_text, 
        gt_image_name, 
        gt_text
    ) = dataset.get_meta(meta_path)
    
    if force_input_text is not None:
        input_text = force_input_text
    input_image_tokens = dataset.get_image_tokens(input_image_name)
    gt_image_tokens = dataset.get_image_tokens(gt_image_name)
    
    with torch.no_grad():
        input_image = image_tokenizer.decode(input_image_tokens[None].to("cuda"))  # .to(image_tokenizer.device)
        input_image = image_processor.postprocess(input_image)["pixel_values"][0]
        gt_image = image_tokenizer.decode(gt_image_tokens[None].to("cuda"))  # .to(image_tokenizer.device)
        gt_image = image_processor.postprocess(gt_image)["pixel_values"][0]
        
    data = {
        'input_text': input_text, 'input_image': input_image, 
        'gt_text': gt_text, 'gt_image': gt_image, 
        'meta_path': meta_path,
    }

    kwargs = dict(return_tensors="pt", padding="longest")
    pos_inputs = processor(
        text=input_text, 
        image=input_image_tokens, 
        **kwargs
    )
    
    return pos_inputs, data



def prepare_config(model, do_sample=True):
    if do_sample:
        return GenerationConfig(
            use_cache=True,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            max_new_tokens=9216, #model.config.max_position_embeddings,#40960,
            do_sample=True,
            top_k=2048)
    else:
        return GenerationConfig(
            use_cache=True,
            eos_token_id=model.config.eos_token_id,
            pad_token_id=model.config.pad_token_id,
            max_new_tokens=9216, #model.config.max_position_embeddings,#40960,
            do_sample=False,)

def get_proc_prefix(use_pc_proc, use_cfg_proc, cfg):
    if use_pc_proc and (not use_cfg_proc):
        return 'pc'
    elif use_pc_proc and use_cfg_proc:
        return f'pc-cfg{cfg}'
    elif (not use_pc_proc) and use_cfg_proc:
        return f'cfg{cfg}'
    else:
        return ''

def prepare_logits_processor(model, processor, pos_inputs, use_pc_proc=True, use_cfg_proc=False, cfg=3.0):
    h = pos_inputs.image_size[:, 0]
    w = pos_inputs.image_size[:, 1]
    proc_list = None
    proc_prefix = ''
    device = model.device
    kwargs = dict(return_tensors="pt", padding="longest")
    neg_inputs = processor(text=NEGATIVE_PROMPT, **kwargs)
    neg_input_ids = neg_inputs.input_ids.to(device)
    constrained_fn = processor.build_prefix_constrained_fn(h, w)
    proc_prefix = get_proc_prefix(use_pc_proc, use_cfg_proc, cfg)

    if use_pc_proc and (not use_cfg_proc):
        proc_list = LogitsProcessorList([
            PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1)
        ])
    elif use_pc_proc and use_cfg_proc:
        proc_list = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(cfg, model, unconditional_ids=neg_input_ids),
            PrefixConstrainedLogitsProcessor(constrained_fn, num_beams=1,)
        ])
    elif (not use_pc_proc) and use_cfg_proc:
        proc_list = LogitsProcessorList([
            UnbatchedClassifierFreeGuidanceLogitsProcessor(cfg, model, unconditional_ids=neg_input_ids)
        ])

    return proc_list, proc_prefix


def run_generation(model, pos_inputs, gen_config, logits_processor=None, use_mask=True):
    input_ids = pos_inputs.input_ids
    attn_mask = pos_inputs.attention_mask

    if use_mask:
        outputs = model.generate(
            input_ids.clone().to("cuda"),
            gen_config,
            logits_processor=logits_processor,
            attention_mask=attn_mask.clone().to("cuda"))
    else:
        outputs = model.generate(
            input_ids.clone().to("cuda"),
            gen_config,
            logits_processor=logits_processor)
    
    return outputs




print(f"""
Inference data args: 
  - meta_path={META_PATH}
  - use_template={USE_TEMPLATE}
  - only_output_vision={ONLY_OUTPUT_VISION}
  - only_output_text={ONLY_OUTPUT_TEXT}
Inference sampling args: 
  - do_sample={DO_SAMPLE}
  - use_pc_proc={USE_PC_PROC}
  - use_cfg_proc={USE_CFG_PROC}
  - cfg={CFG_SCALE}
""")

data_paths = read_json(META_PATH)
data_paths = sorted(list(set(data_paths)))[START_IDX:]
model, processor, dataset = prepare_models(USE_TEMPLATE, ONLY_OUTPUT_VISION, only_output_text=ONLY_OUTPUT_TEXT)
proc_prefix = get_proc_prefix(USE_PC_PROC, USE_CFG_PROC, CFG_SCALE)


for data_path in tqdm(data_paths):
    meta_id = data_path.split('/')[-1].replace('.json', '')
    save_image_path = os.path.join(OUTPUT_DIR, f'{meta_id}_{proc_prefix}.png')
    save_json_path = os.path.join(OUTPUT_DIR, f'{meta_id}_{proc_prefix}.json')
    save_token_path = os.path.join(OUTPUT_DIR, f'{meta_id}_{proc_prefix}.pt')
    if os.path.exists(save_image_path) and os.path.exists(save_json_path) and os.path.exists(save_token_path):
        continue
    
    pos_inputs, data = prepare_input(dataset, processor, index=data_path)
    torch.cuda.empty_cache()
    logits_processor, proc_prefix = prepare_logits_processor(
        model, processor, pos_inputs, 
        cfg=CFG_SCALE, 
        use_cfg_proc=USE_CFG_PROC, 
        use_pc_proc=USE_PC_PROC
    )
    gen_config = prepare_config(model, do_sample=DO_SAMPLE)

    try:
        outputs = run_generation(
            model, pos_inputs, gen_config, 
            logits_processor=logits_processor, 
            use_mask=True
        )
    except Exception as e:
        print(f'Retrying due to error: {e}')
        pos_inputs, data = prepare_input(dataset, processor, index=data_path)
        logits_processor, proc_prefix = prepare_logits_processor(
            model, processor, pos_inputs, 
            cfg=CFG_SCALE, 
            use_cfg_proc=USE_CFG_PROC, 
            use_pc_proc=USE_PC_PROC
        )
        outputs = run_generation(
            model, pos_inputs, gen_config, 
            logits_processor=logits_processor, 
            use_mask=True
        )

    # post-processing
    input_ids = pos_inputs['input_ids'].squeeze()
    output_ids = outputs.squeeze()[len(input_ids):]  # remove the input part
    seq = processor.tokenizer.decode(output_ids)
    visual_token_rows, textual_tokens = processor.split_visual_textual_tokens(seq, keep_text_as_is=False)

    if len(visual_token_rows) == 0:
        image_list = [Image.fromarray(np.uint8(np.zeros((512,512))))]
    else:
        tensor = processor.get_tensor_from_visual_token_rows(visual_token_rows, fit_type='global')
        with torch.no_grad():
            image = processor.vision_tokenizer.decode(
                tensor.to(processor.vision_tokenizer.device)).float()
        image_list = processor.image_processor.postprocess(image)["pixel_values"]

    text_results = {
        'meta_path': data['meta_path'],
        'input_text': data['input_text'],
        'gt_text': data['gt_text'],
        'gen_text': textual_tokens}
    write_json(text_results, save_json_path)

    image_results = [data['input_image'], data['gt_image']] + image_list
    image_results = [im.convert('L') for im in image_results]
    image_results = concat_pil_images(image_results)
    image_results.save(save_image_path)
    torch.save(outputs, save_token_path)


