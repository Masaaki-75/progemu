# -*- coding: utf-8 -*-
import os
import json
import random

import torch
from torch.utils.data import Dataset

def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


class ProgEmuDataset(Dataset):
    def __init__(self, args, tokenizer):
        super().__init__()
        self.args = args
        self.use_template = getattr(args, 'use_template', False)
        self.prompt_template = 'You are a helpful assistant. Given a reference chest X-ray image and/or a ' + \
            'condition for disease progression, you should predict the future chest X-ray image ' + \
            'and describe any radiological changes that reflect the disease progression. ' + \
            'USER: Reference image: {input_image_prompt} Condition: {input_text_prompt} ' + \
            'ASSISTANT: Predicted image: {output_image_prompt} Description of changes: {output_text_prompt}'

        self.load_file()
        self.tokenizer = tokenizer
        self.vis_tok_patn = vis_tok_patn = args.visual_token_pattern
        self.codebook_size = codebook_size = args.codebook_size
        
        # vis_tok_patn == "<|visual token {token_id:0>6d}|>"
        self.bov = tokenizer.encode(vis_tok_patn.format(token_id=0))[0]
        self.eov = tokenizer.encode(vis_tok_patn.format(token_id=codebook_size - 1))[0]
        self.bos_token_id = tokenizer(tokenizer.bos_token)['input_ids'][0]
        self.img_token_id = tokenizer.special_tokens[tokenizer.img_token]
        self.pad_token_id = tokenizer.pad_token_id
        self.eoi_token_id = tokenizer(tokenizer.eoi_token)['input_ids'][0]
        self.eof_token_id = tokenizer(tokenizer.eof_token)['input_ids'][0]

        self.null_prob = self.args.null_prompt_prob  # UNLOCK CFG
        self.shuffle_sentence_prob = self.args.shuffle_sentence_prob 
        self.dropneg_sentence_prob = self.args.dropneg_sentence_prob
        self.apply_loss_on_only_vision = self.args.apply_loss_on_only_vision
        self.apply_loss_on_only_text = self.args.apply_loss_on_only_text
        print(f'[DATASET] Probabilities: null_prob={self.null_prob}, shuffle_sen_prob={self.shuffle_sentence_prob}, dropneg_sen_prob={self.dropneg_sentence_prob}')
        print(f'[DATASET] Supervision: vision_only={self.apply_loss_on_only_vision}, text_only={self.apply_loss_on_only_text}')
        print(f'[DATASET] Dataset Length: {self.__len__()}')

    def load_file(self):
        args = self.args
        path_prefix = args.path_prefix
        d = read_json(args.data_path)

        if isinstance(d, dict):
            path_prefix = d["prefix"] if "prefix" in d else path_prefix
            self.filelist = d["path_list"] if self.is_invalid_dir(path_prefix) else [os.path.join(path_prefix, _) for _ in d["path_list"]]
        elif isinstance(d, (list, tuple)):
            self.filelist = d if self.is_invalid_dir(path_prefix) else [os.path.join(path_prefix, _) for _ in d]
        else:
            raise ValueError(f"Data loaded from {args.data_path} should be dict/tuple/list")
        
        self.path_prefix = path_prefix
        print(f'[DATASET] Using path prefix:  {self.path_prefix}')

        visual_token_folder = args.visual_token_folder
        if visual_token_folder is not None:
            self.visual_token_dir = os.path.join(path_prefix, visual_token_folder) if not self.is_invalid_dir(path_prefix) else visual_token_folder
        else:
            self.visual_token_dir = path_prefix
        print(f'[DATASET] Using visual token dir:  {self.visual_token_dir}')
    
    @staticmethod
    def is_invalid_dir(d):
        return (d is None) or (not os.path.exists(d))

    def __len__(self):
        return len(self.filelist)
    
    def get_meta(self, path):
        info = read_json(path)
        input_text = info["progression-description"]
        output_text = info["changes-of-findings"]
        input_image_path = path.replace('.json', '-ref-reg.pth')
        output_image_path = path.replace('.json', '-flu-reg.pth')
        return input_image_path, input_text, output_image_path, output_text
    
    def get_image_tokens(self, image_path):
        return torch.load(image_path, map_location='cpu')
    
    def get_prompts_from_meta(self, meta):
        if isinstance(meta, str):
            input_image_name, input_text, output_image_name, output_text = self.get_meta(meta)
        elif isinstance(meta, (tuple, list)):
            input_image_name, input_text, output_image_name, output_text = meta
        else:
            raise ValueError(f'Meta info should either be a file path or a tuple, got {type(meta)}')

        input_image_tokens = self.get_image_tokens(input_image_name)
        output_image_tokens = self.get_image_tokens(output_image_name)
        
        input_text_prompt = input_text
        input_image_prompt = self.format_image_prompt(input_image_tokens)
        output_text_prompt = output_text
        output_image_prompt = self.format_image_prompt(output_image_tokens)
        prompts = {
            "input_image_prompt": input_image_prompt,
            "input_text_prompt": input_text_prompt,
            "output_image_prompt": output_image_prompt,
            "output_text_prompt": output_text_prompt
        }
        return prompts

    def format_multimodal_prompt(self, input_image_prompt, input_text_prompt, output_image_prompt, output_text_prompt=None):
        bos_token = self.tokenizer.bos_token
        prompt_template = self.prompt_template
        only_output_vision = self.args.apply_loss_on_only_vision or (output_text_prompt is None)

        # if only_output_vision:
        #     output_text_prompt = ""
        
        input_text_prompt = self.augment_sentences(input_text_prompt, self.shuffle_sentence_prob, self.dropneg_sentence_prob)
        
        if random.random() < self.null_prob:
            input_image_prompt = input_text_prompt = ""
            
        if self.use_template:
            prompt = bos_token + prompt_template.format(
                input_image_prompt=input_image_prompt,
                input_text_prompt=input_text_prompt,
                output_image_prompt=output_image_prompt,
                output_text_prompt=output_text_prompt)
            return prompt.replace(" Description of changes: ", "") if only_output_vision else prompt
        else:
            return bos_token + input_image_prompt + input_text_prompt + output_image_prompt + output_text_prompt
    
    @staticmethod
    def augment_sentences(input_string, shuffle_sentence_prob=0, dropneg_sentence_prob=0):
        if len(input_string) == 0:
            return input_string
        sentences = input_string.strip().split(". ")
        if len(sentences) == 1:
            return input_string
            
        if sentences[-1].endswith("."): 
            sentences[-1] = sentences[-1][:-1]  # Remove the trailing dot from the last sentence
        else:
            sentences = [s for s in sentences if s]  # Remove empty entries if present
        sentences = list(sentences)  # clone to avoid modifying inputs

        if random.random() < shuffle_sentence_prob:
            sentences = random.sample(sentences, len(sentences))

        if random.random() < dropneg_sentence_prob:
            removal_criteria = ("No", "Unchanged", "Persistent")
            for i, sentence in enumerate(sentences):
                if sentence.startswith(removal_criteria) and len(sentences) > 1:
                    sentences.pop(i)
                    break  # Remove only one sentence

        augmented_string = ". ".join(sentences) + "."
        return augmented_string

    def __getitem__(self, index: int):
        path = self.filelist[index]

        try: 
            prompts = self.get_prompts_from_meta(path)
            seq = self.format_multimodal_prompt(**prompts)
        except Exception as err:
            print(f'Error when loading: {path}')
            prompts = self.get_prompts_from_meta(self.filelist[0])
            seq = self.format_multimodal_prompt(**prompts)
        
        sample = self.tokenizer(seq, padding="max_length", return_token_type_ids=False, return_tensors="pt")
        for k, v in sample.items():
            sample[k] = v.squeeze(0)
        labels = sample["input_ids"]
        mask = self.get_target_mask_v2(labels)
        sample['labels'] = torch.where(mask, labels, self.args.ignore_index)
        sample['are_image_ids'] = torch.logical_and(labels >= self.bov, labels <= self.eov)
        return sample
    
    def get_target_mask(self, token_sequence: torch.Tensor):
        """
        Mask all input images and texts and padding tokens.
        Given an input sequence:
        {bos_token_id}{boi_token_id}{meta_text_token_ids}{content_start_token_id}{source_image_token_ids}{eof_token_id}{eoi_token_id}{source_text_token_ids}{boi_token_id}{meta_text_token_ids}{content_start_token_id}{target_image_token_ids}{eof_token_id}{eoi_token_id}{target_text_token_ids}{pad_token_ids}
        """
        img_token_id = self.img_token_id
        pad_token_id = self.pad_token_id

        # Find all positions of the img_token_id
        img_token_id_positions = (token_sequence == img_token_id).nonzero(as_tuple=False).flatten()
        assert len(img_token_id_positions) >= 1
        target_start_idx = (img_token_id_positions[-1]+1).item()

        # Find the end of the target tokens (i.e., the start of padding or the end of the sequence)
        padding_start_positions = (token_sequence == pad_token_id).nonzero(as_tuple=False).flatten()
        if len(padding_start_positions) > 0:
            target_end_idx = (padding_start_positions[0]).item()
        else:
            target_end_idx = len(token_sequence)
        #target_end_idx += 1  # to learn one <endoftext>

        # Create the mask: 1 for target_image_token_ids and target_text_token_ids, 0 otherwise
        mask = torch.zeros_like(token_sequence, dtype=torch.int)
        mask[target_start_idx:target_end_idx] = 1
        return mask.to(bool)
    

    def get_target_mask_v2(self, token_sequence: torch.Tensor):
        """
        Mask output image or text.
        Given an input sequence:
        {bos_token_id}{boi_token_id}{meta_text_token_ids}{content_start_token_id}{source_image_token_ids}{eof_token_id}{eoi_token_id}{source_text_token_ids}{boi_token_id}{meta_text_token_ids}{content_start_token_id}{target_image_token_ids}{eof_token_id}{eoi_token_id}{target_text_token_ids}{pad_token_ids}
        """
        img_token_id = self.img_token_id
        pad_token_id = self.pad_token_id
        eoi_token_id = self.eoi_token_id
        eof_token_id = self.eof_token_id
        #apply_loss_on_both = (not self.args.apply_loss_on_only_text) and (not self.args.apply_loss_on_only_vision)

        if self.args.apply_loss_on_only_text:
            eoi_token_id_positions = (token_sequence == eoi_token_id).nonzero(as_tuple=False).flatten()
            assert len(eoi_token_id_positions) >= 1
            target_start_idx = (eoi_token_id_positions[-1]+1).item()
        else:
            img_token_id_positions = (token_sequence == img_token_id).nonzero(as_tuple=False).flatten()
            assert len(img_token_id_positions) >= 1
            target_start_idx = (img_token_id_positions[-1]+1).item()

        if self.args.apply_loss_on_only_vision:
            eof_token_id_positions = (token_sequence == eof_token_id).nonzero(as_tuple=False).flatten()
            assert len(eof_token_id_positions) >= 1
            target_end_idx = (eof_token_id_positions[-1]-1).item()
        else:
            padding_start_positions = (token_sequence == pad_token_id).nonzero(as_tuple=False).flatten()
            if len(padding_start_positions) > 0:
                target_end_idx = (padding_start_positions[0]).item()
            else:
                target_end_idx = len(token_sequence)
            #target_end_idx += 1  # to learn one <endoftext>

        # Create the mask: 1 for target_image_token_ids and target_text_token_ids, 0 otherwise
        mask = torch.zeros_like(token_sequence, dtype=torch.int)
        mask[target_start_idx:target_end_idx] = 1
        return mask.to(bool)

    def format_image_prompt(self, image_tokens):
        h, w = image_tokens.shape
        imgstr = self.to_imgstr(image_tokens)

        image_prompt = (
            self.tokenizer.boi_token +  # begin of image, <|image start|>
            f"{h}*{w}" +                # meta info
            self.tokenizer.img_token +  # <|image token|>
            imgstr +                    # image token sequence (multiple lines, separated by many eols)
            self.tokenizer.eol_token +  # end of line
            self.tokenizer.eof_token +  # end of file?
            self.tokenizer.eoi_token    # end of image
        )

        return image_prompt

    def to_imgstr(self, image_tokens):
        """
        Converts the 2D image tokens into a string by formatting each token 
        using visual_token_pattern. Rows are joined with eol_token to form 
        a single sequence.
        """
        vis_tok_patn = self.args.visual_token_pattern
        image_token_str = [[
            vis_tok_patn.format(token_id=token_id) for token_id in token_row]
        for token_row in image_tokens]
        
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr



