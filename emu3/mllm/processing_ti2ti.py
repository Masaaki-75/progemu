# coding=utf-8
# Adapted from https://github.com/baaivision/Emu3/blob/main/emu3/mllm/processing_emu3.py

from math import ceil
import re
import warnings
from typing import List, Optional, Sequence, Union
from functools import partial

from PIL import Image
import torch
from torch.nn import functional as F
from transformers.feature_extraction_utils import BatchFeature
from transformers.processing_utils import ProcessingKwargs, ProcessorMixin
from transformers.tokenization_utils_base import TextInput, PreTokenizedInput
from transformers.utils import logging

from .utils_ti2ti import Emu3PrefixConstrainedLogitsHelper


logger = logging.get_logger(__name__)


class Emu3Processor(ProcessorMixin):
    r"""
    Constructs an Emu3 processor which wraps an Emu3 image processor and an Emu3 vision vq model and an Emu3 tokenizer into a single processor.

    [`Emu3Processor`] offers all the functionalities of [`Emu3VisionVQModel`] and [`Emu3Tokenizer`]. See the
    [`~Emu3Processor.__call__`], [`~Emu3Processor.decode`], [`~Emu3Processor.vision_encode`], [`~Emu3Processor.vision_decode`]
    for more information.

    Args:
        image_processor ([`Emu3VisionVQImageProcessor`]):
            The image processor is a required input.
        vision_tokenizer ([`Emu3VisionVQModel`]):
            The vision tokenizer is a required input.
        tokenizer ([`Emu3Tokenizer`]):
            The tokenizer is a required input.
        prefix_template(`str`, *optional*):
            The prefix template for image tokens
        visual_template(`Tuple[str, ...]`, *optional*):
            The visual token template for image tokens
    """

    attributes = ["image_processor", "tokenizer"]
    valid_kwargs = ["vision_tokenizer", "prefix_template", "visual_template"]
    image_processor_class = "AutoImageProcessor"
    tokenizer_class = "AutoTokenizer"

    def __init__(
        self,
        image_processor=None,
        vision_tokenizer=None,
        tokenizer=None,
        chat_template="You are a helpful assistant. Given a reference chest X-ray image and/or a condition for disease progression, you should predict the future chest X-ray image and describe any radiological changes that reflect the disease progression. USER: Reference image: {input_image_prompt} Condition: {input_text_prompt} ASSISTANT: Predicted image: ",
        prefix_template="{H}*{W}",
        visual_template=("<|visual token {token_id:0>6d}|>", r"<\|visual token (\d+)\|>"),
        only_output_vision=False,
        **kwargs,
    ):
        assert vision_tokenizer is not None, "image tokenizer can not be None"

        self.vision_tokenizer = vision_tokenizer
        self.prefix_template = prefix_template
        self.visual_template = visual_template
        self.vis_tok_spatial_factor = 2 ** (len(self.vision_tokenizer.config.ch_mult) - 1)
        self.only_output_vision = only_output_vision

        super().__init__(image_processor, tokenizer, chat_template=chat_template)
        self.const_helper = self.build_const_helper()

    @torch.no_grad()
    def __call__(
        self,
        text: Optional[TextInput | PreTokenizedInput] = None,
        image: Optional[Image.Image | torch.Tensor | List[Image.Image]] = None,
        image2: Optional[Image.Image | torch.Tensor | List[Image.Image]] = None,
        *,
        padding_image: bool = False,
        use_template: bool = False,
        **kwargs,) -> BatchFeature:
        if isinstance(text, str):
            text = [text]

        assert isinstance(text[0], str), "`text` must be string or list of string"


        if not isinstance(image, (tuple, list)):
            image = [image]
        if not isinstance(image2, (tuple, list)):
            image2 = [image2]

        if isinstance(image[0], torch.Tensor):
            is_image_already_tokenized = True
            assert all([len(im.shape) == 2 for im in image])
        else:
            is_image_already_tokenized = False

        if not is_image_already_tokenized:
            image_tokens = self.tokenize_image(image, padding_image=padding_image) if image[0] is not None else [None,]*len(image)
            # image_tokens shape is [len(image), h, w]
            image_tokens2 = self.tokenize_image(image2, padding_image=padding_image) if image2[0] is not None else [None,]*len(image2)
        else:
            image_tokens = torch.stack(image, dim=0)
            image_tokens2 = torch.stack(image2, dim=0) if image2[0] is not None else [None,]*len(image2)

        if len(text) != len(image_tokens):
            raise ValueError("number of image must match number of text prompt")

        prompt_list, size_list = [], []
        bos_token = self.tokenizer.bos_token
        boi_token = self.tokenizer.boi_token
        img_token = self.tokenizer.img_token
        eol_token = self.tokenizer.eol_token
        eof_token = self.tokenizer.eof_token
        eoi_token = self.tokenizer.eoi_token
        image_postfix = eol_token + eof_token + eoi_token

        for idx, text_prompt in enumerate(text):
            if image_tokens[idx] is not None:
                h, w = image_tokens[idx].shape
                imgstr = self.to_imgstr(image_tokens[idx])
                meta = self.prefix_template.format(H=h, W=w)
                image_trigger = boi_token + meta + img_token
                image_prompt = image_trigger + imgstr + image_postfix
            else:
                h, w = 64, 64
                meta = self.prefix_template.format(H=h, W=w)
                image_trigger = boi_token + meta + img_token
                image_prompt = ""
            
            if image_tokens2[idx] is not None:
                imgstr2 = self.to_imgstr(image_tokens2[idx])
                image_prompt2 = image_trigger + imgstr2 + image_postfix
            else:
                image_prompt2 = image_trigger

            if use_template:
                # Template does not support two image input for now.
                prompt = bos_token + self.chat_template.format(
                    input_image_prompt=image_prompt,
                    input_text_prompt=text_prompt) + image_prompt2
            else:
                prompt = bos_token + image_prompt + text_prompt + image_prompt2

            #prompt += image_trigger
            prompt_list.append(prompt)
            size_list.append([h, w])

        input_prompts = self.tokenizer(prompt_list, **kwargs)
        return BatchFeature(data={**input_prompts, "image_size": size_list}, tensor_type=kwargs.get("return_tensors"))


    @torch.no_grad()
    def batch_decode(self, *args, **kwargs):
        docs = self.tokenizer.batch_decode(*args, **kwargs)
        return [self.multimodal_decode(d) for d in docs]

    @torch.no_grad()
    def decode(self, *args, **kwargs):
        # First we translate the token ids to string, the string
        # should look like <|extra_203|><|image start|>h*w<|image token|><visual token 0xxxx|>...
        # ...
        
        return_doc = kwargs.get('return_doc', False)
        doc = self.tokenizer.decode(*args, **kwargs)
        if return_doc:
            return doc
        return self.multimodal_decode(doc, **kwargs)

    @torch.no_grad()
    def vision_encode(self, *args, **kwargs):
        return self.vision_tokenizer.encode(*args, **kwargs)

    @torch.no_grad()
    def vision_decode(self, *args, **kwargs):
        return self.vision_tokenizer.decode(*args, **kwargs)
    
    @property
    def model_input_names(self):
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))

    def to_imgstr(self, image_tokens):
        image_tokens = image_tokens.cpu().numpy().tolist()
        image_token_str = [
            [
                self.visual_template[0].format(token_id=token_id)
                for token_id in token_row
            ]
            for token_row in image_tokens
        ]
        image_row_str = ["".join(token_row) for token_row in image_token_str]
        imgstr = self.tokenizer.eol_token.join(image_row_str)
        return imgstr

    def tokenize_image(self, image: List[Image.Image], *, padding_image: bool = False):
        is_all_same_size, prev_size = True, None
        
        # Check if all images are the same size.
        for im in image:
            if prev_size is not None:
                is_all_same_size &= (prev_size == im.size)
            prev_size = im.size

        if is_all_same_size:
            # If all images are of the same size, directly perform batch tokenization.
            image_inputs = self.image_processor(image, return_tensors="pt")["pixel_values"]
            image_inputs = image_inputs.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
        elif padding_image:
            # If the images differ in size, and the padding is specified, we pad them to the largest size in this batch before performing batch toeknization.
            image_inputs = [self.image_processor(im, return_tensors="pt")["pixel_values"] for im in image]
            image_shapes = [im.shape[2:] for im in image_inputs]
            max_shape = (
                max([im_shape[0] for im_shape in image_shapes]),
                max([im_shape[1] for im_shape in image_shapes]),
            )
            image_inputs = [
                F.pad(im_inp, (0, max_shape[1] - im_shape[1], 0, max_shape[0] - im_shape[0]))
                for im_inp, im_shape in zip(image_inputs, image_shapes)
            ]
            image_inputs = torch.cat(image_inputs, dim=0).to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
            image_tokens = self.vision_tokenizer.encode(image_inputs)
            image_tokens = [
                im_tok[:ceil(im_shape[0] / self.vis_tok_spatial_factor), :ceil(im_shape[1] / self.vis_tok_spatial_factor)]
                for im_tok, im_shape in zip(image_tokens, image_shapes)
            ]
        else:  # Perform image tokenization individually
            image_tokens = []
            for im in image:
                image_input = self.image_processor(im, return_tensors="pt")["pixel_values"]
                image_input = image_input.to(self.vision_tokenizer.device, self.vision_tokenizer.dtype)
                image_tokens.append(self.vision_tokenizer.encode(image_input).squeeze(0))

        return image_tokens

    def build_const_helper(self):
        (
            img_token,
            eoi_token,
            eos_token,
            eol_token,
            eof_token,
            pad_token,
            vis_start,
            vis_end,
        ) = self.tokenizer.encode([
            self.tokenizer.img_token,
            self.tokenizer.eoi_token,
            self.tokenizer.eos_token,
            self.tokenizer.eol_token,
            self.tokenizer.eof_token,
            self.tokenizer.pad_token,
            self.visual_template[0].format(token_id=0),
            self.visual_template[0].format(token_id=self.vision_tokenizer.config.codebook_size - 1),
        ])

        const_helper = partial(
            Emu3PrefixConstrainedLogitsHelper,
            img_token=img_token,
            eoi_token=eoi_token,
            eos_token=eos_token,
            eol_token=eol_token,
            eof_token=eof_token,
            pad_token=pad_token,
            visual_tokens=list(range(vis_start, vis_end + 1)),
            textual_tokens=list(range(100000)),#list(range(151851))# + useful_tokens
            only_output_vision=self.only_output_vision
        )
        return const_helper

    def build_prefix_constrained_fn(self, height, width):
        helper = self.const_helper(height=height, width=width)
        return helper


    @staticmethod
    def extract_hw(input_string):
        pattern = r"<\|image start\|>(.*?)<\|image token\|>"
        match = re.search(pattern, input_string)
        hw = [64, 64]  # default as [64, 64]
        if match:
            hw_str = match.group(1)  # Get the content inside `<|image start|>` and `<|image token|>`
            hw = [int(num) for num in hw_str.split('*')]  # Split by '*' and convert to integers
        return hw
    
    @torch.no_grad()
    def multimodal_decode(self, doc, **kwargs):
        multimodal_output = {'image':[], 'text':[]}
        remove_specials = kwargs.get('remove_specials', False)
        pattern = rf'({re.escape(self.tokenizer.boi_token)}.*?{re.escape(self.tokenizer.eoi_token)})'
        chunks = re.split(pattern, doc)
        for c in chunks:
            if len(c) == 0:
                continue

            if self.tokenizer.boi_token in c:
                image1 = self.get_image_tensor_from_prompt(c, fit_type='local')
                image1 = self.vision_tokenizer.decode(image1[None].to(self.vision_tokenizer.device)).float()
                image1 = self.image_processor.postprocess(image1)["pixel_values"][0]
                
                image2 = self.get_image_tensor_from_prompt(c, fit_type='global')
                image2 = self.vision_tokenizer.decode(image2[None].to(self.vision_tokenizer.device)).float()
                image2 = self.image_processor.postprocess(image2)["pixel_values"][0]
                multimodal_output['image'].extend([image1, image2])

            else:
                text = self.remove_angle_brackets(c) if remove_specials else c
                if len(text) == 0:
                    continue
                multimodal_output['text'].append(text)

        return multimodal_output if len(multimodal_output) > 1 else multimodal_output[0]

    @staticmethod
    def remove_angle_brackets(input_string):
        # Find all substrings that start with '<' and end with '>'
        subparts = re.findall(r"<.*?>", input_string)
        
        # Replace those substrings with an empty string
        filtered_string = input_string
        for subpart in subparts:
            filtered_string = filtered_string.replace(subpart, "")
        
        # Strip any remaining leading/trailing whitespace
        return filtered_string.strip()
    
    def split_visual_textual_tokens(
        self, input_string, 
        visual_token_pattern=r"(<\|visual token \d+\|>)+",
        keep_text_as_is=False):
        visual_token_matches = list(re.finditer(visual_token_pattern, input_string))
        visual_token_rows, textual_tokens = [], []

        # Pointer to track the last position in the input string
        last_end = 0
        for match in visual_token_matches:
            start, end = match.span()
            # Extract the residual text before the current match
            if start > last_end:
                residual_text = input_string[last_end:start]
                if residual_text.strip():  # Skip empty or whitespace-only segments
                    textual_tokens.append(residual_text)

            visual_token_rows.append(match.group())
            last_end = end

        # Handle the remaining text after the last visual token match
        if last_end < len(input_string):
            residual_text = input_string[last_end:]
            if residual_text.strip():
                textual_tokens.append(residual_text)
        
        if not keep_text_as_is:
            textual_tokens = [self.remove_angle_brackets(t) for t in textual_tokens]
            textual_tokens = [t for t in textual_tokens if len(t)>5]

        return visual_token_rows, textual_tokens

    def get_tensor_from_visual_token_rows(self, visual_token_rows, H=64, W=64, fit_type='global'):
        eol_token = self.tokenizer.eol_token
        if isinstance(visual_token_rows, str):
            visual_token_rows = re.split(re.escape(eol_token), visual_token_rows)
        if len(visual_token_rows) < H - 2: 
            fit_type = 'global'
            warnings.warn(f'WARNING: Extremely few visual token rows: {len(visual_token_rows)}, forcing `fit_type=="global"`')
        
        # Return tensor shape: [num_imgs, H, W]
        if fit_type == 'global':
            return self.get_tensor_from_visual_token_rows_globally(visual_token_rows, H, W)
        elif fit_type == 'local':
            return self.get_tensor_from_visual_token_rows_locally(visual_token_rows, H, W)
        else:
            image_tensor = self.get_tensor_from_visual_token_rows_globally(visual_token_rows, H, W)
            image_tensor2 = self.get_tensor_from_visual_token_rows_locally(visual_token_rows, H, W)
            if (image_tensor == image_tensor2).all():
                return image_tensor
            return torch.cat([image_tensor, image_tensor2], dim=0)


    def get_tensor_from_visual_token_rows_globally(self, visual_token_rows, H=64, W=64):
        image = []
        visual_template = self.visual_template[1]  #r"<\|visual token (\d+)\|>"
        eol_token = self.tokenizer.eol_token
        if isinstance(visual_token_rows, str):
            visual_token_rows = re.split(re.escape(eol_token), visual_token_rows)

        for r in visual_token_rows:
            token_ids = re.findall(visual_template, r)
            if len(token_ids) > 0:
                row_token = [int(ii) for ii in token_ids]
                image.extend(row_token)
        num_curr = len(image)
        num_total = H*W
        image = image[:num_total] if num_curr >= H*W else (image + [image[-1]] * (num_total - num_curr))
        image = torch.tensor(image, dtype=torch.long)
        return image.reshape(H, W).detach().unsqueeze(0)

    def get_tensor_from_visual_token_rows_locally(self, visual_token_rows, H=64, W=64):
        image = []
        visual_template = self.visual_template[1] #r"<\|visual token (\d+)\|>"
        eol_token = self.tokenizer.eol_token
        if isinstance(visual_token_rows, str):  # split into list
            visual_token_rows = re.split(re.escape(eol_token), visual_token_rows)

        for r in visual_token_rows:
            token_ids = re.findall(visual_template, r)
            if len(token_ids) > 0:
                row_token = [int(ii) for ii in token_ids]
                num_cols = len(row_token)
                row_token = row_token[:W] if num_cols >= W else (row_token + [row_token[-1]] * (W - num_cols))
                image.append(row_token)

        num_rows = len(image)
        image = image[:H] if num_rows >= H else (image + [image[-1]] * (H - num_rows))
        return torch.tensor(image, dtype=torch.long).detach().unsqueeze(0)
