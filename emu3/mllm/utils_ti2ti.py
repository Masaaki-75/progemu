# coding=utf-8
# Adapted from https://github.com/baaivision/Emu3/blob/main/emu3/mllm/utils_emu3.py

import torch

class Emu3PrefixConstrainedLogitsHelper:

    def __init__(
        self,
        height,
        width,
        img_token,
        eoi_token,
        eos_token,
        eol_token,
        eof_token,
        pad_token,
        visual_tokens,
        textual_tokens,
        only_output_vision=False,
    ):
        self.height = height  # [batch_size,]
        self.width = width  # [batch_size,]
        self.img_token = img_token
        self.eoi_token = eoi_token
        self.eos_token = eos_token
        self.eol_token = eol_token
        self.eof_token = eof_token
        self.pad_token = pad_token
        self.visual_tokens = visual_tokens
        self.textual_tokens = textual_tokens #+ [self.pad_token, self.eos_token]
        self.only_output_vision = only_output_vision
        self.offset_cache = {}

    def __call__(self, batch_id, input_ids):
        if batch_id not in self.offset_cache:
            position = torch.nonzero(input_ids == self.img_token, as_tuple=True)[0][1]
            self.offset_cache[batch_id] = position

        height = self.height[batch_id] if self.height.shape[0] > 1 else self.height[0]
        width = self.width[batch_id] if self.width.shape[0] > 1 else self.width[0]

        offset = input_ids.shape[0] - self.offset_cache[batch_id]
        # height = height.to(offset.device)
        # width = width.to(offset.device)

        if offset % (width + 1) == 0:  # we are now at the end of a row
            return (self.eol_token, )
        elif offset == (width + 1) * height + 1:  # we are now at the end of an image
            return (self.eof_token, )
        elif offset == (width + 1) * height + 2:  # at the end of an image and an <eof> token
            return (self.eoi_token, )
        elif (offset >= (width + 1) * height + 3) and (offset <= (width + 1) * height + 4):  # <eof> and <eoi>
            if self.only_output_vision:
                return (self.eos_token, )
            else:
                return self.textual_tokens
        elif (offset >= (width + 1) * height + 5) and (offset <= (width + 1) * height + 200):
            if self.only_output_vision:
                return (self.eos_token, )
            else:
                return self.textual_tokens + [self.eos_token]
        elif offset > (width + 1) * height + 200:
            return (self.eos_token, )
        else:
            return self.visual_tokens
