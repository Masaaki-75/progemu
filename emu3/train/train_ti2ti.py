# -*- coding: utf-8 -*-

import os
import sys
import pathlib
from datetime import datetime
from typing import Optional, List
from dataclasses import dataclass, field

import transformers as tf
import torch

from emu3.mllm import Emu3Config, Emu3Tokenizer
from emu3.mllm.modeling_ti2ti import Emu3ForCausalLM
from emu3.train.datasets_ti2ti import ProgEmuDataset



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/Emu3-Gen")
    tokenizer_path: Optional[str] = field(default="BAAI/Emu3-Gen")


@dataclass
class DataArguments:
    data_path: Optional[str] = field(default=None)
    path_prefix: Optional[str] = field(default=None)
    visual_token_folder: Optional[str] = field(default='emu3_token_ids')
    null_prompt_prob: float = field(default=0.05)  # Probability for null prompt augmentation (unlocks CFG).
    ignore_index: int = field(default=-100)  # For masking padding tokens.
    visual_token_pattern: str = field(default="<|visual token {token_id:0>6d}|>")  
    codebook_size: Optional[int] = field(default=32768)
    apply_loss_on_only_vision: bool = field(default=False)
    apply_loss_on_only_text: bool = field(default=False)
    shuffle_sentence_prob: float = field(default=0.05)  # Shuffle sentences for data augmentation.
    dropneg_sentence_prob: float = field(default=0.05)  # Drop normal findings for data augmentation.
    use_template: Optional[bool] = field(default=False)  # Whether to use template for input prompt.


@dataclass
class TrainingArguments(tf.TrainingArguments):
    output_dir: str = field(default="outputs")
    report_to: List[str] = field(default_factory=list)  # 存储记录的位置, tensorboard或者wandb
    remove_unused_columns: bool = field(default=False)
    min_learning_rate: Optional[float] = field(default=None)
    attn_type: Optional[str] = field(default="fa2")
    image_area: Optional[int] = field(default=None)
    max_position_embeddings: Optional[int] = field(default=None)


def update_configs(model_config, args, fields):
    cross_update = lambda a, b, field_name: (
        setattr(b, field_name, getattr(a, field_name))
        if getattr(b, field_name, None) is None else
        setattr(a, field_name, getattr(b, field_name))
    )

    for f in fields:
        cross_update(model_config, args, f)

def train():
    parser = tf.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_config = Emu3Config.from_pretrained(model_args.model_name_or_path)
    update_configs(model_config, training_args, ["image_area", "max_position_embeddings"])
    if training_args.min_learning_rate is not None:
        training_args.lr_scheduler_kwargs["min_lr"] = training_args.min_learning_rate
    os.environ["WANDB_DIR"] = os.path.join(training_args.output_dir, "wandb")

    model = Emu3ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        #device_map="balanced_low_0",
        config=model_config,
        attn_implementation="flash_attention_2" if training_args.attn_type == "fa2" else training_args.attn_type,
        torch_dtype=torch.bfloat16 if training_args.bf16 else None)

    tokenizer = Emu3Tokenizer.from_pretrained(
        model_args.tokenizer_path,
        model_max_length=training_args.max_position_embeddings,
        padding_side="right",
        use_fast=False)

    print(f'Trained with ProgEmuDataset (use_template={data_args.use_template}).')
    train_dataset = ProgEmuDataset(data_args, tokenizer=tokenizer)
    
    trainer = tf.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        print(f'{datetime.now()}. Trying to resume training....')
        trainer.train(resume_from_checkpoint=True)
    else:
        print(f'{datetime.now()}. Trying to start new training...')
        trainer.train()
        
    print(f'{datetime.now()}. Finish training!')
    trainer.save_state()

    torch.cuda.synchronize()
    trainer.save_model(training_args.output_dir)


if __name__ == "__main__":
    """
    See scripts/t2i_sft.sh for a good launching example.
    """
    train()
