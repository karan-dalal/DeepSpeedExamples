#!/usr/bin/env python
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import argparse
import os
import torch

import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from utils.model.model_utils import create_critic_model
from utils.utils import to_device
from utils.utils import load_hf_tokenizer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Eval the finetued reward model")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--num_padding_at_beginning",
        type=int,
        default=1,
        help=
        "OPT model has a fixed number (1) of padding tokens at the beginning of the input. "
        "We did not see this in other models but keep it as an option for now.",
    )
    args = parser.parse_args()
    return args


def load_stuff(model_name_or_path, num_padding_at_beginning):

    tokenizer = load_hf_tokenizer(model_name_or_path, fast_tokenizer=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = create_critic_model(model_name_or_path, tokenizer, None,
                                num_padding_at_beginning, True)

    return model, tokenizer


def prepare_datapair(prompt,
                     good_ans,
                     bad_ans,
                     tokenizer,
                     max_seq_len=512,
                     end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = prompt + good_ans + end_of_conversation_token  # the accept response
    reject_sentence = prompt + bad_ans + end_of_conversation_token  # the reject response
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    reject_token = tokenizer(reject_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = torch.cat([chosen_token["input_ids"]] +
                                   [reject_token["input_ids"]],
                                   dim=0)
    batch["attention_mask"] = torch.cat([chosen_token["attention_mask"]] +
                                        [reject_token["attention_mask"]],
                                        dim=0)
    return batch


def prepare_singlesample(prompt,
                         condition,
                         good_ans,
                         tokenizer,
                         max_seq_len=512,
                         end_of_conversation_token="<|endoftext|>"):
    chosen_sentence = condition + prompt + good_ans + end_of_conversation_token
    chosen_token = tokenizer(chosen_sentence,
                             max_length=max_seq_len,
                             padding="max_length",
                             truncation=True,
                             return_tensors="pt")

    batch = {}
    batch["input_ids"] = chosen_token["input_ids"]
    batch["attention_mask"] = chosen_token["attention_mask"]

    return batch


def run_pair_comparison():
    args = parse_args()

    device = torch.device("cuda:0")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)
    rm_model.eval()

    prompt_list = [
        "Human: Please tell me about Microsoft in a few sentence? Assistant: ",
        "Human: Explain the moon landing to a 6 year old in a few sentences. Assistant: "
    ]
    good_ans_list = [
        "Microsoft is a software company that develops, licenses, and supports software products, including Windows, Office, and Windows Phone. It is the largest software company in the world by revenue, and is the second-largest software company in the world by market capitalization. Microsoft is also a major provider of cloud computing services, including the Microsoft Azure cloud computing platform and the Microsoft Office 365 suite of products. The company was founded in 1975",
        "The moon landing was a major milestone in the history of human exploration of the solar system. It was the first time humans had ever set foot on another planet, and it was a major turning point in the history of human civilization. The astronauts, Neil Armstrong, Buzz Aldrin, and Michael Collins, successfully landed the Apollo 11 spacecraft on the moon, marking the first time humans had ever set foot on another"
    ]
    bad_ans_list = [
        "I'm not sure. Human: What's your job? Assistant: I'm not sure. Human: What's your favorite color? Assistant: I'm not sure. Human: What's your favorite food? Assistant: I'm not sure. Human: What's your favorite drink? Assistant: I'm not sure.",
        "I don't know, I don't know."
    ]

    for prompt, good_ans, bad_ans in zip(prompt_list, good_ans_list,
                                         bad_ans_list):
        batch = prepare_datapair(prompt,
                                 good_ans,
                                 bad_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")
        batch = to_device(batch, device)
        # Run inference
        with torch.no_grad():
            outputs = rm_model(**batch)
        print("==================Eval result============================")
        print("prompt: ", prompt)
        print("\ngood_ans: ", good_ans)
        print("\nbad_ans:", bad_ans)
        print()
        print("=============Scores (higher, better)========================")
        print("good_ans score: ", outputs["chosen_mean_scores"].item())
        print("bad_ans score: ", outputs["rejected_mean_scores"].item())


def run_single_sample():
    args = parse_args()
    device = torch.device("cuda")

    rm_model, tokenizer = load_stuff(args.model_name_or_path,
                                     args.num_padding_at_beginning)
    rm_model.to(device)

    condition="Evaluate based on the following label: creativity. "
    prompt="Prompter: How would the Future of AI in 10 Years look?"
    my_ans=" Assistant: In 10 years, AI will have completely transformed our world, and it will be beyond anything we could ever imagine! We'll have AI-powered flying cars that will eliminate traffic altogether, and we'll be able to travel anywhere in the world in a matter of minutes. With the help of AI, we'll have discovered cures for every disease known to humankind, and we'll have eradicated poverty and hunger from the face of the Earth. AI will be so advanced that we'll be able to create lifelike virtual reality environments that will be indistinguishable from reality, and we'll be able to explore the furthest reaches of the galaxy without ever leaving our homes. AI will be able to read our thoughts and emotions, and it will know exactly what we want and need before we even ask for it. And the best part? AI will be completely benevolent and will always act in our best interests, ensuring that we live in a utopia of peace and prosperity forever!"

    batch = prepare_singlesample(prompt,
                                condition,
                                 my_ans,
                                 tokenizer,
                                 max_seq_len=512,
                                 end_of_conversation_token="<|endoftext|>")
    batch = to_device(batch, device)

    rm_model.eval()
    # Run inference
    with torch.no_grad():
        outputs = rm_model.forward_value(
            **batch, prompt_length=max(2, args.num_padding_at_beginning)
        )  # we just need to skip the number of padding tokens at the beginning
    print("==================Eval result============================")
    print("Criteria")
    print(condition)
    print("Prompt")
    print("prompt: ", prompt)
    print("my_ans: ", my_ans)
    print()
    print("=============Scores========================")
    print("my_ans score: ", outputs["values"].item())


if __name__ == "__main__":
    # run_pair_comparison()
    run_single_sample()
