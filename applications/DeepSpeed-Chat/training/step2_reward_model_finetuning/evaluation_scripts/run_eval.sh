#!/bin/bash
# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

# Add the path to the finetuned model
python  rw_eval.py \
    --model_name_or_path /data/yusun/karan/DeepSpeedExamples/applications/DeepSpeed-Chat/training/step2_reward_model_finetuning/output/
