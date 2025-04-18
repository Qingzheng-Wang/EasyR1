# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from typing import Dict

from mathruler.grader import extract_boxed_content, grade_answer

def step_reward(predict_str: str) -> float:
    # Identify if there are Step1, Step2, and Step3 in the <think> ... </think>.
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_match = re.search(think_pattern, predict_str)
    
    if not think_match:
        return 0.0

    think_content = think_match.group(1)
    steps = ["Step1", "Step2", "Step3"]
    step_count = sum(1 for step in steps if step in think_content)
    
    return step_count / len(steps)

def format_reward(predict_str: str) -> float:
    # Check if the format is <think> ... </think> ... <answer> ... </answer>.
    pattern = re.compile(r"<think>.*</think>.*<answer>.*</answer>.*", re.DOTALL)
    format_match = re.fullmatch(pattern, predict_str)
    return 1.0 if format_match else 0.0


def accuracy_reward(predict_str: str, ground_truth: str) -> float:
    answer_pattern = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
    
    answer_match = re.search(answer_pattern, predict_str)
    answer = answer_match.group(1).strip() if answer_match else ""
    
    gt_match = re.search(answer_pattern, ground_truth)
    gt_answer = gt_match.group(1).strip() if gt_match else ""

    return 1.0 if answer == gt_answer else 0.0


def compute_score(predict_str: str, ground_truth: str, format_weight: float = 0.1) -> Dict[str, float]:
    predict_str = re.sub(r"\s*(<|>|/)\s*", r"\1", predict_str)  # handle qwen2.5vl-32b format
    format_score = format_reward(predict_str)
    accuracy_score = accuracy_reward(predict_str, ground_truth)
    return {
        "overall": (1 - format_weight) * accuracy_score + format_weight * format_score,
        "format": format_score,
        "accuracy": accuracy_score,
    }
