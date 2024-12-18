# coding=utf-8
# Copyright 2021 Google AI, Ross Wightman, The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch ViT model."""

import collections.abc
import math
import pickle
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    ImageClassifierOutput,
    MaskedImageModelingOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
    torch_int,
)
from .configuration_vit import ViTConfig

import numpy as np
import torch.nn.functional as F
import csv


def calculate_percentage_ones(G):
    """
    Calculate the percentage of elements in G that are 1 (True).

    Parameters:
        G (torch.Tensor): A boolean tensor.

    Returns:
        float: The percentage of elements that are 1 (True).
    """
    if not G.dtype == torch.bool:
        raise ValueError("Input tensor G must be of type torch.bool.")

    # Total number of elements in G
    total_elements = G.numel()

    # Count the number of 1s (True)
    count_ones = G.sum().item()  # G.sum() calculates the number of True values

    # Calculate percentage
    percentage_ones = (count_ones / total_elements) * 100

    # Print the result
    # print(f"Percentage of 1s (True) in G: {percentage_ones:.2f}%")

    return percentage_ones


def masked_softmax(attention_scores, G, dim=-1):
    """
    Compute softmax with a mask (G).

    Parameters:
        attention_scores (torch.Tensor): Raw attention scores (N x N).
        G (torch.Tensor): Mask matrix (N x N), where G[i, j] = 1 if token j contributes to token i, else 0.
        dim (int): Dimension to apply softmax on (default: -1).

    Returns:
        torch.Tensor: Masked attention matrix (N x N).
    """
    # Ensure G is converted to bool type
    # mask = G.bool()  # Convert G to boolean

    # Mask the attention scores: Set masked elements to a very large negative value
    # print('before mask')
    masked_scores = attention_scores.masked_fill(~G, torch.tensor(float('-inf')))
    # print('after mask')

    # Apply softmax to the masked scores
    attention_probs = F.softmax(masked_scores, dim=dim)

    return attention_probs


def update_G_matrix(G, keep_token_mask):
    """
    Update the G matrix by masking out tokens not in keep_token_mask.

    Parameters:
        G (torch.Tensor): Current G matrix (N x N).
        keep_token_mask (torch.Tensor): Binary mask indicating which tokens to keep (B x N).

    Returns:
        torch.Tensor: Updated G matrix.
    """
    # Ensure `keep_token_mask` has the correct dimensions (B x N x N)
    # keep_token_mask_expanded = keep_token_mask_expanded * keep_token_mask_expanded
    # keep_token_mask_expanded = keep_token_mask.unsqueeze(1) * keep_token_mask.unsqueeze(-1)
    # keep_token_mask_expanded = keep_token_mask_expanded.bool()
    # print('keep_token_mask', keep_token_mask)
    # 去掉第一个维度（1）以便逻辑或操作
    mask_row = keep_token_mask.squeeze(0).unsqueeze(1)  # Shape: (N, 1)
    mask_col = keep_token_mask.squeeze(0).unsqueeze(0)  # Shape: (1, N)

    # print('mask_row', mask_row)
    # print('mask_col', mask_col)

    # 逻辑或操作生成 NxN 矩阵
    keep_token_mask_expanded = mask_row | mask_col
    # print('keep_token_mask_expanded', keep_token_mask_expanded)
    # 将 NxN 矩阵重复 B 次，得到 BxNxN
    keep_token_mask_expanded = keep_token_mask_expanded.unsqueeze(0).repeat(16, 1, 1)

    keep_token_mask_expanded = keep_token_mask_expanded.bool()

    # Update G: Set elements to 0 where keep_token_mask is 0
    # print('before update')
    # print('Gtype=',G.type())
    G_updated = G & keep_token_mask_expanded
    # print('Gupdatetype=', G_updated.type())
    # print('after update')

    return G_updated


def tensor_float_to_fixed_torch(tensor, N, R):
    """
    Convert a PyTorch tensor to fixed-point representation.

    Args:
    - tensor: A PyTorch tensor of floating-point values.
    - N: Total number of bits in the fixed-point representation (including sign bit).
    - R: Number of fractional bits.

    Returns:
    - A PyTorch tensor with fixed-point integer mantissa values.
    """
    scaled_tensor = tensor * (2 ** R)
    fixed_tensor = torch.clamp(scaled_tensor.round(), -2 ** (N - 1), 2 ** (N - 1) - 1)
    return fixed_tensor.int()


def tensor_fixed_to_float_torch(tensor, R):
    """
    Convert a PyTorch fixed-point tensor back to floating-point representation.

    Args:
    - tensor: A PyTorch tensor of fixed-point integer mantissa values.
    - R: Number of fractional bits.

    Returns:
    - A PyTorch tensor of floating-point values.
    """
    # Divide by the scaling factor to reverse the fixed-point quantization
    float_tensor = tensor.float() / (2 ** R)
    return float_tensor


def adaptive_token_pruning(attn_prob, h, N, ratio):
    # 1. Initialize token scores

    # 2. Sum up the attention probabilities for each token
    token_score = torch.sum(attn_prob, dim=1, keepdim=False)
    token_score = token_score[0, 0, :]
    # print(token_score.size())
    # print(token_score)
    # 3. Calculate the threshold for pruning
    threshold = (h - token_score[0]) * ratio
    # print('threshold = ', threshold)
    # print(threshold)
    # 4. Sort token scores (excluding the first token [CLS] typically)
    sorted_token_score, sorted_token_idx = torch.sort(token_score[1:], descending=True)
    # print(sorted_token_idx)
    # 5. Determine how many tokens to keep
    part_sum_score = 0
    keep_num = 0
    while part_sum_score < threshold and keep_num < sorted_token_score.size(0):
        part_sum_score += sorted_token_score[keep_num]
        keep_num += 1

    # 6. Determine indices of tokens to retain
    # print('keep_num', keep_num)
    remained_idx = sorted_token_idx[:keep_num]  # Adjust index as we excluded the first token in sorting
    remained_idx = remained_idx + 1
    # print(remained_idx)
    remained_idx = torch.cat((torch.tensor([0], device=attn_prob.device), remained_idx))  # Include the first token
    # print('remained_idx size')
    # print(remained_idx.size())
    # print(remained_idx.size(0))

    return remained_idx


def create_keep_decision_mask(remained_idx, N, B):
    # 初始化掩码，形状为 [B, N]
    mask = torch.zeros(B, N, dtype=torch.long)

    # 遍历每个批次
    for b in range(B):
        # 设置该批次中应保留的 tokens
        for idx in range(remained_idx.size(0)):
            mask[b, remained_idx[idx]] = 1

    # 确保第一个 token 总是被保留
    mask[:, 0] = 1

    return mask


layer_count = 0
G = 0
calculation_amount = 0
N = 577
D = 64
imag_idx = 1
alpha_values = []

logger = logging.get_logger(__name__)

# General docstring
_CONFIG_FOR_DOC = "ViTConfig"

# Base docstring
_CHECKPOINT_FOR_DOC = "google/vit-base-patch16-224-in21k"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# Image classification docstring
_IMAGE_CLASS_CHECKPOINT = "google/vit-base-patch16-224"
_IMAGE_CLASS_EXPECTED_OUTPUT = "Egyptian cat"


class ViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings. Optionally, also the mask token.
    """

    def __init__(self, config: ViTConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        self.patch_embeddings = ViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.patch_size = config.patch_size
        self.config = config

    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """
        This method allows to interpolate the pre-trained position encodings, to be able to use the model on higher resolution
        images. This method is also adapted to support torch.jit tracing.

        Adapted from:
        - https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174-L194, and
        - https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/models/vision_transformer.py#L179-L211
        """

        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1

        # always interpolate when tracing to ensure the exported model works for dynamic input shapes
        if not torch.jit.is_tracing() and num_patches == num_positions and height == width:
            return self.position_embeddings

        class_pos_embed = self.position_embeddings[:, :1]
        patch_pos_embed = self.position_embeddings[:, 1:]

        dim = embeddings.shape[-1]

        new_height = height // self.patch_size
        new_width = width // self.patch_size

        sqrt_num_positions = torch_int(num_positions ** 0.5)
        patch_pos_embed = patch_pos_embed.reshape(1, sqrt_num_positions, sqrt_num_positions, dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(new_height, new_width),
            mode="bicubic",
            align_corners=False,
        )

        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward(
            self,
            pixel_values: torch.Tensor,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        if bool_masked_pos is not None:
            seq_length = embeddings.shape[1]
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # replace the masked visual tokens by mask_tokens
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # add the [CLS] token to the embedded patch tokens
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # add positional encoding to each token
        if interpolate_pos_encoding:
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            embeddings = embeddings + self.position_embeddings

        embeddings = self.dropout(embeddings)

        return embeddings


class ViTPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config):
        super().__init__()
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
                f" Expected {self.num_channels} but got {num_channels}."
            )
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        return embeddings


class ViTSelfAttention(nn.Module):
    runs_dict = {}

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # self.alpha_values = [] # 用于存储每一层的alpha值

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        query_quantized = nn.Linear(1024, self.all_head_size, bias=True)
        quantized_weight = tensor_float_to_fixed_torch(self.query.weight, 16, 8).to(torch.int32)
        quantized_bias = tensor_float_to_fixed_torch(self.query.bias, 32, 16).to(torch.int32)

        with torch.no_grad():  # 禁用梯度计算
            query_quantized.weight.copy_(quantized_weight)
            if query_quantized.bias is not None:  # 如果偏置存在
                query_quantized.bias.copy_(quantized_bias)

        key_quantized = nn.Linear(1024, self.all_head_size, bias=True)
        quantized_weight = tensor_float_to_fixed_torch(self.key.weight, 16, 8).to(torch.int32)
        quantized_bias = tensor_float_to_fixed_torch(self.key.bias, 32, 16).to(torch.int32)

        with torch.no_grad():  # 禁用梯度计算
            key_quantized.weight.copy_(quantized_weight)
            if key_quantized.bias is not None:  # 如果偏置存在
                key_quantized.bias.copy_(quantized_bias)

        value_quantized = nn.Linear(1024, self.all_head_size, bias=True)
        quantized_weight = tensor_float_to_fixed_torch(self.value.weight, 16, 8).to(torch.int32)
        quantized_bias = tensor_float_to_fixed_torch(self.value.bias, 32, 16).to(torch.int32)

        with torch.no_grad():  # 禁用梯度计算
            value_quantized.weight.copy_(quantized_weight)
            if value_quantized.bias is not None:  # 如果偏置存在
                value_quantized.bias.copy_(quantized_bias)
        # print(hidden_states.size())

        global layer_count, G, calculation_amount, N, D, imag_idx, alpha_values

        if (layer_count == 4 or layer_count == 8 or layer_count == 12 or layer_count == 16 or layer_count == 20):
            ratio = 0.7
        else:
            ratio = 1

        # hidden_states = tensor_float_to_fixed_torch(hidden_states, 16, 8)
        # # self.query.weight = tensor_float_to_fixed_torch(query.weight, 16, 8)
        # # self.query.bias = tensor_float_to_fixed_torch(query.bias, 16, 16)
        # self.key.weight = tensor_float_to_fixed_torch(key.weight, 16, 8)
        # self.key.bias = tensor_float_to_fixed_torch(key.bias, 16, 16)
        # self.value.weight = tensor_float_to_fixed_torch(value.weight, 16, 8)
        # self.value.bias = tensor_float_to_fixed_torch(value.bias, 16, 16)

        # mixed_query_layer = self.query(hidden_states)
        hidden_states_quantized = tensor_float_to_fixed_torch(hidden_states, 16, 8).to(torch.float32)
        mixed_query_layer_quantized = query_quantized(hidden_states_quantized)  #

        # key_layer = self.transpose_for_scores(self.key(hidden_states))
        # value_layer = self.transpose_for_scores(self.value(hidden_states))
        # query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(key_quantized(hidden_states_quantized))
        value_layer = self.transpose_for_scores(value_quantized(hidden_states_quantized))
        query_layer = self.transpose_for_scores(mixed_query_layer_quantized)

        key_layer = tensor_fixed_to_float_torch(key_layer, 16)
        value_layer = tensor_fixed_to_float_torch(value_layer, 16)
        query_layer = tensor_fixed_to_float_torch(query_layer, 16)

        # print('In Layer: ',layer_count,', N and D are: ',key_layer.size())

        # key_layer = tensor_fixed_to_float_torch(key_layer, 16)
        # value_layer = tensor_fixed_to_float_torch(value_layer,  16)
        # query_layer = tensor_fixed_to_float_torch(query_layer,  16)

        # key_layer = tensor_float_to_fixed_torch(key_layer, 16, 8)
        # value_layer = tensor_float_to_fixed_torch(value_layer, 16, 8)
        # query_layer = tensor_float_to_fixed_torch(query_layer, 16, 8)
        key_layer = tensor_float_to_fixed_torch(key_layer, 16, 8)
        value_layer = tensor_float_to_fixed_torch(value_layer, 16, 8)
        query_layer = tensor_float_to_fixed_torch(query_layer, 16, 8)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # print(attention_scores.size())
        # run_dict['attention_scores1'] = attention_scores.clone()
        # print('layer_count', layer_count)
        if (layer_count == 0):
            G = torch.ones(attention_scores.size(), dtype=torch.bool)
            # G = torch.tensor(G, dtype=torch.bool) if not isinstance(G, torch.Tensor) else G
        # print('G',G)
        calculate_percentage_ones(G)
        attention_scores = tensor_fixed_to_float_torch(attention_scores, 16)

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # run_dict['attention_scores2'] = attention_scores.clone()

        # Normalize the attention scores to probabilities.
        # attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = masked_softmax(attention_scores, G)
        # print('attention_probs', attention_probs)

        # run_dict['attention_probs1'] = attention_probs.clone()

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # run_dict['attention_probs2'] = attention_probs.clone()

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        remained_idx = adaptive_token_pruning(attention_probs, h=16, N=577, ratio=ratio)

        keep_token_mask = create_keep_decision_mask(remained_idx, N=577, B=1)

        # print('keep token mask', keep_token_mask.size())
        # print(keep_token_mask)

        if (ratio != 1):
            print(f'In Layer: {layer_count}, alpha^2 = {calculate_percentage_ones(G):.3f}%')

        alpha = (calculate_percentage_ones(G))**0.5
        alpha_values.append(alpha * 10)
        calculation_amount = calculation_amount + (2 * alpha * D + calculate_percentage_ones(G) * N)/ ((2 * D + N) * 24)
        # # 收集alpha值
        # alphas = []
        # calculation_amount = 0
        # for attention_module in attention_modules:
        #     alphas.extend(attention_module.alpha_values)
        #     # 如果calculation_amount是每层的，需要累加
        #     calculation_amount += attention_module.calculation_amount  # 如果适用
        #
        # # 准备要写入CSV的数据
        # row = [image_file] + alphas + [calculation_amount, correct]
        # writer.writerow(row)

        G = update_G_matrix(G, keep_token_mask)

        attention_probs = attention_probs * keep_token_mask.unsqueeze(1).unsqueeze(-1)

        attention_probs = tensor_float_to_fixed_torch(attention_probs, 16, 8)
        context_layer = torch.matmul(attention_probs, value_layer)
        # run_dict['context_layer1'] = context_layer.clone()

        context_layer = tensor_fixed_to_float_torch(context_layer, 16)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # run_dict['context_layer2'] = context_layer.clone()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(new_context_layer_shape)
        # run_dict['context_layer3'] = context_layer.clone()

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # print("cnt =  " + str(len(self.runs_dict)))
        # run_dict = {}
        # run_dict['self_query'] = self.query
        # run_dict['self_key'] = self.key
        # run_dict['self_value'] = self.value

        # run_dict['hidden_states'] = hidden_states.clone()
        # run_dict['mixed_query_layer'] = mixed_query_layer.clone()

        # run_dict['key_layer'] = key_layer.clone()
        # run_dict['value_layer'] = value_layer.clone()
        # run_dict['query_layer'] = query_layer.clone()

        # run_dict['outputs'] = outputs[0].clone()

        # run_key = f"run_{len(self.runs_dict)}"
        # self.runs_dict[run_key] = run_dict

        # torch.save(self.runs_dict, r'D:\desktop\598\598_final_project\runs_dict.pt')
        csv_filename = "test_results.csv"
        csvfile = open(csv_filename, 'a', newline='', encoding='utf-8')
        writer = csv.writer(csvfile)

        if (layer_count == 23):
            # print('last layer G percentage = ',calculate_percentage_ones(G))
            print(f'Last layer G percentage = {calculate_percentage_ones(G):.3f}%')
            print(f'calculation_amount = {calculation_amount:.3f}%')

            row = [imag_idx] + [f"{value:.3f}%" for value in alpha_values] + [f"{calculation_amount:.3f}%"]
            writer.writerow(row)
            imag_idx = imag_idx + 1
            alpha_values = []
            calculation_amount = 0

        layer_count = (layer_count + 1) % 24
        csvfile.close()

        return outputs


class ViTSdpaSelfAttention(ViTSelfAttention):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob

    def forward(
            self,
            hidden_states: torch.FloatTensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        if output_attentions or head_mask is not None:
            logger.warning_once(
                "`ViTSdpaAttention` is used but `torch.nn.functional.scaled_dot_product_attention` does not support "
                "`output_attentions=True` or `head_mask`. Falling back to the manual attention implementation, but "
                "specifying the manual implementation will be required from Transformers version v5.0.0 onwards. "
                'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                head_mask=head_mask,
                output_attentions=output_attentions,
            )

        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        context_layer = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            head_mask,
            self.attention_probs_dropout_prob if self.training else 0.0,
            is_causal=False,
            scale=None,
        )

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        return context_layer, None


class ViTSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViTAttention(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.attention = ViTSelfAttention(config)
        self.output = ViTSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


class ViTSdpaAttention(ViTAttention):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)
        self.attention = ViTSdpaSelfAttention(config)


class ViTIntermediate(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states


class ViTOutput(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


VIT_ATTENTION_CLASSES = {
    "eager": ViTAttention,
    "sdpa": ViTSdpaAttention,
}


class ViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = VIT_ATTENTION_CLASSES["eager"](config)
        self.intermediate = ViTIntermediate(config)
        self.output = ViTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # in ViT, layernorm is applied before self-attention
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # first residual connection
        hidden_states = attention_output + hidden_states

        # in ViT, layernorm is also applied after self-attention
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # second residual connection is done here
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


class ViTEncoder(nn.Module):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ViTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: bool = False,
            output_hidden_states: bool = False,
            return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


class ViTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = ViTConfig
    base_model_prefix = "vit"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ViTEmbeddings", "ViTLayer"]
    _supports_sdpa = True

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # Upcast the input in `fp32` and cast it back to desired `dtype` to avoid
            # `trunc_normal_cpu` not implemented in `half` issues
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, ViTEmbeddings):
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype)

            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype)


VIT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See [`ViTImageProcessor.__call__`]
            for details.

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        interpolate_pos_encoding (`bool`, *optional*):
            Whether to interpolate the pre-trained position encodings.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare ViT Model transformer outputting raw hidden-states without any specific head on top.",
    VIT_START_DOCSTRING,
)
class ViTModel(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        self.embeddings = ViTEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTEncoder(config)

        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ViTPatchEmbeddings:
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # TODO: maybe have a cleaner way to cast the input (from `ImageProcessor` side?)
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class ViTPooler(nn.Module):
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


@add_start_docstrings(
    """ViT Model with a decoder on top for masked image modeling, as proposed in [SimMIM](https://arxiv.org/abs/2111.09886).

    <Tip>

    Note that we provide a script to pre-train this model on custom data in our [examples
    directory](https://github.com/huggingface/transformers/tree/main/examples/pytorch/image-pretraining).

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ViTForMaskedImageModeling(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.vit = ViTModel(config, add_pooling_layer=False, use_mask_token=True)

        self.decoder = nn.Sequential(
            nn.Conv2d(
                in_channels=config.hidden_size,
                out_channels=config.encoder_stride ** 2 * config.num_channels,
                kernel_size=1,
            ),
            nn.PixelShuffle(config.encoder_stride),
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MaskedImageModelingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            bool_masked_pos: Optional[torch.BoolTensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, MaskedImageModelingOutput]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).

        Returns:

        Examples:
        ```python
        >>> from transformers import AutoImageProcessor, ViTForMaskedImageModeling
        >>> import torch
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        >>> model = ViTForMaskedImageModeling.from_pretrained("google/vit-base-patch16-224-in21k")

        >>> num_patches = (model.config.image_size // model.config.patch_size) ** 2
        >>> pixel_values = image_processor(images=image, return_tensors="pt").pixel_values
        >>> # create random boolean mask of shape (batch_size, num_patches)
        >>> bool_masked_pos = torch.randint(low=0, high=2, size=(1, num_patches)).bool()

        >>> outputs = model(pixel_values, bool_masked_pos=bool_masked_pos)
        >>> loss, reconstructed_pixel_values = outputs.loss, outputs.reconstruction
        >>> list(reconstructed_pixel_values.shape)
        [1, 3, 224, 224]
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if bool_masked_pos is not None and (self.config.patch_size != self.config.encoder_stride):
            raise ValueError(
                "When `bool_masked_pos` is provided, `patch_size` must be equal to `encoder_stride` to ensure that "
                "the reconstructed image has the same dimensions as the input. "
                f"Got `patch_size` = {self.config.patch_size} and `encoder_stride` = {self.config.encoder_stride}."
            )

        outputs = self.vit(
            pixel_values,
            bool_masked_pos=bool_masked_pos,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Reshape to (batch_size, num_channels, height, width)
        sequence_output = sequence_output[:, 1:]
        batch_size, sequence_length, num_channels = sequence_output.shape
        height = width = math.floor(sequence_length ** 0.5)
        sequence_output = sequence_output.permute(0, 2, 1).reshape(batch_size, num_channels, height, width)

        # Reconstruct pixel values
        reconstructed_pixel_values = self.decoder(sequence_output)

        masked_im_loss = None
        if bool_masked_pos is not None:
            size = self.config.image_size // self.config.patch_size
            bool_masked_pos = bool_masked_pos.reshape(-1, size, size)
            mask = (
                bool_masked_pos.repeat_interleave(self.config.patch_size, 1)
                .repeat_interleave(self.config.patch_size, 2)
                .unsqueeze(1)
                .contiguous()
            )
            reconstruction_loss = nn.functional.l1_loss(pixel_values, reconstructed_pixel_values, reduction="none")
            masked_im_loss = (reconstruction_loss * mask).sum() / (mask.sum() + 1e-5) / self.config.num_channels

        if not return_dict:
            output = (reconstructed_pixel_values,) + outputs[1:]
            return ((masked_im_loss,) + output) if masked_im_loss is not None else output

        return MaskedImageModelingOutput(
            loss=masked_im_loss,
            reconstruction=reconstructed_pixel_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


@add_start_docstrings(
    """
    ViT Model transformer with an image classification head on top (a linear layer on top of the final hidden state of
    the [CLS] token) e.g. for ImageNet.

    <Tip>

        Note that it's possible to fine-tune ViT on higher resolution images than the ones it has been trained on, by
        setting `interpolate_pos_encoding` to `True` in the forward of the model. This will interpolate the pre-trained
        position embeddings to the higher resolution.

    </Tip>
    """,
    VIT_START_DOCSTRING,
)
class ViTForImageClassification(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vit = ViTModel(config, add_pooling_layer=False)

        # Classifier head
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_IMAGE_CLASS_CHECKPOINT,
        output_type=ImageClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_IMAGE_CLASS_EXPECTED_OUTPUT,
    )
    def forward(
            self,
            pixel_values: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            interpolate_pos_encoding: Optional[bool] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
