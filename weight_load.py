import torch
from torch import nn
from typing import Dict, List, Optional, Set, Tuple, Union
import math
import matplotlib.pyplot as plt
from scipy.io import savemat


def plot_distribution(data, title, ax):
    ax.hist(data.detach().cpu().numpy().flatten(), bins=50, alpha=0.7, color='blue')  # 将张量展平并绘制直方图
    ax.set_title(title)
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')


class ViTSelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.num_attention_heads = 16
        self.attention_head_size = 64
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(1024, self.all_head_size, bias=True)
        self.key = nn.Linear(1024, self.all_head_size, bias=True)
        self.value = nn.Linear(1024, self.all_head_size, bias=True)

        self.dropout = nn.Dropout(0.0)

        self.runs_dict = {}

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:

        run_dict = {}
        run_dict['num_attention_heads'] = self.num_attention_heads
        run_dict['attention_head_size'] = self.attention_head_size
        run_dict['all_head_size'] = self.all_head_size

        run_dict['hidden_states'] = hidden_states.clone()

        mixed_query_layer = self.query(hidden_states)
        run_dict['mixed_query_layer'] = mixed_query_layer.clone()

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        run_dict['key_layer'] = key_layer.clone()
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        run_dict['value_layer'] = value_layer.clone()
        query_layer = self.transpose_for_scores(mixed_query_layer)
        run_dict['query_layer'] = query_layer.clone()

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        run_dict['attention_scores1'] = attention_scores.clone()

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        run_dict['attention_scores2'] = attention_scores.clone()

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        run_dict['attention_probs1'] = attention_probs.clone()

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        run_dict['attention_probs2'] = attention_probs.clone()

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        run_dict['context_layer1'] = context_layer.clone()

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        run_dict['context_layer2'] = context_layer.clone()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        context_layer = context_layer.view(new_context_layer_shape)
        run_dict['context_layer3'] = context_layer.clone()

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        run_key = f"run_{len(self.runs_dict) + 1}"
        self.runs_dict[run_key] = run_dict

        return outputs


layer = ViTSelfAttention()
bin_file_path = "pytorch_model.bin"  # 替换为你的 bin 文件路径
state_dict = torch.load(bin_file_path, weights_only=True)
# 加载权重文件
layer.query.weight.data = state_dict["vit.encoder.layer.0.attention.attention.query.weight"]
layer.query.bias.data = state_dict["vit.encoder.layer.0.attention.attention.query.bias"]
layer.key.weight.data = state_dict["vit.encoder.layer.0.attention.attention.key.weight"]
layer.key.bias.data = state_dict["vit.encoder.layer.0.attention.attention.key.bias"]
layer.value.weight.data = state_dict["vit.encoder.layer.0.attention.attention.value.weight"]
layer.value.bias.data = state_dict["vit.encoder.layer.0.attention.attention.value.bias"]

# 处理权重数据，为了保存成matlab可以读取的格式
query_weight = state_dict["vit.encoder.layer.0.attention.attention.query.weight"].detach().numpy()
query_bias = state_dict["vit.encoder.layer.0.attention.attention.query.bias"].detach().numpy()
key_weight = state_dict["vit.encoder.layer.0.attention.attention.key.weight"].detach().numpy()
key_bias = state_dict["vit.encoder.layer.0.attention.attention.key.bias"].detach().numpy()
value_weight = state_dict["vit.encoder.layer.0.attention.attention.value.weight"].detach().numpy()
value_bias = state_dict["vit.encoder.layer.0.attention.attention.value.bias"].detach().numpy()

weights_dict = {
    "query_weight": query_weight,
    "query_bias": query_bias,
    "key_weight": key_weight,
    "key_bias": key_bias,
    "value_weight": value_weight,
    "value_bias": value_bias
}

savemat('weights.mat', weights_dict)

# 加载运行时保存下来的数据
data_file = r"runs_dict.pt"  # 替换为实际路径
data = torch.load(data_file, weights_only=False)
hidden_states = data["run_0"]["hidden_states"].detach()
run_0_data = data["run_0"]

# 保存到matlab可以读取的数据类型
run_0_data_dict = {}
for key, value in run_0_data.items():
    if isinstance(value, torch.Tensor):  # If the value is a tensor
        run_0_data_dict[key] = value.detach().cpu().numpy()  # Detach, move to CPU, and convert to numpy array
    else:
        run_0_data_dict[key] = value  # Keep non-tensor values as they are

matlab_data = {"data": run_0_data_dict}
savemat("hidden_states.mat", matlab_data)


# 模拟selfattention的前向传播
mixed_query_layer = layer.query(hidden_states)
mixed_query_layer_mat = torch.matmul(hidden_states, layer.query.weight.T) + layer.query.bias

key_layer = layer.transpose_for_scores(layer.key(hidden_states))
value_layer = layer.transpose_for_scores(layer.value(hidden_states))
query_layer = layer.transpose_for_scores(mixed_query_layer)

attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

attention_scores = attention_scores / math.sqrt(layer.attention_head_size)

attention_probs = nn.functional.softmax(attention_scores, dim=-1)

context_layer = torch.matmul(attention_probs, value_layer)

correct_output = data["run_0"]["context_layer1"].detach()

print(context_layer[0][0])
print(correct_output)


# 绘制权重分布
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

weight_params = layer.query.weight.numel()  # 权重的元素总数
bias_params = layer.query.bias.numel()      # 偏置的元素总数

query_weight_max = layer.query.weight.max().item()  # 最大值
query_weight_min = layer.query.weight.min().item()  # 最小值
key_weight_max = layer.key.weight.max().item()  # 最大值
key_weight_min = layer.key.weight.min().item()  # 最小值
value_weight_max = layer.value.weight.max().item()  # 最大值
value_weight_min = layer.value.weight.min().item()  # 最小值

print(f"Maximum value in 'query.weight': {query_weight_max}")
print(f"Minimum value in 'query.weight': {query_weight_max}")
print(f"Maximum value in 'key.weight': {key_weight_max}")
print(f"Minimum value in 'key.weight': {key_weight_min}")
print(f"Maximum value in 'value.weight': {value_weight_max}")
print(f"Minimum value in 'value.weight': {value_weight_min}")


print(f"Number of parameters in 'query.weight': {weight_params}")
print(f"Number of parameters in 'query.bias': {bias_params}")

plot_distribution(layer.query.weight, "Query Weight Distribution", axes[0, 0])
plot_distribution(layer.query.bias, "Query Bias Distribution", axes[0, 1])
plot_distribution(layer.key.weight, "Key Weight Distribution", axes[1, 0])
plot_distribution(layer.key.bias, "Key Bias Distribution", axes[1, 1])
plot_distribution(layer.value.weight, "Value Weight Distribution", axes[2, 0])
plot_distribution(layer.value.bias, "Value Bias Distribution", axes[2, 1])

plt.tight_layout()
plt.show()


