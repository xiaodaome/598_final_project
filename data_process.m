clear;
load("hidden_states.mat");
load("weights.mat");

num_attention_heads = 16;
attention_head_size = 64;

hidden_states = data.hidden_states;
hidden_states_mat = squeeze(hidden_states); 

mixed_query_layer = hidden_states_mat * query_weight' + query_bias;

key_layer = transpose_for_scores(hidden_states_mat * key_weight' + key_bias, attention_head_size, num_attention_heads);

value_layer = transpose_for_scores(hidden_states_mat * value_weight' + value_bias, attention_head_size, num_attention_heads);

% Compute query_layer
query_layer = transpose_for_scores(mixed_query_layer, attention_head_size, num_attention_heads);

% Compute attention_scores
key_layer_transpose = permute(key_layer, [1, 3, 2]);
attention_scores = zeros(16, 577, 577);
for i = 1:size(query_layer, 1)
    attention_scores(i, :, :) = squeeze(query_layer(i, :, :)) * squeeze(key_layer_transpose(i, :, :));
end

% Scale attention_scores
attention_scores = attention_scores ./ sqrt(attention_head_size);
attention_scores_0 = squeeze(attention_scores(1,1,:));

% Apply softmax
attention_probs = zeros(size(attention_scores));
for i = 1:size(attention_probs, 1)
    for j = 1:size(attention_probs, 2)
        attention_probs(i, j, :) = softmax(squeeze(attention_scores(i,j,:)), 1);
    end
end
attention_probs_test = softmax(squeeze(attention_scores(1,1,:)), 1);
attention_probs_0 = squeeze(attention_probs(1,1,:));

% Compute context_layer
context_layer = zeros(16, 577, 64);
for i = 1:size(attention_probs, 1)
    context_layer(i, :, :) = squeeze(attention_probs(i, :, :)) * squeeze(value_layer(i, :, :));
end


diff = context_layer - squeeze(data.context_layer1(1,:,:, :));
diff_0 = max(squeeze(diff(1,:, :)));



function layer_permuted = transpose_for_scores(layer, attention_head_size, num_attention_heads)
    new_x_shape = [size(layer, 1), attention_head_size, num_attention_heads];  % [batch_size, attention_head_size, num_attention_heads]
    layer_reshaped = reshape(layer, new_x_shape);
    layer_reshaped = permute(layer_reshaped, [1, 3, 2]);  % [batch_size, num_attention_heads, attention_head_size]
    layer_permuted = permute(layer_reshaped, [2, 1, 3]);  % [num_attention_heads, batch_size, attention_head_size]
end

