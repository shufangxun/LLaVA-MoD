def num_param(vocab_size, hidden_size, num_hidden_layers, intermediate_size, ffn_factor, freq_moe_layer, num_experts):
    num_moe_layers = num_hidden_layers // freq_moe_layer
    num_extra_ffns = num_moe_layers * (num_experts - 1)

    moe_num_params = vocab_size * hidden_size + \
                     num_hidden_layers * (
                             hidden_size * hidden_size * 4 + hidden_size * intermediate_size * ffn_factor + hidden_size * 2) + \
                     hidden_size + hidden_size * vocab_size + \
                     num_extra_ffns * (hidden_size * intermediate_size * ffn_factor + hidden_size * 2) + \
                     num_moe_layers * (hidden_size * num_experts)

    print(f'Number of parameters of MoE Model (B) /w {num_experts} experts: {round(moe_num_params / 1e9, 2)}')
    return round(moe_num_params / 1e9, 1)


if __name__ == '__main__':
    model_qwen_2_1_5b = dict(vocab_size=151936,
                             hidden_size=1536,
                             num_hidden_layers=28,
                             intermediate_size=8960,
                             ffn_factor=3,
                             freq_moe_layer=2)

    num_param(**model_qwen_2_1_5b, num_experts=1)
