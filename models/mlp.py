from torch import nn

def mlp(mp):
    layers = []
    in_size = mp.input_size
    for size in mp.hidden_sizes:
        layers.append(nn.Linear(in_size, size))
        layers.append(mp.activation)
        in_size = size
    layers.append(nn.Linear(in_size, mp.output_size))
    layers.append(mp.output_activation)
    return nn.Sequential(*layers).to(mp.device)