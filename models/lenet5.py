from torch import nn

def lenet5(mp):
    in_size = mp.input_size
    conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)
    max_pool1 = nn.MaxPool2d(kernel_size=2)
    conv2 = torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)
    max_pool2 = torch.nn.MaxPool2d(kernel_size=2)
    fc1 = torch.nn.Linear(16*5*5, 120)
    fc2 = torch.nn.Linear(120, 84)
    fc3 = torch.nn.Linear(84, 10)
    layers = [conv1, mp.activation, max_pool1, conv2, mp.activation, max_pool2, nn.Flatten(start_dim=1), \
              fc1, mp.activation, fc2, mp.activation, fc3, nn.LogSoftmax(dim=-1)]
    return nn.Sequential(*layers).to(mp.device)
    