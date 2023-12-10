import torch.nn as nn
from torchvision.models import resnet


class Network(nn.Module):
    """
    An encoder network (image -> feature_dim)
    """
    def __init__(self, arch, feature_dim, cifar_small_image=False):
        super(Network, self).__init__()

        resnet_arch = getattr(resnet, arch)
        net = resnet_arch(num_classes=feature_dim)

        self.encoder = []
        for name, module in net.named_children():
            if isinstance(module, nn.Linear):
                self.encoder.append(nn.Flatten(1))
                self.encoder.append(module)
            else:
                if cifar_small_image:
                    # replace first conv from 7x7 to 3x3
                    if name == 'conv1':
                        module = nn.Conv2d(module.in_channels, module.out_channels,
                                           kernel_size=3, stride=1, padding=1, bias=False)
                    # drop first maxpooling
                    if isinstance(module, nn.MaxPool2d):
                        continue
                self.encoder.append(module)
        self.encoder = nn.Sequential(*self.encoder)

    def forward(self, x):
        return self.encoder(x)
