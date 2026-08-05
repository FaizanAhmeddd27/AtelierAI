import torch
import torch.nn as nn
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent / "models"
ENCODER_PATH = MODELS_DIR / "vgg19_encoder.pth"


def build_vgg19_features():
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(128, 128, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(256, 512, 3, 1, 1), nn.ReLU(inplace=True),
    )


def calc_mean_std(feat, eps=1e-5):
    N, C = feat.size()[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (content_feat - content_mean.expand(size)) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def build_decoder():
    return nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
        nn.Upsample(scale_factor=2, mode="nearest"),

        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )


class AdaINNet(nn.Module):
    def __init__(self, encoder_path=ENCODER_PATH):
        super().__init__()
        features = build_vgg19_features()
        if encoder_path and Path(encoder_path).exists():
            features.load_state_dict(
                torch.load(encoder_path, map_location="cpu", weights_only=True)
            )
        layers = list(features.children())
        self.enc_1 = nn.Sequential(*layers[:2])
        self.enc_2 = nn.Sequential(*layers[2:7])
        self.enc_3 = nn.Sequential(*layers[7:12])
        self.enc_4 = nn.Sequential(*layers[12:21])

        for p in self.parameters():
            p.requires_grad = False

        self.decoder = build_decoder()

    def encode(self, x):
        for enc in (self.enc_1, self.enc_2, self.enc_3, self.enc_4):
            x = enc(x)
        return x