import torch
import torch.nn as nn
import torchvision.models as models


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


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)), nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)), nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)


def build_vgg_encoder():
    vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    layers = list(vgg19.children())
    enc_1 = nn.Sequential(*layers[:2])     # input  -> relu1_1
    enc_2 = nn.Sequential(*layers[2:7])    # relu1_1 -> relu2_1
    enc_3 = nn.Sequential(*layers[7:12])   # relu2_1 -> relu3_1
    enc_4 = nn.Sequential(*layers[12:21])  # relu3_1 -> relu4_1
    return enc_1, enc_2, enc_3, enc_4


class AdaINNet(nn.Module):
    def __init__(self):
        super().__init__()
        e1, e2, e3, e4 = build_vgg_encoder()
        self.enc_1, self.enc_2, self.enc_3, self.enc_4 = e1, e2, e3, e4
        for p in self.parameters():
            p.requires_grad = False   # freeze encoder

        self.decoder = decoder
        self.mse_loss = nn.MSELoss()

    def encode_with_intermediate(self, x):
        results = [x]
        for f in (self.enc_1, self.enc_2, self.enc_3, self.enc_4):
            results.append(f(results[-1]))
        return results[1:]   # [relu1_1, relu2_1, relu3_1, relu4_1]

    def encode(self, x):
        for f in (self.enc_1, self.enc_2, self.enc_3, self.enc_4):
            x = f(x)
        return x

    def calc_content_loss(self, inp, target):
        return self.mse_loss(inp, target)

    def calc_style_loss(self, inp, target):
        i_mean, i_std = calc_mean_std(inp)
        t_mean, t_std = calc_mean_std(target)
        return self.mse_loss(i_mean, t_mean) + self.mse_loss(i_std, t_std)

    def forward(self, content, style, alpha=1.0):
        style_feats = self.encode_with_intermediate(style)
        content_feat = self.encode(content)

        t = adain(content_feat, style_feats[-1])
        t = alpha * t + (1 - alpha) * content_feat

        g_t = self.decoder(t)
        g_t_feats = self.encode_with_intermediate(g_t)

        loss_c = self.calc_content_loss(g_t_feats[-1], t)
        loss_s = self.calc_style_loss(g_t_feats[0], style_feats[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(g_t_feats[i], style_feats[i])

        return loss_c, loss_s