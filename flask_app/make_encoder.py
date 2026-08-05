"""One-time helper: build a compact VGG19 encoder checkpoint.

The network only uses vgg19 features up to relu4_2 (layers[:21], i.e. 9 conv
layers). Downloading + instantiating the full VGG19 (~143M params / ~574MB in
float32) is what causes the Out-Of-Memory crash on Render's 512MB free tier.
This writes a trimmed checkpoint (~21MB float32) so runtime never loads the
whole model.

Run:  python -m flask_app.make_encoder
Output: flask_app/models/vgg19_encoder.pth
"""

from pathlib import Path

import torch
import torchvision.models as models

OUT = Path(__file__).resolve().parent / "models" / "vgg19_encoder.pth"


def main():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    layers = list(vgg.children())

    kept = layers[:21]  # conv1_1 ... relu4_2 (matches AdaINNet enc groups)

    encoder = torch.nn.Sequential(*kept)
    state = {k: v.clone() for k, v in encoder.state_dict().items()}

    torch.save(state, OUT)
    n_params = sum(t.numel() for t in state.values())
    size_mb = sum(t.numel() * t.element_size() for t in state.values()) / 1e6
    print(f"Saved {OUT}  ({n_params:,} params, {size_mb:.1f} MB float32)")


if __name__ == "__main__":
    main()