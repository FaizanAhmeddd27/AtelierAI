from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

from .network import AdaINNet, adain

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "decoder_final.pth"

_model = None


def get_model():
    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

        model = AdaINNet()
        model.decoder.load_state_dict(
            torch.load(MODEL_PATH, map_location=device, weights_only=True)
        )
        model.to(device)
        model.eval()

        _model = model

    return _model


def preview_image(path, max_size=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_size / max(w, h)
    return img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.LANCZOS,
    )


def to_tensor(img):
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_pil(tensor):
    arr = (tensor.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def load_image(path, max_size=512):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    scale = max_size / max(w, h)
    img = img.resize(
        (max(1, int(w * scale)), max(1, int(h * scale))),
        Image.LANCZOS,
    )
    return to_tensor(img)


def color_transfer(result, target):
    src = result.convert("YCbCr")
    ref = target.convert("YCbCr")

    s = np.array(src).astype(np.float32)
    r = np.array(ref).astype(np.float32)

    for c in range(3):
        ms, ss = s[:, :, c].mean(), s[:, :, c].std() + 1e-6
        mr, sr = r[:, :, c].mean(), r[:, :, c].std() + 1e-6
        s[:, :, c] = (s[:, :, c] - ms) * (sr / ss) + mr

    out = Image.fromarray(
        np.clip(s, 0, 255).astype(np.uint8),
        "YCbCr",
    )

    return out.convert("RGB")


def style_transfer(content_path, style_path, alpha=0.7, max_size=512):
    model = get_model()

    content = load_image(content_path, max_size).to(device)
    style = load_image(style_path, max_size).to(device)

    with torch.no_grad():
        content_feat = model.encode(content)
        style_feat = model.encode(style)

        t = adain(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat

        output = model.decoder(t)

    output = output.cpu().clamp(0, 1).squeeze(0)

    result_img = to_pil(output)

    style_img = to_pil(style.cpu().squeeze(0))

    colored = color_transfer(result_img, style_img)

    result_img = Image.blend(result_img, colored, alpha)
    result_img = ImageEnhance.Sharpness(result_img).enhance(1.2)

    return result_img