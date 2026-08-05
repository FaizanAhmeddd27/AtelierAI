from PIL import Image, ImageEnhance, ImageFilter
from pathlib import Path
import numpy as np

import torch
from torchvision import transforms

from .network import AdaINNet, adain

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'models' / 'decoder_final.pth'

model = AdaINNet()
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing checkpoint: {MODEL_PATH}")

model.decoder.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()


def preview_image(path, max_size=512):
    """Return the image resized exactly like the model input, as a PIL image."""
    img = Image.open(path).convert('RGB')
    w, h = img.size
    scale = max_size / max(w, h)
    return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)


def load_image(path, max_size=512):
    img = Image.open(path).convert('RGB')
    w, h = img.size
    scale = max_size / max(w, h)
    img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return transforms.ToTensor()(img).unsqueeze(0)


def color_transfer(result, target):
    """Reinhard color transfer: match result palette to target (the style image)."""
    src = result.convert('YCbCr')
    ref = target.convert('YCbCr')
    s = np.array(src).astype(np.float32)
    r = np.array(ref).astype(np.float32)
    for c in range(3):
        ms, ss = s[:, :, c].mean(), s[:, :, c].std() + 1e-6
        mr, sr = r[:, :, c].mean(), r[:, :, c].std() + 1e-6
        s[:, :, c] = (s[:, :, c] - ms) * (sr / ss) + mr
    out = Image.fromarray(np.clip(s, 0, 255).astype(np.uint8), 'YCbCr')
    return out.convert('RGB')


def style_transfer(content_path, style_path, alpha=0.7, max_size=512):
    content = load_image(content_path, max_size).to(device)
    style = load_image(style_path, max_size).to(device)

    with torch.no_grad():
        content_feat = model.encode(content)
        style_feat = model.encode(style)
        t = adain(content_feat, style_feat)
        t = alpha * t + (1 - alpha) * content_feat
        output = model.decoder(t)

    output = output.cpu().clamp(0, 1).squeeze(0)
    result_img = transforms.ToPILImage()(output)

    style_img = transforms.ToPILImage()(style.cpu().squeeze(0))
    colored = color_transfer(result_img, style_img)
    result_img = Image.blend(result_img, colored, alpha)
    result_img = ImageEnhance.Sharpness(result_img).enhance(1.2)

    return result_img
