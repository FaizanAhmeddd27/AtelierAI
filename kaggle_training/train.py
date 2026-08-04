import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from network import AdaINNet
from dataset import FlatFolderDataset


def adjust_lr(optimizer, iteration_count, lr=1e-4, lr_decay=5e-5):
    new_lr = lr / (1.0 + lr_decay * iteration_count)
    for g in optimizer.param_groups:
        g['lr'] = new_lr


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    content_dir = "/kaggle/working/data/content"
    style_dir   = "/kaggle/working/data/style"
    save_dir    = "/kaggle/working/output"
    os.makedirs(save_dir, exist_ok=True)

    crop_size   = 256
    batch_size  = 8
    max_iter    = 40000        # adjust based on time you have (see notes below)
    style_weight = 10.0
    content_weight = 1.0
    lr = 1e-4
    save_every = 1000

    tf = transforms.Compose([
        transforms.Resize(crop_size + 30),
        transforms.RandomCrop(crop_size),
        transforms.ToTensor()
    ])

    content_ds = FlatFolderDataset(content_dir, tf)
    style_ds   = FlatFolderDataset(style_dir, tf)
    print("Content images:", len(content_ds), "| Style images:", len(style_ds))

    def make_loader(ds):
        return iter(DataLoader(ds, batch_size=batch_size, shuffle=True,
                                num_workers=2, drop_last=True))

    content_loader = make_loader(content_ds)
    style_loader   = make_loader(style_ds)

    model = AdaINNet().to(device)
    model.decoder.train()

    optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)

    start_iter = 0
    resume_path = os.path.join(save_dir, "checkpoint_latest.pth")
    if os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location=device)
        model.decoder.load_state_dict(ckpt['decoder'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_iter = ckpt['iter']
        print(f"Resumed training from iteration {start_iter}")

    for it in tqdm(range(start_iter, max_iter)):
        adjust_lr(optimizer, it, lr)

        try:
            content_imgs = next(content_loader)
        except StopIteration:
            content_loader = make_loader(content_ds)
            content_imgs = next(content_loader)

        try:
            style_imgs = next(style_loader)
        except StopIteration:
            style_loader = make_loader(style_ds)
            style_imgs = next(style_loader)

        content_imgs = content_imgs.to(device)
        style_imgs = style_imgs.to(device)

        loss_c, loss_s = model(content_imgs, style_imgs)
        loss = content_weight * loss_c + style_weight * loss_s

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if it % 100 == 0:
            print(f"iter {it} | total {loss.item():.4f} | content {loss_c.item():.4f} | style {loss_s.item():.4f}")

        if (it + 1) % save_every == 0 or (it + 1) == max_iter:
            torch.save(model.decoder.state_dict(),
                       os.path.join(save_dir, f"decoder_iter_{it+1}.pth"))
            torch.save({'decoder': model.decoder.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter': it + 1}, resume_path)

    torch.save(model.decoder.state_dict(), os.path.join(save_dir, "decoder_final.pth"))
    print("Training complete! Final weights at:", os.path.join(save_dir, "decoder_final.pth"))


if __name__ == '__main__':
    train()