import os
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = ('.jpg', '.jpeg', '.png')

class FlatFolderDataset(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.paths = [os.path.join(root, f) for f in os.listdir(root)
                      if f.lower().endswith(IMG_EXT)]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert('RGB')
            return self.transform(img)
        except Exception:
            # skip corrupted image, try next one (Painter by Numbers has a few bad files)
            return self.__getitem__((idx + 1) % len(self.paths))