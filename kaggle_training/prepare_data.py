import os, glob, zipfile, random, shutil
from io import BytesIO
from PIL import Image

random.seed(42)

N_STYLE = 10000
N_CONTENT = 10000

STYLE_OUT   = "/kaggle/working/data/style"
CONTENT_OUT = "/kaggle/working/data/content"
os.makedirs(STYLE_OUT, exist_ok=True)
os.makedirs(CONTENT_OUT, exist_ok=True)


def find_dataset_root(keyword, base='/kaggle/input'):
    """Recursively search for a folder whose name contains the keyword."""
    matches = []
    for root, dirs, files in os.walk(base):
        for d in dirs:
            if keyword.lower() in d.lower():
                matches.append(os.path.join(root, d))
        # also check if current root itself matches (leaf case)
        if keyword.lower() in os.path.basename(root).lower():
            matches.append(root)
    # return the shortest path (closest to /kaggle/input, avoids over-nesting)
    matches = sorted(set(matches), key=len)
    return matches[0] if matches else None


# ---------------- STYLE: Painter by Numbers (competition, multi-zip) ----------------
style_root = "/kaggle/input/competitions/painter-by-numbers"
if not os.path.exists(style_root):
    style_root = find_dataset_root('painter-by-numbers')

if style_root is None:
    raise FileNotFoundError("Could not find Painter by Numbers folder.")

print("Using style dataset root:", style_root)

# Get ALL train zip parts, sorted (train.zip, train_1.zip, train_2.zip, ...)
all_zips = glob.glob(os.path.join(style_root, '*.zip'))
train_zips = sorted([z for z in all_zips if 'train' in os.path.basename(z).lower()])
print(f"Found {len(train_zips)} train zip file(s):")
for z in train_zips:
    print("  -", os.path.basename(z), f"({os.path.getsize(z)/1e6:.1f} MB)")

count = 0
for zip_path in train_zips:
    if count >= N_STYLE:
        break
    print(f"\nExtracting from {os.path.basename(zip_path)} ...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        names = [n for n in z.namelist() if n.lower().endswith(('.jpg', '.jpeg', '.png'))]
        random.shuffle(names)
        for n in names:
            if count >= N_STYLE:
                break
            try:
                data = z.read(n)
                Image.open(BytesIO(data)).verify()
            except Exception:
                continue
            with open(os.path.join(STYLE_OUT, f"style_{count}.jpg"), 'wb') as f:
                f.write(data)
            count += 1
            if count % 1000 == 0:
                print(f"{count}/{N_STYLE} style images extracted")

print(f"\n✅ Style extraction done: {count} images in {STYLE_OUT}")


# ---------------- CONTENT: COCO ----------------
content_root = find_dataset_root('coco', base='/kaggle/input/datasets')
if content_root is None:
    content_root = find_dataset_root('coco')  # fallback: search everywhere

if content_root is None:
    raise FileNotFoundError("Could not find COCO dataset folder. Check /kaggle/input/datasets/awsaf49 manually.")

print("\nUsing content dataset root:", content_root)

content_images = glob.glob(os.path.join(content_root, '**', '*.jpg'), recursive=True)
print(f"Found {len(content_images)} content images total under {content_root}")

if len(content_images) == 0:
    raise FileNotFoundError(f"No .jpg files found under {content_root}. "
                             f"Run the os.walk diagnostic to find the correct subfolder (e.g. train2017/).")

random.shuffle(content_images)
selected = content_images[:N_CONTENT]

for i, path in enumerate(selected):
    shutil.copy(path, os.path.join(CONTENT_OUT, f"content_{i}.jpg"))
    if i % 1000 == 0:
        print(f"{i}/{N_CONTENT} content images copied")

print("\nDONE ✅")
print("Style images:", len(os.listdir(STYLE_OUT)))
print("Content images:", len(os.listdir(CONTENT_OUT)))