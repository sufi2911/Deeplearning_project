import os
import shutil
import pandas as pd

base_dir = "dataset/test"
image_dir = os.path.join(base_dir, "image")

real_csv = os.path.join(base_dir, "image_labels_real.csv")
fake_csv = os.path.join(base_dir, "image_labels_fake.csv")

real_dir = os.path.join(base_dir, "real")
fake_dir = os.path.join(base_dir, "fake")

os.makedirs(real_dir, exist_ok=True)
os.makedirs(fake_dir, exist_ok=True)

# Load CSVs (no header assumed)
real_df = pd.read_csv(real_csv, header=None)
fake_df = pd.read_csv(fake_csv, header=None)

def move_images(df, target_dir):
    for _, row in df.iterrows():
        relative_path = row[0]

        # FIX PATH HERE
        relative_path = relative_path.replace("fake_image", "image").replace("real_image", "image")

        src_path = os.path.join(base_dir, relative_path)

        if os.path.exists(src_path):
            shutil.copy(src_path, target_dir)
        else:
            print(f"Missing: {src_path}")
# Move real images
move_images(real_df, real_dir)

# Move fake images
move_images(fake_df, fake_dir)

print("✅ Dataset successfully reorganized into real/fake folders.")