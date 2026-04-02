from PIL import Image
import os

def patch_image(image_path, output_dir, patch_size=224):
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    os.makedirs(output_dir, exist_ok=True)

    patch_count = 0
    for top in range(0, h - patch_size + 1, patch_size):
        for left in range(0, w - patch_size + 1, patch_size):
            patch = img.crop((left, top, left + patch_size, top + patch_size))
            patch.save(os.path.join(output_dir, f"patch_{patch_count}.jpg"))
            patch_count += 1

    print(f"Saved {patch_count} patches from {image_path}")


def patch_all(input_dir, output_dir, patch_size=224):
    for fname in os.listdir(input_dir):
        if fname.lower().endswith((".jpg", ".jpeg", ".png")):
            name = os.path.splitext(fname)[0]
            patch_image(
                os.path.join(input_dir, fname),
                os.path.join(output_dir, name),
                patch_size
            )
