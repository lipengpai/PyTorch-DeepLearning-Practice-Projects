from PIL import Image
import os

def clean_bad_images(folder):
    total, removed = 0, 0
    for root, _, files in os.walk(folder):
        for f in files:
            if not f.lower().endswith(('.jpg','.jpeg','.png')):
                continue
            fp = os.path.join(root, f)
            total += 1
            try:
                img = Image.open(fp)
                img.verify()  # 检查是否为有效图片
            except Exception:
                print(f"Remove bad image: {fp}")
                os.remove(fp)
                removed += 1
    print(f"扫描图片总数: {total}, 移除坏图: {removed}")

if __name__ == "__main__":
    clean_bad_images("./datasets/PetImages")