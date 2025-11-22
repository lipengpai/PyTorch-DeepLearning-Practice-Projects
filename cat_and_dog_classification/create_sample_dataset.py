import os
import shutil
import random

def select_and_split(src_dir, dst_dir, n_per_class=100, test_ratio=0.2, seed=42):
    random.seed(seed)
    classes = ['Cat', 'Dog']
    for cls in classes:
        src_cls_dir = os.path.join(src_dir, cls)
        imgs = [f for f in os.listdir(src_cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))]
        imgs = [f for f in imgs if os.path.getsize(os.path.join(src_cls_dir,f)) > 0]  # 跳过空文件
        if len(imgs) < n_per_class:
            print(f"警告: '{cls}'类别只找到{len(imgs)}张图片，少于采样数{n_per_class}")
            sample_imgs = imgs
        else:
            sample_imgs = random.sample(imgs, n_per_class)

        # 切分训练集和测试集
        random.shuffle(sample_imgs)
        n_test = int(len(sample_imgs) * test_ratio)
        test_imgs = sample_imgs[:n_test]
        train_imgs = sample_imgs[n_test:]

        # 复制训练集
        train_dst_cls = os.path.join(dst_dir, 'train', cls)
        os.makedirs(train_dst_cls, exist_ok=True)
        for img in train_imgs:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(train_dst_cls, img))
        # 复制测试集
        test_dst_cls = os.path.join(dst_dir, 'test', cls)
        os.makedirs(test_dst_cls, exist_ok=True)
        for img in test_imgs:
            shutil.copy(os.path.join(src_cls_dir, img), os.path.join(test_dst_cls, img))

        print(f"已采样 {cls}: 共{len(sample_imgs)}张, 其中训练{len(train_imgs)}张, 测试{len(test_imgs)}张")

if __name__ == "__main__":
    # 根据实际路径修改
    SOURCE_DIR = "./datasets/PetImages/train"               # 原始大数据集目录
    TARGET_DIR = "./datasets/PetImagesSample"         # 新的小样本集目录
    SAMPLES_PER_CLASS = 1000                           # 每类采样图片数
    TEST_RATIO = 0.2                                  # 测试集比例

    select_and_split(SOURCE_DIR, TARGET_DIR, n_per_class=SAMPLES_PER_CLASS, test_ratio=TEST_RATIO)
    print("\n小样本采样&分割完成，可用于快速训练！")