import os, shutil, random

def split_dataset(root_dir, test_rate=0.1):
    categories = ['Cat', 'Dog']
    for category in categories:
        files = os.listdir(os.path.join(root_dir, category))
        random.shuffle(files)
        test_size = int(len(files) * test_rate)
        train_files = files[:-test_size]
        test_files = files[-test_size:]
        for phase, phase_files in zip(['train','test'], [train_files, test_files]):
            target = os.path.join(root_dir, phase, category)
            os.makedirs(target, exist_ok=True)
            for fname in phase_files:
                src = os.path.join(root_dir, category, fname)
                dst = os.path.join(target, fname)
                if os.path.exists(src): shutil.move(src, dst)
    print("数据集划分完成。")

if __name__ == '__main__':
    split_dataset('./datasets/PetImages', test_rate=0.1)