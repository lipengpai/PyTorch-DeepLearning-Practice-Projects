import os
import torch
from PIL import Image
from torchvision import transforms
from model import Net

def predict_image(image_path, model, device, classes=['Cat', 'Dog']):
    model.eval()
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.01,contrast=0.01,saturation=0.01,hue=0.01),
        transforms.RandomResizedCrop(150, scale=(0.8,1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225]),
    ])
    img = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(img)
    pred = out.argmax(1).item()
    return classes[pred]

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Net().to(device)
    model.load_state_dict(torch.load('./models/model_weights.pth', map_location=device))

    output_dir = "./predictions"
    os.makedirs(output_dir, exist_ok=True)

    data_dir = './data'
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    if not image_files:
        print("未在 ./data 目录下找到图片。")
    else:
        for img_name in image_files:
            img_path = os.path.join(data_dir, img_name)
            pred = predict_image(img_path, model, device)
            print(f"{img_name} 预测为: {pred}")
            img_disp = Image.open(img_path)
            plt.imshow(img_disp)
            plt.title(f'Prediction: {pred}')
            plt.axis('off')
            save_path = os.path.join(output_dir, f"prediction_{img_name}")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            print(f"预测结果图片已保存到: {save_path}")