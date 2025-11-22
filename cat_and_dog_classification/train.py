import torch, os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model import Net

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.01,contrast=0.01,saturation=0.01,hue=0.01),
        transforms.RandomResizedCrop(150, scale=(0.8,1.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    root_dir = "./datasets/PetImagesSample"
    train_ds = datasets.ImageFolder(os.path.join(root_dir,'train'), transform)
    test_ds  = datasets.ImageFolder(os.path.join(root_dir,'test'), transform)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=32)

    model = Net().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    epochs = 20
    for epoch in range(epochs):
        # 训练
        model.train()
        train_loss, train_acc, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            loss = criterion(outs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
            train_acc  += (outs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/total:.3f} | Train Acc: {train_acc/total:.3f}")
        # 验证
        model.eval()
        test_loss, test_acc, tot = 0,0,0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outs = model(imgs)
                loss = criterion(outs, labels)
                test_loss += loss.item() * imgs.size(0)
                test_acc  += (outs.argmax(dim=1) == labels).sum().item()
                tot += labels.size(0)
        print(f"           | Test Loss: {test_loss/tot:.3f} | Test Acc: {test_acc/tot:.3f}")
    
    os.makedirs('./models', exist_ok=True)
    torch.save(model.state_dict(), './models/model_weights.pth')
    print("训练完成，模型已保存.")

if __name__ == '__main__':
    main()