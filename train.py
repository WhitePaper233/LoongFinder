import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

# 图片转换器
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载数据集
train_dataset = datasets.ImageFolder('./datasets/train_processed', transform=data_transforms)
val_dataset = datasets.ImageFolder('./datasets/val_processed', transform=data_transforms)

# 创建数据加载器
batch_size = 8  # 增大batch size以提升训练速度
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# 加载模型
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)
model = model.to(device)  # 将模型移动到GPU

# 定义损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for inputs, labels in tqdm(train_loader):
        inputs = inputs.to(device)  # 将输入数据移动到GPU
        labels = labels.to(device)  # 将标签数据移动到GPU

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)  # 将输入数据移动到GPU
            labels = labels.to(device)  # 将标签数据移动到GPU

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Accuracy: {100 * correct / total}%')

# 保存模型
torch.save(model.state_dict(), './model/best_model.pth')
