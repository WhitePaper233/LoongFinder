import torch
from torchvision import datasets, transforms, models
from PIL import Image
import os
import torch.nn.functional as F

# 图片转换器
data_transforms = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 加载模型
model = models.resnet50(weights=None)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 2)

# 加载模型权重
model.load_state_dict(torch.load('./model/best_model.pth'))

# 检查是否有可用的GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)  # 将模型移动到GPU


# 测试推理
def test_inference(image_path):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.resize((128, 128))
    image = data_transforms(image).unsqueeze(0)
    image = image.to(device)  # 将输入数据移动到GPU

    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        is_loong = 'loong' if predicted.item() == 0 else 'notloong'
        confidence = probabilities[0][predicted.item()].item()
        return is_loong, confidence


# 测试推理
test_dir = './tests'
for file in sorted(os.listdir(test_dir)):
    image_path = os.path.join(test_dir, file)
    is_loong, confidence = test_inference(image_path)
    print(f'Image: {file}, Is Loong: {is_loong}, Confidence: {confidence}')
