import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

BATCH_SIZE = 32
LEARNING_RATE = 0.01
EPOCH = 40
N_CLASSES = 2

# ==================== 数据转换（9通道输入）===================
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


# 自定义数据集类
class FFTImageFolder(dsets.ImageFolder):
    """自定义数据集，自动添加FFT幅度谱和相位谱通道"""

    def __init__(self, root, transform=None):
        super(FFTImageFolder, self).__init__(root, transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

            # 提取FFT幅度谱和相位谱作为额外通道
            # sample: (3, H, W)
            magnitude_channels = []
            phase_channels = []
            for c in range(3):
                channel = sample[c]

                # 2D FFT
                fft_result = torch.fft.fft2(channel)
                fft_shift = torch.fft.fftshift(fft_result)

                # 计算幅度谱
                magnitude = torch.abs(fft_shift)

                # 对数变换
                magnitude = torch.log1p(magnitude)

                # 归一化幅度谱
                magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)

                # 计算相位谱
                phase = torch.angle(fft_shift)

                # 将相位从[-π, π]归一化到[0, 1]
                phase = (phase + np.pi) / (2 * np.pi)

                magnitude_channels.append(magnitude)
                phase_channels.append(phase)

            # 拼接为9通道（3通道RGB + 3通道幅度谱 + 3通道相位谱）
            magnitude_channel = torch.stack(magnitude_channels)  # (3, H, W)
            phase_channel = torch.stack(phase_channels)  # (3, H, W)
            sample = torch.cat([sample, magnitude_channel, phase_channel], dim=0)  # (9, H, W)

            # 标准化（9通道：3通道RGB + 3通道幅度谱 + 3通道相位谱）
            mean = torch.tensor([0.485, 0.456, 0.406, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).view(9, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.229, 0.229, 0.229, 0.229, 0.229]).view(9, 1, 1)
            sample = (sample - mean) / std

        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target



# ==================== 数据加载（使用FFT数据集）===================
trainData = FFTImageFolder('/root/autodl-tmp/dataset/train', transform=train_transform)
testData = FFTImageFolder('/root/autodl-tmp/dataset/val', transform=train_transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


#定义卷积层
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),   #批归一化
        tnn.ReLU()   #激活函数
    )
    return layer

#定义卷积块
def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]   #最大池化
    return tnn.Sequential(*layers)


# 定义全连接层
def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer


# 定义模型（修改为接受9通道输入）
class VGG16_FFT(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16_FFT, self).__init__()

        # 修改第一层以接受9通道输入（3通道RGB + 3通道幅度谱 + 3通道相位谱）
        self.layer1 = vgg_conv_block([9, 64], [64, 64], [3, 3], [1, 1], 2, 2)  # 输入改为9通道
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2, )
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2, )
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2,)
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2,)

        # 全连接层
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)
        self.layer7 = vgg_fc_layer(4096, 4096)
        self.layer8 = tnn.Linear(4096, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)
        out = vgg16_features.view(out.size(0), -1)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out


# 实例化模型
vgg16 = VGG16_FFT(n_classes=N_CLASSES)
vgg16.cuda()

# 损失函数，优化器和学习率调度器
cost = tnn.CrossEntropyLoss()

optimizer = torch.optim.Adam(vgg16.parameters(), lr=LEARNING_RATE)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# 训练模型
for epoch in range(EPOCH):
    avg_loss = 0
    cnt = 0
    total_correct = 0
    total_samples = 0

    train_bar = tqdm(trainLoader, desc=f'Epoch {epoch + 1}/{EPOCH}')

    for images, labels in train_bar:
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        _, outputs = vgg16(images)
        loss = cost(outputs, labels)

        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        batch_acc = 100.0 * correct / labels.size(0)
        epoch_acc = 100.0 * total_correct / total_samples

        avg_loss += loss.data
        cnt += 1

        train_bar.set_postfix({
            'loss': f'{loss.data:.4f}',
            'avg_loss': f'{avg_loss / cnt:.4f}',
            'batch_acc': f'{batch_acc:.2f}%',
            'epoch_acc': f'{epoch_acc:.2f}%'
        })

        loss.backward()
        optimizer.step()

    epoch_final_acc = 100.0 * total_correct / total_samples
    print(f"[Epoch {epoch + 1}] Average loss: {avg_loss / cnt:.4f}, Accuracy: {epoch_final_acc:.2f}%")
    scheduler.step(avg_loss)

# 测试模型
vgg16.eval()
correct = 0
total = 0


test_bar = tqdm(testLoader, desc='Testing')

for images, labels in test_bar:
    images = images.cuda()
    _, outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    current_acc = 100 * correct / total
    test_bar.set_postfix({'accuracy': f'{current_acc:.2f}%'})

print(f"Final Accuracy: {100 * correct / total:.2f}%")

# 保存模型
torch.save(vgg16.state_dict(), 'cnn_fft.pkl')
