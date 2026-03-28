import torch
import torch.nn as tnn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from tqdm import tqdm

BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCH = 40
N_CLASSES = 2


# ==================== 新增：CBAM注意力模块 ====================
class ChannelAttention(tnn.Module):
    """通道注意力模块：关注特征通道重要性"""

    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = tnn.AdaptiveAvgPool2d(1)   #全局平均池化
        self.max_pool = tnn.AdaptiveMaxPool2d(1)   #全局最大池化

        # 共享MLP
        self.mlp = tnn.Sequential(
            tnn.Linear(channels, channels // reduction, bias=False),   #降维
            tnn.ReLU(inplace=True),
            tnn.Linear(channels // reduction, channels, bias=False)   #升维
        )
        self.sigmoid = tnn.Sigmoid()   #激活函数，输出0-1的注意力权重

    def forward(self, x):
        b, c, _, _ = x.size()
        #两种池化后的通道注意力
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.mlp(avg_out)
        max_out = self.max_pool(x).view(b, c)
        max_out = self.mlp(max_out)
        #合并两种注意力并使用Sigmoid
        attention = self.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention.expand_as(x)   #注意力权重×原特征


class SpatialAttention(tnn.Module):
    """空间注意力模块：关注伪造边界位置"""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        #将2个特征图合并后卷积，生成空间注意力特征图
        self.conv = tnn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = tnn.Sigmoid()

    def forward(self, x):
        #沿通道维度计算平均值和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        #拼接两种池化结果
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(combined)   #卷积生成空间注意力
        return x * self.sigmoid(attention)   #注意力权重×原特征


class CBAM(tnn.Module):
    """CBAM注意力模块：让网络聚焦于伪造边界"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        #顺序执行通道注意力和空间注意力
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)   #应用通道注意力
        x = self.spatial_attention(x)   #应用空间注意力
        return x


# ==================== 注意力模块定义结束 ====================

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 随机裁裁剪到224×224
    transforms.RandomHorizontalFlip(),  # 随机水平翻转（数据增强）
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet标准化
                         std=[0.229, 0.224, 0.225]),
])

# 加载数据
trainData = dsets.ImageFolder('/root/autodl-tmp/dataset/train', transform)
testData = dsets.ImageFolder('/root/autodl-tmp/dataset/val', transform)

trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


# 定义卷积层
def conv_layer(chann_in, chann_out, k_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, padding=p_size),
        tnn.BatchNorm2d(chann_out),  # 批归一化
        tnn.ReLU()  # 激活函数
    )
    return layer


# 定义卷积块（带可选CBAM注意力）
def vgg_conv_block(in_list, out_list, k_list, p_list, pooling_k, pooling_s, use_attention=False,
                   attention_channels=None):
    layers = [conv_layer(in_list[i], out_list[i], k_list[i], p_list[i]) for i in range(len(in_list))]
    layers += [tnn.MaxPool2d(kernel_size=pooling_k, stride=pooling_s)]  # 最大池化

    # 在卷积块后插入CBAM注意力模块，让网络聚焦于伪造边界
    if use_attention and attention_channels is not None:
        layers += [CBAM(channels=attention_channels)]   #添加CBAM模块

    return tnn.Sequential(*layers)


# 定义全连接层
def vgg_fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out),
        tnn.BatchNorm1d(size_out),  # 1D批归一化
        tnn.ReLU()
    )
    return layer


# 定义模型（在部分卷积块后插入注意力模块）
class VGG16(tnn.Module):
    def __init__(self, n_classes=1000):
        super(VGG16, self).__init__()

        # 5个卷积块，在layer2、layer3、layer4、layer5后插入注意力模块
        self.layer1 = vgg_conv_block([3, 64], [64, 64], [3, 3], [1, 1], 2, 2)  # 2层64通道
        self.layer2 = vgg_conv_block([64, 128], [128, 128], [3, 3], [1, 1], 2, 2,)  # 2层128通道
        self.layer3 = vgg_conv_block([128, 256, 256], [256, 256, 256], [3, 3, 3], [1, 1, 1], 2, 2,)  # 3层256通道
        self.layer4 = vgg_conv_block([256, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2, use_attention=True,
                                     attention_channels=512)  # 3层512通道+注意力
        self.layer5 = vgg_conv_block([512, 512, 512], [512, 512, 512], [3, 3, 3], [1, 1, 1], 2, 2, use_attention=True,
                                     attention_channels=512)  # 3层512通道

        # 全连接层
        self.layer6 = vgg_fc_layer(7 * 7 * 512, 4096)  # 输入：512*7*7特征图展平
        self.layer7 = vgg_fc_layer(4096, 4096)
        self.layer8 = tnn.Linear(4096, n_classes)  # 输出层

    # 向前传播
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        vgg16_features = self.layer5(out)  # 保存特征用于可视化
        out = vgg16_features.view(out.size(0), -1)  # 展平
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)

        return vgg16_features, out  # 返回特征和分类结果


vgg16 = VGG16(n_classes=N_CLASSES)
vgg16.cuda()

# 损失函数，优化器和学习率调度器
cost = tnn.CrossEntropyLoss()
# 分层学习率设置：卷积层1e-5，注意力模块1e-3

attention_params = []
conv_params = []

for name, param in vgg16.named_parameters():
    if 'channel_attention' in name or 'spatial_attention' in name:
        attention_params.append(param)
    else:
        conv_params.append(param)

optimizer = torch.optim.Adam([
    {'params': conv_params, 'lr': 1e-5},      # 卷积层：极小学习率，微调预训练特征
    {'params': attention_params, 'lr': 1e-3}   # 注意力模块：大学习率，快速学习聚焦能力
])

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

# 训练模型
for epoch in range(EPOCH):
    avg_loss = 0
    cnt = 0
    total_correct = 0
    total_samples = 0

    # 训练进度条
    train_bar = tqdm(trainLoader, desc=f'Epoch {epoch + 1}/{EPOCH}')

    for images, labels in train_bar:  # 修改循环变量名
        images = images.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()  # 梯度清零
        _, outputs = vgg16(images)  # 向前传播
        loss = cost(outputs, labels)  # 计算损失

        # 计算当前batch的准确率
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total_correct += correct
        total_samples += labels.size(0)
        batch_acc = 100.0 * correct / labels.size(0)
        epoch_acc = 100.0 * total_correct / total_samples

        # 更新统计信息
        avg_loss += loss.data
        cnt += 1

        # 在进度条中显示实时损失和准确率
        train_bar.set_postfix({
            'loss': f'{loss.data:.4f}',
            'avg_loss': f'{avg_loss / cnt:.4f}',
            'batch_acc': f'{batch_acc:.2f}%',
            'epoch_acc': f'{epoch_acc:.2f}%'
        })

        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数

    # 打印每个epoch的平均损失和准确率
    epoch_final_acc = 100.0 * total_correct / total_samples
    print(f"[Epoch {epoch + 1}] Average loss: {avg_loss / cnt:.4f}, Accuracy: {epoch_final_acc:.2f}%")
    scheduler.step(avg_loss)

# 测试模型
vgg16.eval()
correct = 0
total = 0

# 添加测试进度条
test_bar = tqdm(testLoader, desc='Testing')

for images, labels in test_bar:  # 修改循环变量名
    images = images.cuda()
    _, outputs = vgg16(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

    # 在进度条中显示实时准确率
    current_acc = 100 * correct / total
    test_bar.set_postfix({'accuracy': f'{current_acc:.2f}%'})

# 打印最终准确率
print(f"Final Accuracy: {100 * correct / total:.2f}%")

# Save the Trained Model
torch.save(vgg16.state_dict(), 'cnn.pkl')
