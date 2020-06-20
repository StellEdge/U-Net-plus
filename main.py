# 主程序

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from models import Unet
from datasets import ISBI2012Dataset
from trainers import Trainer

import warnings

warnings.filterwarnings("ignore")

transform = transforms.Compose([transforms.ToTensor()])

isbi = ISBI2012Dataset('./dataset/train-volume.tif', './dataset/train-labels.tif',
                       transforms=transform)

# 训练U-Net
unet = Unet()
unet.cuda()

trainer = Trainer(unet)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(trainer.model.parameters(), lr=1e-3)
loss_history = trainer.fit_generator(isbi, criterion, optimizer, 25);

# 测试阶段
isbi1 = ISBI2012Dataset('./dataset/test-volume.tif', './dataset/test-volume.tif',
                        transforms=transform)

preds = trainer.predict_generator(isbi1)

# 每个像素点取值为0或1，阈值为0.5
thresh = 0.5
for y_pred in preds:
    y_pred[y_pred >= thresh] = 1
    y_pred[y_pred < thresh] = 0

# 显示图片，并保存npy文件
i = 0
for y_pred in preds:
    plt.figure()
    plt.axis('off')
    plt.imshow(y_pred.reshape(512, 512), cmap='gray')
    np.save(str(i) + '.npy', y_pred.reshape(512, 512))
    i = i + 1
