import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision.models.feature_extraction import create_feature_extractor


def feature_extractor(model, modeltype):
    # _, eval_nodes = get_graph_node_names(model)
    # print(eval_nodes)
    return_nodes = {'lenet1': {'feature_extractor.3': 'feat_block', 'classifier.0': 'logit'},
                    'lenet5': {'feature_extractor.4': 'feat_block', 'classifier.4': 'logit'},
                    'vgg': {'features.33': 'block4_maxpool', 'classifier': 'logit'},
                    'resnet9': {'residual2.module.0.0': 'residual2_layer0', 'linear': 'logit'},
                    'resnet18': {'layer4.0.bn2': 'block4_bn', 'linear': 'logit'}}

    feat_extractor = create_feature_extractor(model, return_nodes=return_nodes[modeltype])
    return feat_extractor


class NeuralNet(nn.Module):
    def __init__(self, epochs, model_path, device):
        super(NeuralNet, self).__init__()
        self.epochs = epochs
        self.model_path = model_path
        self.device = device

    def train_model(self, train_loader, optimizer, criterion):
        n_total_steps = len(train_loader)

        # Cyclic LR with single triangle
        lr_peak_epoch = 5
        lr_schedule = np.interp(np.arange((self.epochs + 1) * n_total_steps),
                                [0, lr_peak_epoch * n_total_steps, self.epochs * n_total_steps],
                                [0, 1, 0])
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_schedule.__getitem__)

        for epoch in range(self.epochs):
            total_loss = 0
            for i, (images, labels) in tqdm(enumerate(train_loader)):
                # origin shape: [100, 1, 28, 28]
                # resized: [100, 784]
                self.train()
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.forward(images)
                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                total_loss += float(loss)

                # if (i+1) % 100 == 0:
                #     print (f'Epoch [{epoch+1}/{self.epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')
            if (epoch+1) % 5 == 0:
                # print every 5 epochs
                acc = self.evaluate(train_loader)
                print(f'Epoch [{epoch+1}/{self.epochs}] loss: {total_loss / n_total_steps}, train accuracy: {acc} %')

    def evaluate(self, test_loader):
        self.eval()
        with torch.no_grad():
            n_correct = 0
            n_samples = 0
            for images, labels in test_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.forward(images)#.cpu().detach().numpy()
                # max returns (value ,index)
                _, predicted = torch.max(outputs.data, 1)
                n_samples += labels.size(0)
                n_correct += (predicted == labels).sum().item()

        acc = 100.0 * n_correct / n_samples
        # print(f'Accuracy of the network on the 10000 test images: {acc} %')
        return acc

    def save_model(self):
        torch.save(self.state_dict(), self.model_path)

    def load_model(self):
        self.load_state_dict(torch.load(self.model_path))

#######################
# LeNet1
#######################
class LENET1(NeuralNet):
#Adapted from https://github.com/grvk/lenet-1/blob/master/LeNet-1.ipynb
    def __init__(self, epochs, model_path, device, num_classes):
        super().__init__(epochs, model_path, device)
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=12, kernel_size=5),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=12*4*4, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
   

#######################
# LeNet5
#######################
class LENET5(NeuralNet):
    def __init__(self, epochs, model_path, device, num_classes):
        super().__init__(epochs, model_path, device)
        self.feature_extractor = nn.Sequential(            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*5*5, out_features=120),
            nn.Tanh(),
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


#######################
# VGG
#######################
class VGG(NeuralNet):
    def __init__(self, epochs, model_path, device, num_classes):
        super().__init__(epochs, model_path, device)
        cfg = (64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M')
        self.features = self._make_layers(cfg)
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
    
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

#######################
# ResNet
#######################

class Mul(nn.Module):
    def __init__(self, weight):
       super(Mul, self).__init__()
       self.weight = weight

    def forward(self, x):
        return x * self.weight


class Residual(nn.Module):
    def __init__(self, module):
        super(Residual, self).__init__()
        self.module = module

    def forward(self, x):
        return x + self.module(x)


class RESNET9(NeuralNet):
    def __init__(self, epochs, model_path, device, num_classes=10):
        super().__init__(epochs, model_path, device)

        num_dim = 3  # 3 if RGB image, else 1
        """
        self.features = nn.Sequential(
            self.conv_bn(num_dim, 64, kernel_size=3, stride=1, padding=1),
            self.conv_bn(64, 128, kernel_size=5, stride=2, padding=2),
            Residual(torch.nn.Sequential(self.conv_bn(128, 128), self.conv_bn(128, 128))),
            self.conv_bn(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            Residual(torch.nn.Sequential(self.conv_bn(256, 256), self.conv_bn(256, 256))),
            self.conv_bn(256, 128, kernel_size=3, stride=1, padding=0),
            torch.nn.AdaptiveMaxPool2d((1, 1)),
            Flatten(),
            torch.nn.Linear(128, num_classes, bias=False),
            Mul(0.2),
        )
        """

        self.conv1 = self.conv_bn(num_dim, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = self.conv_bn(64, 128, kernel_size=5, stride=2, padding=2)
        self.residual1 = Residual(nn.Sequential(self.conv_bn(128, 128), self.conv_bn(128, 128)))
        self.conv3 = self.conv_bn(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.residual2 = Residual(nn.Sequential(self.conv_bn(256, 256), self.conv_bn(256, 256)))
        self.conv4 = self.conv_bn(256, 128, kernel_size=3, stride=1, padding=0)
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.linear = nn.Linear(128, num_classes, bias=False)

    def conv_bn(self, channels_in, channels_out, kernel_size=3, stride=1, padding=1, groups=1):
        return nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=kernel_size,
                      stride=stride, padding=padding, groups=groups, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual1(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.residual2(x)
        x = self.maxpool(self.conv4(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.linear(x)
        # x = Mul(0.2)(x)
        x = x * 0.2

        """
        out = self.features(x)
        out = self.classifier(out)
        return out
        """

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class RESNET18(NeuralNet):
    def __init__(self, epochs, model_path, device, num_classes=10): #(self, block, num_blocks, num_classes=10):
        super().__init__(epochs, model_path, device)
        self.in_planes = 64
        """
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            self._make_layer(BasicBlock, 64, 2, stride=1),
            self._make_layer(BasicBlock, 128, 2, stride=2),
            self._make_layer(BasicBlock, 256, 2, stride=2),
        )
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

        """
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.linear = nn.Linear(512*BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        """
        out = self.features(x)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
