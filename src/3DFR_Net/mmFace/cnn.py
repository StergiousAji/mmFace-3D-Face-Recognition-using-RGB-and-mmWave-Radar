import torch.nn as nn

class MMFace_Padding(nn.Module):
    def __init__(self, num_classes=50):
        super(MMFace_Padding, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding="same"),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding="same"),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.fc = nn.Linear(6400, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    
class MMFace(nn.Module):
    def __init__(self, num_classes=50):
        super(MMFace, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3)
        self.fc = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        # print(x.shape)
        # x = self.conv1(x)
        # print(x.shape)
        # x = self.conv2(x)
        # print(x.shape)
        # x = self.conv3(x)
        # print(x.shape)
        # x = self.maxpool(x)
        # print(x.shape)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # print(x.shape)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# class MMFace3D(nn.Module):
#     def __init__(self, num_classes=50):
#         super(MMFace3D, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv3d(3, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(64),
#             nn.ReLU()
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(128),
#             nn.ReLU()
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv3d(128, 256, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(256),
#             nn.ReLU()
#         )
#         self.conv4 = nn.Sequential(
#             nn.Conv3d(256, 512, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm3d(512),
#             nn.ReLU()
#         )
#         self.maxpool = nn.MaxPool2d()
#         self.fc1 = nn.Linear(512, num_classes)
    
#     def forward(self, x):
#         print(x.shape)
#         x = self.conv1(x)
#         print(x.shape)
#         x = self.maxpool(x)
#         print(x.shape)
#         x = self.conv2(x)
#         print(x.shape)
#         x = self.conv3(x)
#         print(x.shape)
#         x = self.conv4(x)
#         print(x.shape)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         print(x.shape)

        
#         # x = self.conv1(x)
#         # x = self.conv2(x)
#         # x = self.conv3(x)
#         # x = self.conv4(x)
#         # x = x.view(x.size(0), -1)
#         # x = self.fc1(x)
    
#         return x