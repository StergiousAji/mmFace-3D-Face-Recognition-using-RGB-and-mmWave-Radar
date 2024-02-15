import torch
import torch.nn as nn
import numpy as np
    
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
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.maxpool =  nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Sequential(
            nn.Linear(128*8*2, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(512, 32),
            nn.ReLU()
        )
        self.fc4 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        # Flatten vector before FC layers
        x = x.reshape(-1, 128*8*2)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

# mmFace Feature Extraction: (Frames, 3, 32, 16) -> (512)
class MMFaceFE(nn.Module):
    def __init__(self):
        super(MMFaceFE, self).__init__()
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
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.maxpool =  nn.MaxPool2d(kernel_size=3)
        self.fc1 = nn.Sequential(
            nn.Linear(128*8*2, 1024),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(1024, 512)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        # Flatten vector before FC layers
        x = x.view(-1, 128*8*2)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

# mmFace Classifier: (512) -> (liveness?)
class MMFaceClassifier_Liveness(nn.Module):
    def __init__(self):
        super(MMFaceClassifier_Liveness, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x):
        return self.fc(x)

# InsightFace Feature Extraction: (480, 640, 3) [BGR] -> (512)
def insightface_model(x, det_model, Face, rec_model):
    bboxes, kpss = det_model.detect(x, max_num=0, metric="default")
    # If no face detected, log and add dummy embedding with np.infs
    if len(bboxes) != 1:
        return np.zeros((512)) + np.inf
    
    face = Face(bbox=bboxes[0, :4], kps=kpss[0], det_score=bboxes[0, 4])
    rec_model.get(x, face)
    return face.normed_embedding

# InsightFace Subject Classifier: (512) -> (subject?) OR (512) -> (liveness?)
class InsightFaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super(InsightFaceClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.fc(x)

# Combined Classifier
class IntermediateFusionClassifier(nn.Module):
    def __init__(self, in_features, num_classes):
        super(IntermediateFusionClassifier, self).__init__()
        self.fc1 = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.ReLU(),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

# HYBRID: (Radar (32, 16, 3), RGB Embedding (512)) -> (subject?, liveness?)
class MMFaceHybrid(nn.Module):
    def __init__(self, num_subjects):
        super(MMFaceHybrid, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.maxpool =  nn.MaxPool2d(kernel_size=3)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Sequential(
            nn.Linear(128*8*2, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        # Hybrid: mmFace + InsightFace2D Features
        self.fc_hybrid1 = nn.Sequential(
            nn.Linear(2*512, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.fc_subject = nn.Linear(64, num_subjects)
        self.fc_liveness = nn.Linear(64, 2)
    
    def forward(self, x1, x2):
        x = self.conv1(x1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.maxpool(x)
        # Flatten vector before FC layers
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)

        x = torch.concat((x, x2), axis=1)
        x = self.fc_hybrid1(x)
        y1 = self.fc_subject(x)
        y2 = self.fc_liveness(x)

        return y1, y2