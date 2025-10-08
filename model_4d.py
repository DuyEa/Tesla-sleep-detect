import torch.nn as nn

class Model4D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),  # 24x24
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),  # 12x12
            nn.Dropout(0.25),

            nn.Conv2d(64, 128, kernel_size=5, padding=2),  # 12x12
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),  # 6x6
            nn.Dropout(0.25),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),  # 6x6
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),  # 3x3
            nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 1)  # logits
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x.squeeze(1)  # (B,)