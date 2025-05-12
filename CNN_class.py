class CNN_Block(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_shape, output_shape, kernel_size=3, padding=1),
            nn.BatchNorm1d(output_shape),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        return self.net(x)


class CNN_NET(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            CNN_Block(2, 24),
            CNN_Block(24, 24),
            CNN_Block(24, 48),
            CNN_Block(48, 48),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_labels),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))
