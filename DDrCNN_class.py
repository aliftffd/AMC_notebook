class DDrCNN(nn.Module):
    """
    Modified DDrCNN with architecture similar to the image model
    - Uses Conv1D blocks with BatchNorm and MaxPool
    - Matches channel progression from the image (24→24→48→48)
    - Maintains dropout rate of 0.25 from the image model
    """

    def __init__(self, input_shape=(2, 1024), num_classes=24):
        super(DDrCNN, self).__init__()

        # Convert to 1D architecture like the image model
        self.backbone = nn.Sequential(
            self._make_block(2, 24),
            self._make_block(24, 24),
            self._make_block(24, 48),
            self._make_block(48, 48),
        )

        # Calculate flattened dimension
        with torch.no_grad():
            dummy = torch.zeros(1, *input_shape)
            dummy = self.backbone(dummy)
            flat_dim = dummy.view(1, -1).size(1)

        # Classifier matching the image model
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def _make_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(0.25),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.classifier(x)
