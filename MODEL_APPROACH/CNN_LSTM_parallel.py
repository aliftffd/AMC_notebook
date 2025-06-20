import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_LSTM_Parallel(nn.Module):
    def __init__(self, 
                 input_channels=2,      # I/Q channels
                 sequence_length=1024,  # frame_size
                 num_classes=19,        # n_labels
                 cnn_filters=[32, 64, 128],
                 lstm_hidden_dim=64,
                 lstm_num_layers=2,
                 dropout=0.3):
        
        super(CNN_LSTM_Parallel, self).__init__()
        
        self.sequence_length = sequence_length
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_classes = num_classes
        
        # CNN Branch - Multiple kernel sizes for multi-scale features
        self.cnn_kernels = [3, 5, 7, 11]  # Different receptive fields
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, cnn_filters[0], kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters[0]),
                nn.ReLU(),
                nn.Conv1d(cnn_filters[0], cnn_filters[1], kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters[1]),
                nn.ReLU(),
                nn.Conv1d(cnn_filters[1], cnn_filters[2], kernel_size=k, padding=k//2),
                nn.BatchNorm1d(cnn_filters[2]),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)  # Global max pooling
            ) for k in self.cnn_kernels
        ])
        
        # LSTM Branch
        self.lstm = nn.LSTM(
            input_size=input_channels,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=dropout if lstm_num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True  # Better for feature extraction
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Calculate concatenated feature size
        cnn_feature_size = len(self.cnn_kernels) * cnn_filters[2]  # 4 * 128
        lstm_feature_size = lstm_hidden_dim * 2  # *2 for bidirectional
        total_features = cnn_feature_size + lstm_feature_size
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(total_features, total_features // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(total_features // 2, num_classes)
        )
    
    def forward(self, x):
        # Input shape: (batch_size, 2, 1024)
        batch_size = x.size(0)
        
        # CNN Branch - Parallel processing with different kernel sizes
        cnn_outputs = []
        for conv in self.convs:
            cnn_out = conv(x)  # (batch, filters, 1)
            cnn_out = cnn_out.squeeze(-1)  # (batch, filters)
            cnn_outputs.append(cnn_out)
        
        cnn_features = torch.cat(cnn_outputs, dim=1)  # (batch, total_cnn_features)
        cnn_features = self.dropout(cnn_features)
        
        # LSTM Branch
        # Transpose for LSTM: (batch, seq_len, features)
        lstm_input = x.transpose(1, 2)  # (batch, 1024, 2)
        
        lstm_out, (h_n, c_n) = self.lstm(lstm_input)
        
        # Use final hidden state from both directions
        # h_n shape: (num_layers * 2, batch, hidden_dim)
        lstm_features = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, hidden_dim*2)
        lstm_features = self.dropout(lstm_features)
        
        # Concatenate CNN and LSTM features
        combined_features = torch.cat([cnn_features, lstm_features], dim=1)
        
        # Classification
        output = self.classifier(combined_features)
        
        return output