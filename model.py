import torch 
import torch.nn as nn 
from torchvision import models

class ResNet18Encoder(nn.Module):
    def __init__(self, embedding_dim=256):
        super(ResNet18Encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        
        # Extract all layers of ResNet18
        all_layers = list(resnet18.children())
        
        # Split layers into first 3, middle (frozen), and last 3 (excluding final classifier)
        self.first_layers = nn.Sequential(*all_layers[:3])  # First 3 layers
        self.middle_layers = nn.Sequential(*all_layers[3:-3])  # Middle layers (to freeze)
        self.last_layers = nn.Sequential(*all_layers[-3:-1])  # Last 3 layers (excluding fc layer)
        
        # Freeze middle layers
        for param in self.middle_layers.parameters():
            param.requires_grad = True 
        
        # Add a linear layer to reduce dimensions to embedding_dim
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling to 1x1
        self.fc = nn.Linear(512, embedding_dim)          # Map ResNet output to embedding_dim
        
    def forward(self, x):
        # print(f"Input shape to Encoder: {x.shape}")  # Print input shape
        x = self.first_layers(x)
        # # print(f"Shape after first layers: {x.shape}")
        x = self.middle_layers(x)
        # # print(f"Shape after frozen middle layers: {x.shape}")
        x = self.last_layers(x)
        # # print(f"Shape after last layers: {x.shape}")
        x = self.global_pool(x)
        # # print(f"Shape after global pooling: {x.shape}")
        x = x.view(x.size(0), -1)
        # # print(f"Shape after flattening: {x.shape}")
        x = self.fc(x)
        # print(f"Final output shape: {x.shape}")
        return x




# Define the BiLSTMDecoder class
class BiLSTMDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1):
        super(BiLSTMDecoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.bidirectional = True

        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # BiLSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
        )

        # Fully connected layer to map BiLSTM hidden states to vocabulary
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # Hidden size is doubled for BiLSTM

    def forward_step(self, inputs, h, c):

        embedded = self.embedding(inputs)  # Shape: (batch_size, 1, embedding_dim)
        lstm_out, (h, c) = self.lstm(embedded, (h, c))  # Shape: (batch_size, 1, hidden_dim * 2)
        logits = self.fc(lstm_out.squeeze(1))  # Shape: (batch_size, vocab_size)
        return logits, h, c

    def init_hidden(self, batch_size, encoder_output):
        """
        Initialize hidden and cell states using encoder output.
        """
        num_directions = 2 if self.bidirectional else 1
        h = encoder_output.unsqueeze(0).repeat(self.num_layers * num_directions, 1, 1)  # Hidden state
        c = torch.zeros_like(h)  # Cell state
        return h, c




class EncoderDecoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, num_layers=1, teacher_forcing_ratio=0.5):
        super(EncoderDecoder, self).__init__()
        self.encoder = ResNet18Encoder(embedding_dim)
        self.decoder = BiLSTMDecoder(embedding_dim, hidden_dim, vocab_size, num_layers)
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, images, captions, max_length=None):

        batch_size = images.size(0)
        max_length = max_length or captions.size(1)

        # Step 1: Encode the images
        encoder_output = self.encoder(images)  # Shape: (batch_size, embedding_dim)

        # Step 2: Initialize the decoder's hidden and cell states
        h, c = self.decoder.init_hidden(batch_size, encoder_output)

        # Step 3: Initialize the START token as the first input
        inputs = captions[:, 0].unsqueeze(1)  # Shape: (batch_size, 1)

        outputs = []  # To store outputs at each timestep

        # Step 4: Decoding loop
        for t in range(max_length):
            logits, h, c = self.decoder.forward_step(inputs, h, c)  # Single timestep decoding
            outputs.append(logits)  # Store output logits

            # Teacher forcing or greedy decoding
            if torch.rand(1).item() < self.teacher_forcing_ratio and t + 1 < captions.size(1):
                # Use the next token from ground truth
                inputs = captions[:, t + 1].unsqueeze(1)
            else:
                # Use the predicted token as the next input
                predicted_token = logits.argmax(dim=1)  # Shape: (batch_size,)
                inputs = predicted_token.unsqueeze(1)  # Shape: (batch_size, 1)

        # Stack outputs along the time dimension
        outputs = torch.stack(outputs, dim=1)  # Shape: (batch_size, max_length, vocab_size)

        return outputs

