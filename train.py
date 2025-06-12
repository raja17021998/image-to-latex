import torch 
import torch.nn as nn 
import torch.optim as optim
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from dataloader import build_vocab_from_csv_syn, get_dataloaders
import json 
from model import EncoderDecoder


# Define Training Function
def train_model(model, train_loader, val_loader, vocab_dict, num_epochs=10, learning_rate=1e-3, device="cuda", save_path="best_model.pth"):
    # Set the device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=vocab_dict["<PAD>"])  # Ignore padding token in loss calculation
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize lists for plotting
    train_losses = []
    val_losses = []

    # Initialize best validation loss to a large number
    best_val_loss = float('inf')

    # Start Training Loop
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        # Train over each batch
        for images, captions, lengths in tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{num_epochs}"):
            images = images.to(device)  # Move images to device
            captions = captions.to(device)  # Move captions to device

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, captions, max_length=captions.size(1))  # (batch_size, max_length, vocab_size)

            # Prepare for loss: Shift predictions and captions
            outputs = outputs[:, :-1, :].contiguous().view(-1, len(vocab_dict))  # Predicted logits
            targets = captions[:, 1:].contiguous().view(-1)  # Ground truth tokens (shifted by one)

            # Compute Loss
            loss = criterion(outputs, targets)
            loss.backward()  # Backpropagation
            optimizer.step()

            epoch_train_loss += loss.item()

        # Average train loss
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation Loop
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for images, captions, lengths in tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{num_epochs}"):
                images = images.to(device)
                captions = captions.to(device)

                # Forward pass
                outputs = model(images, captions, max_length=captions.size(1))

                # Prepare for loss: Shift predictions and captions
                outputs = outputs[:, :-1, :].contiguous().view(-1, len(vocab_dict))
                targets = captions[:, 1:].contiguous().view(-1)

                # Compute Loss
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Check if this is the best validation loss so far
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

    # Plot Training and Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.savefig("train_val_loss.jpg", format="jpg")
    print("Loss plot saved as train_val_loss.jpg")

    # Close the plot
    plt.close()

    return model

# Load configuration
config_path="config.json"
with open(config_path, "r") as f:
    config = json.load(f)
    
# vocab_file_path= config["paths"]["vocab_file_path"]
# with open(vocab_file_path, "r") as f:
#     vocab_data = json.load(f)
    
    
# Load word-to-index and index-to-word mappings
vocab_w2idx_file_path = config["paths"]["vocab_w2idx_syn"]
with open(vocab_w2idx_file_path, "r") as f:
    vocab_w2idx = json.load(f)

vocab_idx2w_file_path = config["paths"]["vocab_idx2w_syn"]
with open(vocab_idx2w_file_path, "r") as f:
    vocab_idx2w = json.load(f)
    
# Paths
train_csv_path = config["paths"]["train_csv_path_syn"]
train_image_dir = config["paths"]["train_image_dir_syn"]

# Hyperparameters
NUM_EPOCHS = config["hyperparameters"]["num_epochs"]
LEARNING_RATE = config["hyperparameters"]["learning_rate"]



# Instantiate Model
embedding_dim = config["hyperparameters"]["embedding_dim"]
hidden_dim = config["hyperparameters"]["hidden_dim"]
vocab_dict= vocab_w2idx
vocab_size = len(vocab_dict)  # Vocabulary size
device = config["device"] if torch.cuda.is_available() else "cpu"
num_layers= config["hyperparameters"]["num_layers"]
teacher_forcing_ratio= config["hyperparameters"]["teacher_forcing_ratio"]
NUM_EPOCHS= config["hyperparameters"]["num_epochs"]
LEARNING_RATE= config["hyperparameters"]["learning_rate"]


model = EncoderDecoder(embedding_dim, hidden_dim, vocab_size, num_layers=num_layers, teacher_forcing_ratio=teacher_forcing_ratio)


train_loader, val_loader, vocab = get_dataloaders("config.json")
# Train Model
trained_model = train_model(model, train_loader, val_loader, vocab_dict, num_epochs=NUM_EPOCHS, learning_rate=LEARNING_RATE, device=device)
