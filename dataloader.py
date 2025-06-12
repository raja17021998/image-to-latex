import os
import json
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from collections import Counter
import re

# Load configuration
config_path="config.json"
with open(config_path, "r") as f:
    config = json.load(f)

def tokenize_latex(expression):
    expression = expression.strip()
    if expression.startswith('$') and expression.endswith('$'):
        expression = expression[1:-1].strip()

    pattern = (
    r'(\\[a-zA-Z]+|\\left|\\right|\\bigg|\\big|\\lbrace|\\rbrace|\\langle|\\rangle|'
    r'\{|\}|\(|\)|\[|\]|~|.|[0-9]+|\+|\-|\^|\_|[a-zA-Z])'
)

    tokens = re.findall(pattern, expression)
    tokens = [t for t in tokens if t.strip()]

    return tokens

# Step 1: Vocabulary Class
class Vocabulary:
    def __init__(self):
        self.word2idx = {"<PAD>": 0, "<START>": 1, "<END>": 2, "<UNK>": 3}
        self.idx2word = {0: "<PAD>", 1: "<START>", 2: "<END>", 3: "<UNK>"}
        self.word_count = Counter()

    def add_sentence(self, sentence):
        tokens = tokenize_latex(sentence.lower())
        for token in tokens:
            self.word_count[token] += 1

    def build_vocab(self, min_freq=1):
        idx = len(self.word2idx)
        for word, count in self.word_count.items():
            if count >= min_freq:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def sentence_to_indices(self, sentence):
        tokens = tokenize_latex(sentence.lower())
        return [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]

# Step 2: Build Vocabulary from CSV for Handwriting
def build_vocab_from_csv_hw(csv_file, min_freq=1):
    vocab = Vocabulary()
    data = pd.read_csv(csv_file)
    for caption in data['formula']:
        vocab.add_sentence(caption)

    # Add all lowercase alphabets
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        vocab.word_count[ch] += 1
    # Add all uppercase alphabets
    for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        vocab.word_count[ch] += 1

    vocab.build_vocab(min_freq=min_freq)
    
    file_path= config["paths"]["vocab_file_path"]
    with open(file_path, "w") as f:
        json.dump(vocab.word2idx, f, indent=4)
    return vocab

# Step 2: Build Vocabulary from CSV for Synthetic
def build_vocab_from_csv_syn(csv_file, min_freq=1):
    vocab = Vocabulary()
    data = pd.read_csv(csv_file)
    for caption in data['formula']:
        vocab.add_sentence(caption)

    # Add all lowercase alphabets
    for ch in 'abcdefghijklmnopqrstuvwxyz':
        vocab.word_count[ch] += 1
    # Add all uppercase alphabets
    for ch in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        vocab.word_count[ch] += 1

    vocab.build_vocab(min_freq=min_freq)
    
    vocab_w2idx_file_path_syn= config["paths"]["vocab_w2idx_syn"]
    with open(vocab_w2idx_file_path_syn, "w") as f:
        json.dump(vocab.word2idx, f, indent=4)

    vocab_idx2w_file_path_syn= config["paths"]["vocab_idx2w_syn"]
    with open(vocab_idx2w_file_path_syn, "w") as f:
        idx2w_str_keys = {str(k): v for k, v in vocab.idx2word.items()}
        json.dump(idx2w_str_keys, f, indent=4)

    return vocab

# Step 3: Dataset Classes
class HandwritingDataset(Dataset):
    def __init__(self, csv_file, image_dir, vocab, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        caption = self.data.iloc[idx, 1]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption_indices = [self.vocab.word2idx["<START>"]] + \
                          self.vocab.sentence_to_indices(caption) + \
                          [self.vocab.word2idx["<END>"]]
        return image, torch.tensor(caption_indices)

class SyntheticDataset(Dataset):
    def __init__(self, csv_file, image_dir, vocab, transform=None):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        caption = self.data.iloc[idx, 1]
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        caption_indices = [self.vocab.word2idx["<START>"]] + \
                          self.vocab.sentence_to_indices(caption) + \
                          [self.vocab.word2idx["<END>"]]
        return image, torch.tensor(caption_indices)

# Step 4: Collate Function
def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    caption_lengths = [len(cap) for cap in captions]
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions, caption_lengths

transform=0

# Step 5: Single get_dataloaders function
def get_dataloaders(config_path="config.json"):
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)

    # Paths for handwriting data
    train_csv_path_hw = config["paths"]["train_csv_path_hw"]
    train_image_dir_hw = config["paths"]["train_image_dir_hw"]

    # Paths for synthetic data
    train_csv_path_syn = config["paths"]["train_csv_path_syn"]
    train_image_dir_syn = config["paths"]["train_image_dir_syn"]
    
    # Hyperparameters
    batch_size = config["hyperparameters"]["batch_size"]
    min_freq = config["hyperparameters"]["min_freq"]

    # Image Transformations
    transform = transforms.Compose([
        transforms.Resize(tuple(config["transformations"]["resize"])),
        transforms.ToTensor(),
    ])

    # Build Vocabulary for Handwriting
    vocab_hw = build_vocab_from_csv_hw(train_csv_path_hw, min_freq=min_freq)
    vocab_w2idx_hw= vocab_hw.word2idx
    vocab_idx2w_hw= vocab_hw.idx2word
    
    # Save HW vocab
    vocab_w2idx_file_path_hw= config["paths"]["vocab_w2idx_hw"]
    with open(vocab_w2idx_file_path_hw, "w") as f:
        json.dump(vocab_w2idx_hw, f, indent=4)
        
    vocab_idx2w_file_path_hw= config["paths"]["vocab_idx2w_hw"]
    with open(vocab_idx2w_file_path_hw, "w") as f:
        idx2w_str_keys = {str(k): v for k, v in vocab_idx2w_hw.items()}
        json.dump(idx2w_str_keys, f, indent=4)

    print("Handwriting vocab created!!")

    # Create Handwriting Dataset
    full_dataset_hw = HandwritingDataset(train_csv_path_hw, train_image_dir_hw, vocab_hw, transform=transform)

    # Split Handwriting Dataset
    train_size_hw = int(0.8 * len(full_dataset_hw))
    val_size_hw = len(full_dataset_hw) - train_size_hw
    train_dataset_hw, val_dataset_hw = random_split(full_dataset_hw, [train_size_hw, val_size_hw])

    # DataLoaders for Handwriting
    train_loader_hw = DataLoader(train_dataset_hw, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader_hw = DataLoader(val_dataset_hw, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print("Handwriting DataLoaders Created!")

    # --- Synthetic Dataset Preparation ---
    vocab_syn = build_vocab_from_csv_syn(train_csv_path_syn, min_freq=min_freq)
    print("Synthetic vocab created!!")

    # Directly load train, val, and test sets for Synthetic dataset from their respective CSV files
    val_csv_path_syn = config["paths"]["val_csv_path_syn"]
    test_csv_path_syn = config["paths"]["test_csv_path_syn"]

    train_dataset_syn = SyntheticDataset(train_csv_path_syn, train_image_dir_syn, vocab_syn, transform=transform)
    val_dataset_syn = SyntheticDataset(val_csv_path_syn, train_image_dir_syn, vocab_syn, transform=transform)
    test_dataset_syn = SyntheticDataset(test_csv_path_syn, train_image_dir_syn, vocab_syn, transform=transform)

    # Synthetic DataLoaders for train, val, and test
    train_loader_syn = DataLoader(train_dataset_syn, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader_syn = DataLoader(val_dataset_syn, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader_syn = DataLoader(test_dataset_syn, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print("Synthetic DataLoaders Created!")


    # Return original three variables as requested
    return train_loader_syn, val_loader_syn, vocab_syn

# Test Code
if __name__ == "__main__":
    train_loader, val_loader, vocab = get_dataloaders("config.json")

    for images, captions, lengths in train_loader:
        print(f"Syn Train Batch: Images shape {images.shape}, Captions shape {captions.shape}")
        print("First Syn caption indices:", captions[0].tolist())
        break

    for images, captions, lengths in val_loader:
        print(f"Syn Val Batch: Images shape {images.shape}, Captions shape {captions.shape}")
        break
