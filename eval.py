import json
import torch 
from evaluate import load 
import pandas as pd 
from beamsearch import beam_search_decoder
from model import EncoderDecoder
from dataloader import get_dataloaders, HandwritingDataset, transform
import os 
from tqdm import tqdm 



def evaluate_model(model, val_loader, vocab, save_path, beam_width, device, original_dataset):
    bleu = load("bleu")
    chrf = load("chrf")

    model.eval()
    start_token = vocab.word2idx["<START>"]
    end_token = vocab.word2idx["<END>"]
    pad_token = vocab.word2idx["<PAD>"]
    results = []

    # Iterate over the val_loader
    with torch.no_grad():
        for batch_idx, (images, captions, lengths) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Get batch indices and image names
            batch_indices = val_loader.dataset.indices[batch_idx * val_loader.batch_size:
                                                       (batch_idx + 1) * val_loader.batch_size]
            image_names = [os.path.basename(original_dataset.data.iloc[i, 0]) for i in batch_indices]

            # Clean actual captions
            actual_captions = [
                " ".join([vocab.idx2word[idx.item()] for idx in cap 
                          if idx.item() not in {start_token, end_token, pad_token}])
                for cap in captions
            ]

            images = images.to(device)

            # Generate predictions
            generated_captions = beam_search_decoder(
                model, images, start_token, end_token, beam_width=beam_width, device=device
            )

            # Calculate BLEU and CHRF2 scores
            for img_name, actual, predicted in zip(image_names, actual_captions, generated_captions):
                bleu_score = bleu.compute(predictions=[predicted], references=[[actual]])["bleu"] * 100
                chrf_score = chrf.compute(predictions=[predicted], references=[[actual]])["score"]
                results.append({
                    "image": img_name,
                    "actual_formula": actual,
                    "predicted_formula": predicted,
                    "bleu_score": bleu_score,
                    "chrf2_score": chrf_score
                })

    # Save to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(save_path, index=False)
    print(f"Inference results saved to: {save_path}")

    # Compute average BLEU and CHRF2 scores
    avg_bleu = result_df['bleu_score'].mean()
    avg_chrf2 = result_df['chrf2_score'].mean()
    print(f"\nAverage BLEU Score: {avg_bleu:.2f}")
    print(f"Average CHRF2 Score: {avg_chrf2:.2f}")
    
    
# Load configuration
config_path="config.json"
with open(config_path, "r") as f:
    config = json.load(f)
    
vocab_file_path= config["paths"]["vocab_file_path"]
with open(vocab_file_path, "r") as f:
    vocab_data = json.load(f)

vocab=vocab_data 
# Paths and Hyperparameters
save_path = "evaluation_results.csv"
beam_width = config["hyperparameters"]["beam_width"]
device= config["device"]
# Paths
train_csv_path = config["paths"]["train_csv_path"]
train_image_dir = config["paths"]["train_image_dir"]

full_dataset = HandwritingDataset(train_csv_path, train_image_dir, vocab, transform=transform)

embedding_dim = config["hyperparameters"]["embedding_dim"]
hidden_dim = config["hyperparameters"]["hidden_dim"]

num_layers= config["hyperparameters"]["num_layers"]

# Load Model
model = EncoderDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(vocab_data), num_layers=num_layers)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
train_loader, val_loader, vocab = get_dataloaders("config.json")


# Run Evaluation
evaluate_model(
    model=model,
    val_loader=val_loader,
    vocab=vocab,
    save_path=save_path,
    beam_width=beam_width,
    device=device,
    original_dataset=full_dataset
)