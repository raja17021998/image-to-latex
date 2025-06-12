import json
import torch
from evaluate import load
import pandas as pd
import os
from tqdm import tqdm
from model import EncoderDecoder
from dataloader import get_dataloaders, HandwritingDataset, transform, SyntheticDataset


def greedy_decode(model, images, start_token, end_token, device, max_length=100):
    model.eval()
    with torch.no_grad():
        # Encode images
        features = model.encoder(images)
        batch_size = images.size(0)
        
        # Initialize hidden states
        h, c = model.decoder.init_hidden(batch_size, features)
        
        # Initialize input with START token
        inputs = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        sampled_ids = torch.zeros(batch_size, max_length, dtype=torch.long, device=device)
        sampled_ids[:, 0] = start_token
        
        for t in range(1, max_length):
            outputs, h, c = model.decoder.forward_step(inputs, h, c)
            # Greedy: pick the token with the highest probability
            _, predicted = outputs.max(1)
            sampled_ids[:, t] = predicted
            
            # If all predicted are END tokens, we can stop early
            if (predicted == end_token).all():
                break
            
            inputs = predicted.unsqueeze(1)
        
        return sampled_ids.cpu().numpy()


####
def evaluate_model_syn(model, val_loader, vocab_w2idx, vocab_idx2w, save_path, device, max_length=100):
    from evaluate import load
    bleu = load("bleu")
    chrf = load("chrf")

    model.eval()
    start_token = vocab_w2idx["<START>"]
    end_token = vocab_w2idx["<END>"]
    pad_token = vocab_w2idx["<PAD>"]

    results = []

    with torch.no_grad():
        for batch_idx, (images, captions, lengths) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Compute the index range for this batch
            start_idx = batch_idx * val_loader.batch_size
            end_idx = start_idx + images.size(0)
            # Access image names from val_loader.dataset.data directly
            image_names = [os.path.basename(val_loader.dataset.data.iloc[i, 0]) for i in range(start_idx, end_idx)]

            # Convert actual captions from indices to strings
            actual_captions = [
                " ".join([vocab_idx2w[idx.item()] for idx in cap 
                          if idx.item() not in {start_token, end_token, pad_token}])
                for cap in captions
            ]

            images = images.to(device)

            # Generate predictions using greedy decoding
            sampled_ids = greedy_decode(model, images, start_token, end_token, device, max_length=max_length)

            # Convert sampled_ids to strings
            generated_captions = []
            for ids in sampled_ids:
                caption_tokens = []
                for token_id in ids:
                    if token_id == end_token:
                        break
                    if token_id != start_token:
                        caption_tokens.append(vocab_idx2w[token_id])
                generated_captions.append(" ".join(caption_tokens))

            # Calculate BLEU and CHRF2 scores
            for img_name, actual, predicted in zip(image_names, actual_captions, generated_captions):
                actual_tokens = actual.strip().split()
                ref_len = len(actual_tokens)

                if ref_len == 0:
                    # Print diagnostics
                    print("----- DIAGNOSTIC INFO -----")
                    print(f"Image causing error: {img_name}")
                    print(f"Actual caption: '{actual}'")
                    print(f"Predicted caption: '{predicted}'")
                    print(f"Number of tokens in actual caption: {ref_len}")
                    print("The reference caption is empty, which may cause a division by zero error in BLEU computation.")
                    print("---------------------------")
                    # Raise an error to stop execution and allow debugging
                    raise ValueError("Encountered a zero-length reference caption, cannot compute BLEU.")

                bleu_score = bleu.compute(predictions=[predicted], references=[[actual]])["bleu"] * 100
                chrf_score = chrf.compute(predictions=[predicted], references=[[actual]])["score"]

                results.append({
                    "image": img_name,
                    "actual_formula": actual,
                    "predicted_formula": predicted,
                    "bleu_score": bleu_score,
                    "chrf2_score": chrf_score
                })

    # Save results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(save_path, index=False)
    print(f"Inference results saved to: {save_path}")

    # Compute average BLEU and CHRF2 scores
    avg_bleu = result_df['bleu_score'].mean()
    avg_chrf2 = result_df['chrf2_score'].mean()
    print(f"\nAverage BLEU Score: {avg_bleu:.2f}")
    print(f"Average CHRF2 Score: {avg_chrf2:.2f}")




def evaluate_model_hw(model, val_loader, vocab_w2idx, vocab_idx2w, save_path, device, original_dataset, max_length=100):
    bleu = load("bleu")
    chrf = load("chrf")

    model.eval()
    start_token = vocab_w2idx["<START>"]
    end_token = vocab_w2idx["<END>"]
    pad_token = vocab_w2idx["<PAD>"]

    results = []

    with torch.no_grad():
        for batch_idx, (images, captions, lengths) in enumerate(tqdm(val_loader, desc="Evaluating")):
            # Get image names
            batch_indices = val_loader.dataset.indices[batch_idx * val_loader.batch_size:
                                                       (batch_idx + 1) * val_loader.batch_size]
            image_names = [os.path.basename(original_dataset.data.iloc[i, 0]) for i in batch_indices]

            # Convert actual captions from indices to strings
            # Note: Make sure captions are LongTensors of shape (batch_size, seq_length)
            actual_captions = [
                " ".join([vocab_idx2w[idx.item()] for idx in cap 
                          if idx.item() not in {start_token, end_token, pad_token}])
                for cap in captions
            ]

            images = images.to(device)

            # Generate predictions using greedy decoding
            sampled_ids = greedy_decode(model, images, start_token, end_token, device, max_length=max_length)

            # Convert sampled_ids to strings
            generated_captions = []
            for ids in sampled_ids:
                caption_tokens = []
                for token_id in ids:
                    if token_id == end_token:
                        break
                    if token_id != start_token:
                        caption_tokens.append(vocab_idx2w[token_id])
                generated_captions.append(" ".join(caption_tokens))

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

    # Save results to CSV
    result_df = pd.DataFrame(results)
    result_df.to_csv(save_path, index=False)
    print(f"Inference results saved to: {save_path}")

    # Compute average BLEU and CHRF2 scores
    avg_bleu = result_df['bleu_score'].mean()
    avg_chrf2 = result_df['chrf2_score'].mean()
    print(f"\nAverage BLEU Score: {avg_bleu:.2f}")
    print(f"Average CHRF2 Score: {avg_chrf2:.2f}")


if __name__ == "__main__":
    # Load configuration
    config_path = "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    # Load vocab data if needed
    vocab_file_path = config["paths"]["vocab_file_path"]
    with open(vocab_file_path, "r") as f:
        vocab_data = json.load(f)

    # Load word-to-index and index-to-word mappings
    vocab_w2idx_file_path = config["paths"]["vocab_w2idx_syn"]
    with open(vocab_w2idx_file_path, "r") as f:
        vocab_w2idx = json.load(f)

    vocab_idx2w_file_path = config["paths"]["vocab_idx2w_syn"]
    with open(vocab_idx2w_file_path, "r") as f:
        vocab_idx2w = json.load(f)

    # Convert keys of vocab_idx2w to int if they are strings
    vocab_idx2w = {int(k): v for k, v in vocab_idx2w.items()}
    print("@@@@@@@@@@@@@")
    print(len(vocab_idx2w), len(vocab_w2idx))

    # Paths and Hyperparameters
    save_path = "evaluation_results.csv"
    device = config["device"]
    train_csv_path = config["paths"]["train_csv_path_syn"]
    train_image_dir = config["paths"]["train_image_dir_syn"]
    embedding_dim = config["hyperparameters"]["embedding_dim"]
    hidden_dim = config["hyperparameters"]["hidden_dim"]
    num_layers = config["hyperparameters"]["num_layers"]
    max_caption_length= config["hyperparameters"]["max_caption_length"]
    min_freq= config["hyperparameters"]["min_freq"]
    
    # Directly load train, val, and test sets for Synthetic dataset from their respective CSV files
    val_csv_path = config["paths"]["val_csv_path_syn"]
    test_csv_path = config["paths"]["test_csv_path_syn"]
    # vocab_syn = build_vocab_from_csv_syn(train_csv_path_syn, min_freq=min_freq)
    
    
    # Create dataset and dataloaders
    
    # train_dataset_syn = SyntheticDataset(train_csv_path, train_image_dir, vocab_w2idx, transform=transform)
    # val_dataset_syn = SyntheticDataset(val_csv_path, train_image_dir, vocab_w2idx, transform=transform)
    # test_dataset_syn = SyntheticDataset(test_csv_path, train_image_dir, vocab_w2idx, transform=transform)

    
    # full_dataset = HandwritingDataset(train_csv_path, train_image_dir, vocab_data, transform=transform)
    train_loader, val_loader, vocab_data = get_dataloaders("config.json")  # This might overwrite vocab_data if your code does that, be aware.

    # Load Model
    model = EncoderDecoder(embedding_dim=embedding_dim, hidden_dim=hidden_dim, vocab_size=len(vocab_idx2w), num_layers=num_layers)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)

    # Run Evaluation with Greedy Decoding
    evaluate_model_syn(
        model=model,
        val_loader=val_loader,
        vocab_w2idx=vocab_w2idx,
        vocab_idx2w=vocab_idx2w,
        save_path=save_path,
        device=device,
        max_length=max_caption_length
    )
    

