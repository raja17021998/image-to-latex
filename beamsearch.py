import torch 
import json



# Load configuration



vocab_w2idx ={}
vocab_idx2w={}

config_path="config.json"
with open(config_path, "r") as f:
    config = json.load(f)
    
vocab_file_path= config["paths"]["vocab_file_path"]
with open(vocab_file_path, "r") as f:
    vocab_data = json.load(f)
    
vocab_w2idx_file_path= config["paths"]["vocab_w2idx"]
with open(vocab_w2idx_file_path, "r") as f:
    vocab_w2idx=json.load(f)
    
    
vocab_idx2w_file_path= config["paths"]["vocab_idx2w"]
with open(vocab_idx2w_file_path, "r") as f:
    vocab_idx2w=json.load(f)
    
    


def beam_search_decoder(model, images, start_token, end_token, beam_width=3, max_length=20, device="cuda"):
    """
    Perform beam search to generate captions for images.
    """
    batch_size = images.size(0)
    model.eval()
    images = images.to(device)

    with torch.no_grad():
        # Encode images
        encoder_output = model.encoder(images)  # (batch_size, embedding_dim)

        # Initialize hidden and cell states for the decoder
        h, c = model.decoder.init_hidden(batch_size, encoder_output)

    # Initialize beams for each sample
    beams = [[(torch.tensor([start_token], device=device), 0.0, 
               h[:, i:i+1, :].contiguous(), c[:, i:i+1, :].contiguous())] 
             for i in range(batch_size)]  # Ensure hidden states are contiguous

    final_captions = [[] for _ in range(batch_size)]

    for t in range(max_length):
        new_beams = [[] for _ in range(batch_size)]
        for i, beam_list in enumerate(beams):
            for seq, score, h, c in beam_list:
                if seq[-1].item() == end_token:
                    final_captions[i].append((seq, score))
                    continue
                inputs = seq[-1].unsqueeze(0).unsqueeze(1)  # Shape: (1, 1)
                with torch.no_grad():
                    logits, h_new, c_new = model.decoder.forward_step(inputs, h, c)
                    log_probs = torch.log_softmax(logits, dim=1)
                top_log_probs, top_tokens = log_probs.topk(beam_width, dim=1)
                for j in range(beam_width):
                    new_seq = torch.cat([seq, top_tokens[0, j].unsqueeze(0)], dim=0)
                    new_score = score + top_log_probs[0, j].item()
                    new_beams[i].append((new_seq, new_score, h_new.contiguous(), c_new.contiguous()))
        for i in range(batch_size):
            new_beams[i] = sorted(new_beams[i], key=lambda x: x[1], reverse=True)[:beam_width]
            beams[i] = new_beams[i]

    for i in range(batch_size):
        if not final_captions[i]:
            final_captions[i] = beams[i]
        final_captions[i] = sorted(final_captions[i], key=lambda x: x[1], reverse=True)[0][0]

    generated_captions = [
        " ".join([vocab_idx2w[str(idx.item())] for idx in caption if idx.item() not in {start_token, end_token}])
        for caption in final_captions
    ]
    return generated_captions