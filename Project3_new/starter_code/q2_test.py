# generate_q2_test.py - Generates test predictions from a specific checkpoint

from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np
import os
import logging
from positional_encoding import PositionalEncoding
from pig_latin_sentences import PigLatinSentences

# --- Constants and Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters corresponding to the SAVED MODEL architecture
NUM_TOKENS = 30
EMB_DIM = 100
# Architecture constants (must match saved model: 2/2/128)
NHEAD = 2
NUM_ENC_LAYERS = 2
NUM_DEC_LAYERS = 2
DIM_FFN = 128
TRANSFORMER_DROPOUT = 0.1 # Dropout used when saving the model

# Fine-tuning/Loading parameters
BATCH_SIZE = 64 # Batch size for test loader

# Character mapping (must be identical to training)
alphabets = "abcdefghijklmnopqrstuvwxyz"
char_to_idx = {}
# Split assignment and loop onto separate lines
idx = 0
for char in alphabets:
    char_to_idx[char] = idx
    idx += 1
char_to_idx[' '] = idx; char_to_idx['<sos>'] = idx + 1; char_to_idx['<eos>'] = idx + 2; char_to_idx['<pad>'] = idx + 3
PAD_IDX = char_to_idx['<pad>']
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx)

# --- Utility Functions ---
@torch.no_grad()
def decode_output_test(output_indices, idx_to_char): # Simplified for generation
    out_decoded = []
    pad_pos = char_to_idx['<pad>']
    eos_pos = char_to_idx['<eos>']
    for i in range(output_indices.shape[0]):
        current_out = []
        # Skip the initial <sos> token (index 0 of dim 1)
        for idx in output_indices[i, 1:]:
            idx_item = idx.item()
            if idx_item == eos_pos or idx_item == pad_pos:
                break
            current_out.append(idx_to_char.get(idx_item, '?'))
        out_decoded.append("".join(current_out))
    return out_decoded

# --- Collate Function ---
def collate_fn_test(batch):
    # Extract the tensor from each item in the batch
    # (DataLoader might wrap single return values)
    eng_indices = [item[0] if isinstance(item, (tuple, list)) else item for item in batch]
    # Pad the extracted tensors
    eng_indices_padded = torch.nn.utils.rnn.pad_sequence(eng_indices, batch_first=True, padding_value=PAD_IDX)
    # Test loader doesn't need targets, just source indices
    return eng_indices_padded.to(device) # Only return source indices

# --- Main Execution ---
if __name__ == '__main__':
    # Ensure results dir exists
    os.makedirs('results', exist_ok=True)

    # --- Load Test Dataset ---
    print("Loading test dataset...")
    try:
        test_dataset = PigLatinSentences("test", char_to_idx)
    except FileNotFoundError as e:
         print(f"Error loading test dataset file: {e}. Ensure data files exist.")
         exit()
    print("Test dataset loaded.")

    print("Creating Test DataLoader...")
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn_test, num_workers=0) # Use num_workers=0 for simplicity here
    print("Test DataLoader created.")

    # --- Initialize Model Structure ---
    print("Initializing model structure...")
    embedding = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD_IDX)
    model = nn.Transformer(d_model=EMB_DIM, nhead=NHEAD, num_encoder_layers=NUM_ENC_LAYERS, num_decoder_layers=NUM_DEC_LAYERS, dim_feedforward=DIM_FFN, dropout=TRANSFORMER_DROPOUT, batch_first=True, activation='relu')
    decoder = nn.Linear(EMB_DIM, NUM_TOKENS)
    pos_enc = PositionalEncoding(EMB_DIM, dropout=TRANSFORMER_DROPOUT)

    embedding = embedding.to(device); model = model.to(device); decoder = decoder.to(device); pos_enc = pos_enc.to(device)
    print("Model structures initialized.")

    # --- Load Best Checkpoint --- 
    # *** IMPORTANT: Choose which checkpoint file to load ***
    # model_load_path = "results/q2_model_best_char.pt" # Best char acc from fine-tune
    # model_load_path = "results/q2_model_best_sent.pt" # Best sentence acc from fine-tune
    model_load_path = "results/q2_model.pt" # <<< Use the best model from the main q2.py run

    print(f"Loading checkpoint: {model_load_path}")
    if os.path.exists(model_load_path):
        try:
            checkpoint = torch.load(model_load_path, map_location=device)
            embedding.load_state_dict(checkpoint['embeddings'])
            model.load_state_dict(checkpoint['transformer'])
            decoder.load_state_dict(checkpoint['decoder'])
            loaded_epoch = checkpoint.get('epoch', 'unknown')
            loaded_sent_acc = checkpoint.get('best_val_sentence_acc', 'N/A')
            loaded_char_acc = checkpoint.get('best_val_char_acc', 'N/A')
            print(f"  -> Model weights loaded successfully from epoch {loaded_epoch} (SentAcc: {loaded_sent_acc}, CharAcc: {loaded_char_acc})")
        except Exception as e:
            print(f"Error loading checkpoint {model_load_path}: {e}. Cannot generate test predictions.")
            exit()
    else:
        print(f"Error: Checkpoint file not found at {model_load_path}. Cannot generate test predictions.")
        exit()

    # --- Generate Test Predictions ---
    print(f"Generating test predictions using model from {model_load_path}...")
    model.eval(); embedding.eval(); decoder.eval() # Set to evaluation mode
    test_predictions = []

    with torch.no_grad():
        for src_indices in tqdm(test_loader, desc="Test Prediction"):
            # src_indices already on device from collate_fn_test

            # --- Autoregressive Generation ---
            batch_size_current = src_indices.size(0)
            max_len_test = src_indices.size(1) + 30 # Same buffer as before

            generated_indices = torch.full((batch_size_current, 1), char_to_idx['<sos>'], dtype=torch.long, device=device)

            # Precompute encoder output
            input_emb_pos_gen = pos_enc(embedding(src_indices))
            src_key_padding_mask_gen = (src_indices == PAD_IDX)
            memory = model.encoder(input_emb_pos_gen, src_key_padding_mask=src_key_padding_mask_gen)

            finished_sequences = torch.zeros(batch_size_current, dtype=torch.bool, device=device)
            for _ in range(max_len_test - 1):
                tgt_emb_gen = embedding(generated_indices)
                tgt_emb_pos = pos_enc(tgt_emb_gen)
                current_tgt_len = generated_indices.size(1)
                tgt_mask_gen = nn.Transformer.generate_square_subsequent_mask(current_tgt_len).to(device)
                output_step = model.decoder(tgt=tgt_emb_pos, memory=memory, tgt_mask=tgt_mask_gen, memory_key_padding_mask=src_key_padding_mask_gen)
                last_token_logits = decoder(output_step[:, -1, :])
                predicted_idx = last_token_logits.argmax(-1).unsqueeze(1)
                generated_indices = torch.cat([generated_indices, predicted_idx], dim=1)
                finished_sequences |= (predicted_idx.squeeze(1) == char_to_idx['<eos>'])
                if finished_sequences.all(): break
            # --- End Autoregressive Generation ---

            # Decode generated indices
            # output_indices shape: (batch_size, generated_seq_len)
            batch_predictions = decode_output_test(generated_indices, idx_to_char)
            test_predictions.extend(batch_predictions)

    # --- Save Test Predictions ---
    test_output_path = "results/q2_test.txt" # Overwrite the existing file
    try:
        with open(test_output_path, 'w') as f:
            for line in test_predictions:
                f.write(line + '\n')
        print(f"Test predictions saved to {test_output_path} using model {model_load_path}")
    except Exception as e:
        print(f"Error writing test predictions to {test_output_path}: {e}")
