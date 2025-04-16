# q2_continue_dual_save.py - Fine-tuning script (Saving best char AND best sentence models)

from tqdm import trange, tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import logging
import multiprocessing as mp # Import multiprocessing
from positional_encoding import PositionalEncoding
from pig_latin_sentences import PigLatinSentences

# --- Global Definitions ---

# Parameters (Define constants globally)
NUM_TOKENS = 30
EMB_DIM = 100
BATCH_SIZE_FINETUNE = 64 # Renamed BATCH_SIZE for clarity
LR_FINETUNE = 1e-5 # Low constant learning rate
WEIGHT_DECAY_FINETUNE = 1e-2 # Renamed WEIGHT_DECAY
MSE_LOSS_WEIGHT = 0.0 # Keep MSE off - Used only in loss calculation formula
GRAD_CLIP_NORM_FINETUNE = 1.0 # Renamed GRAD_CLIP_NORM
NUM_EPOCHS_FINETUNE = 20 # Fine-tuning epochs
TRANSFORMER_DROPOUT = 0.1 # Should match the best performing model
LABEL_SMOOTHING_FINETUNE = 0.0 # Renamed LABEL_SMOOTHING
# Architecture constants (must match saved model)
NHEAD = 2
NUM_ENC_LAYERS = 2
NUM_DEC_LAYERS = 2
DIM_FFN = 128

# Character to integer mapping
alphabets = "abcdefghijklmnopqrstuvwxyz"
char_to_idx = {}
idx = 0
for char in alphabets: char_to_idx[char] = idx; idx += 1
char_to_idx[' '] = idx
char_to_idx['<sos>'] = idx + 1
char_to_idx['<eos>'] = idx + 2
char_to_idx['<pad>'] = idx + 3
PAD_IDX = char_to_idx['<pad>'] # Define PAD_IDX globally for collate_fn
idx_to_char = {idx: char for char, idx in char_to_idx.items()}
VOCAB_SIZE = len(char_to_idx) # Derive vocab size globally

# Device (Define globally)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Utility Functions ---
@torch.no_grad()
def decode_output(output_logits, expected_words, idx_to_char):
    # ... (implementation unchanged) ...
    out_indices = output_logits.argmax(2).detach().cpu().numpy(); expected_indices = expected_words.detach().cpu().numpy()
    out_decoded = []; exp_decoded = []
    pad_pos = char_to_idx['<pad>']; eos_pos = char_to_idx['<eos>']
    for i in range(output_logits.size(0)):
        current_out = []; current_exp = []
        for idx in out_indices[i]:
            if idx == eos_pos or idx == pad_pos: break
            current_out.append(idx_to_char.get(idx, '?'))
        out_decoded.append("".join(current_out))
        for idx in expected_indices[i]:
            if idx == char_to_idx['<sos>']: continue
            if idx == eos_pos or idx == pad_pos: break
            current_exp.append(idx_to_char.get(idx, '?'))
        exp_decoded.append("".join(current_exp))
    return out_decoded, exp_decoded

def compare_outputs(output_text, expected_text):
    # ... (implementation unchanged) ...
    sentence_correct = 0; total_chars = 0; correct_chars = 0
    for i in range(len(output_text)):
        out = output_text[i]; exp = expected_text[i]
        if "<eos>" in out: out = out.split("<eos>")[0]
        if out == exp: sentence_correct += 1
        len_out, len_exp = len(out), len(exp)
        max_common_len = min(len_out, len_exp)
        for j in range(max_common_len):
            if out[j] == exp[j]: correct_chars += 1
        total_chars += len_exp
    return sentence_correct, total_chars, correct_chars

# --- Collate Function ---
def collate_fn(batch):
    # Uses global PAD_IDX and device
    eng_indices = [item[0] for item in batch]
    pig_latin_indices = [item[1] for item in batch if item[1] is not None]
    eng_indices_padded = torch.nn.utils.rnn.pad_sequence(eng_indices, batch_first=True, padding_value=PAD_IDX)
    tgt_in, tgt_loss = None, None
    if pig_latin_indices:
        pig_latin_padded = torch.nn.utils.rnn.pad_sequence(pig_latin_indices, batch_first=True, padding_value=PAD_IDX)
        tgt_in = pig_latin_padded[:, :-1].to(device)
        tgt_loss = pig_latin_padded[:, 1:].to(device)
    return eng_indices_padded.to(device), tgt_in, tgt_loss

# --- Training Function ---
def train_one_epoch(epoch, num_epochs, loader, model, embedding, decoder, pos_enc, optimizer, criterion, params):
    # Removed mse_criterion as it's not used when weight is 0
    avg_ce_loss=0; total_batches=0; total_sent_correct=0; total_char_correct=0; total_char_count=0; total_num_samples=0
    model.train(); embedding.train(); decoder.train()
    pbar = tqdm(loader, leave=False, desc=f"Fine-Tune Epoch {epoch+1}/{num_epochs}")
    for batch_idx, (src_indices, tgt_indices_input, target_indices_loss) in enumerate(pbar):
        # --- Forward Pass ---
        input_emb = embedding(src_indices); target_emb_input = embedding(tgt_indices_input)
        input_emb_pos = pos_enc(input_emb); target_emb_input_pos = pos_enc(target_emb_input)
        src_key_padding_mask = (src_indices == PAD_IDX); tgt_key_padding_mask = (tgt_indices_input == PAD_IDX)
        tgt_len = target_emb_input_pos.size(1); tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
        output_emb = model(src=input_emb_pos, tgt=target_emb_input_pos, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output_logits = decoder(output_emb)

        # --- Loss ---
        # No need to calculate MSE loss if weight is 0
        ce_loss = criterion(output_logits.reshape(-1, NUM_TOKENS), target_indices_loss.reshape(-1));
        total_loss = ce_loss

        # --- Backward & Optimize ---
        optimizer.zero_grad(); total_loss.backward();
        torch.nn.utils.clip_grad_norm_(params, max_norm=GRAD_CLIP_NORM_FINETUNE);
        optimizer.step()

        # --- Metrics ---
        avg_ce_loss += ce_loss.item(); total_batches += 1
        pbar.set_postfix({'loss': f'{ce_loss.item():.4f}'})
        with torch.no_grad():
            output_text, expected_text = decode_output(output_logits, target_indices_loss, idx_to_char)
            sent_corr, char_count, char_corr = compare_outputs(output_text, expected_text)
            total_sent_correct += sent_corr; total_char_count += char_count; total_char_correct += char_corr; total_num_samples += len(output_text)

    # --- Logging ---
    if total_num_samples > 0:
        rand_idx = [_.item() for _ in torch.randint(0, len(output_text), (min(3, len(output_text)),))]; print("\n" + "----"*40)
        for i in rand_idx:
             out_, exp_ = output_text[i], expected_text[i]
             print(f'Train Output:   "{out_}"', flush=True)
             print(f'Train Expected: "{exp_}"', flush=True)
             print("----"*40, flush=True)

    avg_sent_acc = total_sent_correct / total_num_samples if total_num_samples > 0 else 0
    avg_char_acc = total_char_correct / total_char_count if total_char_count > 0 else 0
    return (avg_ce_loss / total_batches if total_batches > 0 else 0, avg_sent_acc, avg_char_acc)

# --- Validation Function ---
@torch.no_grad()
def validate(epoch, num_epochs, loader, model, embedding, decoder, pos_enc, criterion):
    # Removed mse_criterion as it's not used when weight is 0
    is_initial_check = (epoch == -1)
    if is_initial_check: print("--- Inside Initial Validation Check ---")
    print("Model training status:", model.training) # Should be False
    avg_ce_loss=0; total_batches=0; total_sent_correct=0; total_char_correct=0; total_char_count=0; total_num_samples=0
    model.eval(); embedding.eval(); decoder.eval()
    pbar_desc = f"Initial Val Check" if is_initial_check else f"Fine-Tune Val Epoch {epoch+1}/{num_epochs}"
    pbar = tqdm(loader, leave=False, desc=pbar_desc)
    for batch_num, ( src_indices, tgt_indices_input, target_indices_loss ) in enumerate(pbar):
        # Loss Calculation
        input_emb = embedding(src_indices); target_emb_input = embedding(tgt_indices_input)
        input_emb_pos = pos_enc(input_emb); target_emb_input_pos = pos_enc(target_emb_input)
        if is_initial_check and batch_num == 0: print(f" Initial Val: pos_enc(input_emb) norm: {torch.linalg.norm(input_emb_pos).item():.4f}\n Initial Val: pos_enc(target_emb_input) norm: {torch.linalg.norm(target_emb_input_pos).item():.4f}")
        src_key_padding_mask = (src_indices == PAD_IDX); tgt_key_padding_mask = (tgt_indices_input == PAD_IDX)
        tgt_len = target_emb_input_pos.size(1); tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
        output_emb = model(src=input_emb_pos, tgt=target_emb_input_pos, tgt_mask=tgt_mask, src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask)
        output_logits_loss = decoder(output_emb)
        # No need to calculate MSE loss if weight is 0
        ce_loss = criterion(output_logits_loss.reshape(-1, NUM_TOKENS), target_indices_loss.reshape(-1))
        avg_ce_loss += ce_loss.item(); total_batches += 1
        pbar.set_postfix({'loss': f'{ce_loss.item():.4f}'})
        # Autoregressive Generation
        batch_size_current = src_indices.size(0); max_len = src_indices.size(1) + 30
        generated_indices = torch.full((batch_size_current, 1), char_to_idx['<sos>'], dtype=torch.long, device=device)
        input_emb_pos_gen = pos_enc(embedding(src_indices)); src_key_padding_mask_gen = (src_indices == PAD_IDX)
        memory = model.encoder(input_emb_pos_gen, src_key_padding_mask=src_key_padding_mask_gen)
        if is_initial_check and batch_num == 0: print(f" Initial Val: memory (encoder output) norm: {torch.linalg.norm(memory).item():.4f}")
        finished_sequences = torch.zeros(batch_size_current, dtype=torch.bool, device=device)
        for step in range(max_len - 1):
            tgt_emb_gen = embedding(generated_indices); tgt_emb_pos = pos_enc(tgt_emb_gen)
            current_tgt_len = generated_indices.size(1); tgt_mask_gen = nn.Transformer.generate_square_subsequent_mask(current_tgt_len).to(device)
            output_step = model.decoder(tgt=tgt_emb_pos, memory=memory, tgt_mask=tgt_mask_gen, memory_key_padding_mask=src_key_padding_mask_gen)
            if is_initial_check and batch_num == 0 and step < 2: print(f"  Initial Val: Step {step}, decoder output_step norm: {torch.linalg.norm(output_step).item():.4f}")
            last_token_logits = decoder(output_step[:, -1, :]); predicted_idx = last_token_logits.argmax(-1).unsqueeze(1)
            generated_indices = torch.cat([generated_indices, predicted_idx], dim=1)
            finished_sequences |= (predicted_idx.squeeze(1) == char_to_idx['<eos>']);
            if finished_sequences.all(): break
        fake_logits_for_decode = torch.zeros(generated_indices.size(0), generated_indices.size(1), NUM_TOKENS, device=device)
        for i in range(generated_indices.size(0)):
            for j in range(generated_indices.size(1)): idx = generated_indices[i, j].item();
            if idx < NUM_TOKENS: fake_logits_for_decode[i, j, idx] = 1.0
        output_text, expected_text = decode_output(fake_logits_for_decode[:, 1:, :], target_indices_loss, idx_to_char)
        sent_corr, char_count, char_corr = compare_outputs(output_text, expected_text)
        total_sent_correct += sent_corr; total_char_count += char_count; total_char_correct += char_corr; total_num_samples += len(output_text)
    # Logging
    if total_num_samples > 0:
        rand_idx = [_.item() for _ in torch.randint(0, len(output_text), (min(3, len(output_text)),))]; print("\n" + "----"*40)
        for i in rand_idx:
             out_, exp_ = output_text[i], expected_text[i]
             print(f'Val Output:   "{out_}"', flush=True)
             print(f'Val Expected: "{exp_}"', flush=True)
             print("----"*40, flush=True)
    avg_sent_acc = total_sent_correct / total_num_samples if total_num_samples > 0 else 0
    avg_char_acc = total_char_correct / total_char_count if total_char_count > 0 else 0
    if is_initial_check: print("--- Finished Initial Validation Check ---")
    return (avg_ce_loss / total_batches if total_batches > 0 else 0, avg_sent_acc, avg_char_acc)


# --- Main Execution Guard ---
if __name__ == '__main__':
    # Set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True); print("Set multiprocessing start method to 'spawn'.")
    except RuntimeError: print("Multiprocessing context already set. Skipping set_start_method.")

    # Basic Setup
    os.makedirs('plots', exist_ok=True); os.makedirs('results', exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    print(f"Using device: {device}") # Device is defined globally

    # --- Dataset and DataLoader Setup ---
    print("Loading datasets...")
    try:
        train_dataset = PigLatinSentences("train", char_to_idx)
        val_dataset = PigLatinSentences("val", char_to_idx)
    except FileNotFoundError as e:
        print(f"Error loading dataset files: {e}.")
        exit()
    print("Datasets loaded.")
    print("Creating DataLoaders...")
    # Uses global collate_fn, BATCH_SIZE_FINETUNE
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE_FINETUNE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE_FINETUNE, shuffle=False, collate_fn=collate_fn, num_workers=4)
    print("DataLoaders created.")

    # --- Model Definition ---
    print("Initializing model structure...")
    # Use the globally defined constants (with _Q2 suffix where applicable)
    embedding = nn.Embedding(VOCAB_SIZE, EMB_DIM, padding_idx=PAD_IDX)
    model = nn.Transformer(
        d_model=EMB_DIM, nhead=NHEAD, num_encoder_layers=NUM_ENC_LAYERS,
        num_decoder_layers=NUM_DEC_LAYERS, dim_feedforward=DIM_FFN,
        dropout=TRANSFORMER_DROPOUT, batch_first=True, activation='relu'
    )
    decoder = nn.Linear(EMB_DIM, NUM_TOKENS)
    pos_enc = PositionalEncoding(EMB_DIM, dropout=TRANSFORMER_DROPOUT)
    embedding = embedding.to(device); model = model.to(device); decoder = decoder.to(device); pos_enc = pos_enc.to(device)
    print("Model structures initialized.")

    # --- Optimizer ---
    # Use fine-tuning specific parameters defined globally
    params = list(embedding.parameters()) + list(model.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR_FINETUNE, weight_decay=WEIGHT_DECAY_FINETUNE)
    print(f"Optimizer initialized with constant LR: {LR_FINETUNE}")

    # --- Loss Function ---
    # Use fine-tuning specific parameters defined globally
    ce_criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=LABEL_SMOOTHING_FINETUNE)
    mse_criterion = nn.MSELoss() # Define even if unused
    print(f"Loss function initialized with label_smoothing={LABEL_SMOOTHING_FINETUNE}")

    # --- Load Checkpoint ---
    model_load_path = "results/q2_model.pt"; start_epoch = 0; best_val_char_acc = 0.0; best_val_sentence_acc = 0.0
    if os.path.exists(model_load_path):
        print(f"Loading base checkpoint: {model_load_path}")
        try:
            checkpoint = torch.load(model_load_path, map_location=device)
            embedding.load_state_dict(checkpoint['embeddings'])
            model.load_state_dict(checkpoint['transformer'])
            decoder.load_state_dict(checkpoint['decoder'])
            print("  -> Model weights loaded successfully.")
            # --- DO NOT LOAD OPTIMIZER STATE FOR FINE-TUNING ---
            print("  -> Using NEW optimizer state for fine-tuning.")
            best_val_char_acc = checkpoint.get('best_val_char_acc', 0.0)
            best_val_sentence_acc = checkpoint.get('best_val_sentence_acc', 0.0)
            print(f"  -> Previous best validation character accuracy: {best_val_char_acc:.4f}")
            print(f"  -> Previous best validation sentence accuracy: {best_val_sentence_acc:.4f}")
        except Exception as e: print(f"Error loading checkpoint {model_load_path}: {e}. Cannot continue fine-tuning."); exit()
        # Initial Validation Check
        print("\nPerforming initial validation on loaded model before fine-tuning...")
        # Pass the correct criterion object
        initial_val_ce_loss, initial_val_sentence_acc, initial_val_char_acc = validate(epoch=-1, num_epochs=0, loader=val_loader, model=model, embedding=embedding, decoder=decoder, pos_enc=pos_enc, criterion=ce_criterion)
        print(f"Initial Validation Results: Loss={initial_val_ce_loss:.4f}, SentAcc={initial_val_sentence_acc:.4f}, CharAcc={initial_val_char_acc:.4f}")
        print("--- End Initial Validation ---\n")
    else: print(f"Error: Base checkpoint file not found at {model_load_path}. Cannot continue."); exit()

    # --- Training Loop ---
    train_loss_list, train_sentence_acc_list, train_char_acc_list = [], [], []
    val_loss_list, val_sentence_acc_list, val_char_acc_list = [], [], []
print("Starting fine-tuning (saving best char AND best sentence models)...")
    # Uses global NUM_EPOCHS_FINETUNE
    for epoch in trange(start_epoch, NUM_EPOCHS_FINETUNE, initial=start_epoch, total=NUM_EPOCHS_FINETUNE, desc="Fine-Tuning"):
        train_ce_loss, train_sentence_acc, train_char_acc = train_one_epoch(epoch, NUM_EPOCHS_FINETUNE, train_loader, model, embedding, decoder, pos_enc, optimizer, ce_criterion, params)
        val_ce_loss, val_sentence_acc, val_char_acc = validate(epoch, NUM_EPOCHS_FINETUNE, val_loader, model, embedding, decoder, pos_enc, ce_criterion)
        # Save Checkpoints
        save_dict = { 'transformer': model.state_dict(), 'decoder': decoder.state_dict(), 'embeddings': embedding.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'epoch': f"fine-tune-{epoch + 1}", 'best_val_char_acc': best_val_char_acc, 'best_val_sentence_acc': best_val_sentence_acc }
    save_char_model = False
        if val_char_acc > best_val_char_acc: print(f"Fine-Tune Epoch {epoch+1}: New best validation CHARACTER accuracy: {val_char_acc:.4f} (improved from {best_val_char_acc:.4f})"); best_val_char_acc = val_char_acc; save_dict['best_val_char_acc'] = best_val_char_acc; save_char_model = True
    save_sent_model = False
        if val_sentence_acc > best_val_sentence_acc: print(f"Fine-Tune Epoch {epoch+1}: New best validation SENTENCE accuracy: {val_sentence_acc:.4f} (improved from {best_val_sentence_acc:.4f})"); best_val_sentence_acc = val_sentence_acc; save_dict['best_val_sentence_acc'] = best_val_sentence_acc; save_sent_model = True
        if save_char_model: model_save_path_char = "results/q2_model_best_char.pt"; torch.save(save_dict, model_save_path_char); print(f"  -> Saved best char model checkpoint to {model_save_path_char}")
        if save_sent_model: model_save_path_sent = "results/q2_model_best_sent.pt"; torch.save(save_dict, model_save_path_sent); print(f"  -> Saved best sentence model checkpoint to {model_save_path_sent}")
        # Append results
        train_loss_list.append(train_ce_loss); train_sentence_acc_list.append(train_sentence_acc); train_char_acc_list.append(train_char_acc)
        val_loss_list.append(val_ce_loss); val_sentence_acc_list.append(val_sentence_acc); val_char_acc_list.append(val_char_acc)
    # Print epoch results
        print(f"Fine-Tune Epoch {epoch+1}/{NUM_EPOCHS_FINETUNE} Results (LR: {optimizer.param_groups[0]['lr']:.2e}):")
    print(f"  Train Loss: {train_ce_loss:.4f}, Acc: Sent={train_sentence_acc:.4f}, Char={train_char_acc:.4f}")
    print(f"  Val   Loss: {val_ce_loss:.4f}, Acc: Sent={val_sentence_acc:.4f}, Char={val_char_acc:.4f} (Best Sent: {best_val_sentence_acc:.4f}, Best Char: {best_val_char_acc:.4f})")

    # --- Plotting ---
print("Fine-tuning finished. Plotting fine-tuning results...")
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
epochs_range = np.arange(start_epoch, start_epoch + len(train_loss_list))
axs[0].plot(epochs_range, train_loss_list, label="Train CE Loss")
axs[0].plot(epochs_range, val_loss_list, label="Val CE Loss")
axs[0].legend(); axs[0].set_title("Fine-Tuning CE Loss"); axs[0].set_xlabel("Fine-Tuning Epoch"); axs[0].set_ylabel("Loss")
axs[1].plot(epochs_range, np.array(train_sentence_acc_list)*100, label="Train Sent Acc", linestyle='--')
axs[1].plot(epochs_range, np.array(val_sentence_acc_list)*100, label="Val Sent Acc", linewidth=2)
axs[1].plot(epochs_range, np.array(train_char_acc_list)*100, label="Train Char Acc", linestyle=':')
axs[1].plot(epochs_range, np.array(val_char_acc_list)*100, label="Val Char Acc")
axs[1].legend(); axs[1].set_title("Fine-Tuning Accuracy"); axs[1].set_xlabel("Fine-Tuning Epoch"); axs[1].set_ylabel("Accuracy (%)"); axs[1].grid(True)
fig.tight_layout()
    fig.savefig("plots/q2_finetune_dual_save_results.png", dpi=300)
plt.close()
print("Fine-tuning plots saved to plots/q2_finetune_dual_save_results.png")
print(f"Fine-tuning complete.")
print(f"  Final best validation sentence accuracy model saved: {best_val_sentence_acc:.4f}")
print(f"  Final best validation character accuracy model saved: {best_val_char_acc:.4f}")
