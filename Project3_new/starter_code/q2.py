from tqdm import trange, tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os # Added os import
from positional_encoding import PositionalEncoding
from pig_latin_sentences import PigLatinSentences

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") # Added print statement

# Parameters
num_tokens = 30
emb_dim = 100
batch_size = 64 # Example value
lr = 1e-4 # Initial learning rate (Max LR for warm-up/decay)
weight_decay = 1e-2 # Weight decay for AdamW
mse_loss_weight = 0.0 # Weight for MSE loss term (Set to 0.0 to remove)
grad_clip_norm = 1.0 # Max norm for gradient clipping
num_epochs = 70 # Reset to lower epoch count based on Piazza
warmup_steps = 2000 # Steps for linear warm-up
transformer_dropout = 0.1 # Dropout for the transformer model itself
label_smoothing = 0.0 # <<< Disable label smoothing for fine-tuning accuracy

# Character to integer mapping
alphabets = "abcdefghijklmnopqrstuvwxyz"
char_to_idx = {}
idx = 0
for char in alphabets:
    char_to_idx[char] = idx
    idx += 1
char_to_idx[' '] = idx
char_to_idx['<sos>'] = idx + 1
char_to_idx['<eos>'] = idx + 2
char_to_idx['<pad>'] = idx + 3
pad_idx = char_to_idx['<pad>']

# reverse, integer to character mapping
idx_to_char = {}
for char, idx in char_to_idx.items():
    idx_to_char[idx] = char

@torch.no_grad()
def decode_output(output_logits, expected_words, idx_to_char):
    # Note: Adjusted logic slightly to handle batch dimension correctly
    # Assumes batch_first=True for tensors
    out_indices = output_logits.argmax(2).detach().cpu().numpy()
    expected_indices = expected_words.detach().cpu().numpy()
    out_decoded = []
    exp_decoded = []
    pad_pos = char_to_idx['<pad>']
    eos_pos = char_to_idx['<eos>']

    for i in range(output_logits.size(0)):
        # Decode output, stop at <eos> or <pad>
        current_out = []
        for idx in out_indices[i]:
            if idx == eos_pos or idx == pad_pos:
                break
            current_out.append(idx_to_char[idx])
        out_decoded.append("".join(current_out))

        # Decode expected, stop at <eos> or <pad>
        current_exp = []
        for idx in expected_indices[i]:
            # Skip <sos> for comparison
            if idx == char_to_idx['<sos>']:
                continue
            if idx == eos_pos or idx == pad_pos:
                break
            current_exp.append(idx_to_char[idx])
        exp_decoded.append("".join(current_exp))

    return out_decoded, exp_decoded

train_dataset = PigLatinSentences("train", char_to_idx)
val_dataset = PigLatinSentences("val", char_to_idx)
test_dataset = PigLatinSentences("test", char_to_idx)

# Define your embedding
embedding = nn.Embedding(num_tokens, emb_dim, padding_idx=pad_idx)
embedding = embedding.to(device)

# Write your collate_fn
def collate_fn(batch):
    # Separate English and Pig Latin, filter out items where pig_latin is None if necessary
    eng_indices = [item[0] for item in batch]
    pig_latin_indices = [item[1] for item in batch if item[1] is not None]

    # Pad English index sequences (always present)
    eng_indices_padded = torch.nn.utils.rnn.pad_sequence(eng_indices, batch_first=True, padding_value=pad_idx)

    # Pad Pig Latin sequences ONLY if they exist (i.e., not the test set)
    if pig_latin_indices:
        pig_latin_indices_padded = torch.nn.utils.rnn.pad_sequence(pig_latin_indices, batch_first=True, padding_value=pad_idx)
        # Target for decoder input: Use indices <sos>...<eos> (exclude last)
        target_sequence_indices_input = pig_latin_indices_padded[:, :-1]
        # Target for loss: Use indices ...<eos> (exclude <sos>)
        target_sequence_indices_loss = pig_latin_indices_padded[:, 1:]
    else:
        # If no Pig Latin sequences (e.g., test set), return None for target parts
        target_sequence_indices_input = None
        target_sequence_indices_loss = None

    # Return padded indices
    return (
        eng_indices_padded.to(device), # Source indices for encoder input
        target_sequence_indices_input.to(device) if target_sequence_indices_input is not None else None,
        target_sequence_indices_loss.to(device) if target_sequence_indices_loss is not None else None
    )

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=collate_fn)

# Create your Transformer model (Reverted to original specs)
model = nn.Transformer(
    d_model=emb_dim,
    nhead=2, # Original spec
    num_encoder_layers=2, # Original spec
    num_decoder_layers=2, # Original spec
    dim_feedforward=128, # Original spec
    dropout=transformer_dropout, # Use defined dropout
    batch_first=True, # KEEPING True, fixed PositionalEncoding
    activation='relu'
)
model = model.to(device)

# Create your decoder from embedding space to the vocabulary space
decoder = nn.Linear(emb_dim, num_tokens)
decoder = decoder.to(device)

# Your positional encoder (Initialize with appropriate dropout)
pos_enc = PositionalEncoding(emb_dim, dropout=transformer_dropout) # Pass dropout
pos_enc = pos_enc.to(device) # Explicitly move pos_enc to the correct device

# Get all parameters to optimize and create your optimizer
params = list(embedding.parameters()) + list(model.parameters()) + list(decoder.parameters())
# Use AdamW optimizer
optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

# Set up your loss functions
mse_criterion = nn.MSELoss()
# Add label smoothing to CrossEntropyLoss
ce_criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=label_smoothing)

# Store your intermediate results for plotting
epoch_list = [] # Not actually used, can be removed later
train_mse_loss_list = []
train_ce_loss_list = []
train_acc_list = []
val_mse_loss_list = []
val_ce_loss_list = []
val_acc_list = []
val_char_acc_list = [] # Added list for char accuracy
best_val_char_acc = 0.0 # Initialize, will be loaded if checkpoint exists
start_epoch = 0 # Always start epoch count from 0 for this setup

# --- Load Checkpoint If Exists (Warm Start) ---
model_load_path = "results/q2_model.pt"
if os.path.exists(model_load_path):
    print(f"\nLoading checkpoint for warm start: {model_load_path}")
    try:
        # Load the checkpoint onto the correct device
        checkpoint = torch.load(model_load_path, map_location=device)

        # Load model weights
        embedding.load_state_dict(checkpoint['embeddings'])
        model.load_state_dict(checkpoint['transformer'])
        decoder.load_state_dict(checkpoint['decoder'])
        print("  -> Model weights loaded successfully.")

        # Load optimizer state IF AVAILABLE (for momentum)
        if 'optimizer_state_dict' in checkpoint:
             optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
             # NOTE: We DO NOT reset the LR here, the schedule in train_one_epoch will override it based on step count starting from 0
             print("  -> Optimizer state loaded.")
        else:
             print("  -> Optimizer state not found in checkpoint. Using new optimizer state.")

        # Load the previous best accuracy
        best_val_char_acc = checkpoint.get('best_val_char_acc', 0.0)
        loaded_epoch_info = checkpoint.get('epoch', 'unknown') # Get epoch info just for logging
        print(f"  -> Loaded weights from model saved at epoch {loaded_epoch_info}.")
        print(f"  -> Previous best validation character accuracy: {best_val_char_acc:.4f}")
        print(f"  -> Starting new training run from Epoch 1/{num_epochs} (LR schedule restarts).")

    except Exception as e:
        print(f"Error loading checkpoint {model_load_path}: {e}. Starting training from scratch.")
        # Reset best_val_char_acc if loading failed
        best_val_char_acc = 0.0
else:
    print("No checkpoint found, starting training from scratch.")
    best_val_char_acc = 0.0
# --- End Load Checkpoint ---

def compare_outputs(output_text, expected_text):
    sentence_correct = 0
    total_chars = 0
    correct_chars = 0

    for i in range(len(output_text)):
        out = output_text[i]
        exp = expected_text[i]
        # remove <eos> from generated output if present
        if "<eos>" in out:
            out = out.split("<eos>")[0]
        # Expected text already has <sos> and <eos> removed by decode_output

        # Sentence-level accuracy
        if out == exp:
            sentence_correct += 1

        # Character-level accuracy
        len_out = len(out)
        len_exp = len(exp)
        max_common_len = min(len_out, len_exp)
        
        # Count correct characters up to the common length
        for j in range(max_common_len):
            if out[j] == exp[j]:
                correct_chars += 1
        
        # Total characters considered is the length of the expected string
        # (as we measure how well we reconstructed the target)
        total_chars += len_exp 
    
    # Return counts for sentences and characters
    return sentence_correct, total_chars, correct_chars
        

def train_one_epoch(epoch, num_epochs, train_loader, model, embedding, decoder, pos_enc, optimizer, ce_criterion, mse_criterion, params, grad_clip_norm, lr, warmup_steps):
    avg_mse_loss = 0
    avg_ce_loss = 0
    total_batches = 0
    total_sentence_correct = 0
    total_char_correct = 0
    total_char_count = 0
    total_num_samples = 0

    model.train()
    # Calculate total steps for the LR scheduler
    total_steps = len(train_loader) * num_epochs

    for batch_idx, (
        src_indices, # Source indices for encoder input
        tgt_indices_input, # Target indices for decoder input (shifted)
        target_indices_loss # Target indices for loss calculation
    ) in enumerate(tqdm(train_loader, leave=False, desc=f"Train epoch {epoch+1}/{num_epochs}")):

        # --- Calculate current step and update LR --- 
        current_step = epoch * len(train_loader) + batch_idx + 1
        if current_step < warmup_steps:
            # Linear warm-up
            new_lr = lr * (current_step / warmup_steps)
        else:
            # Linear decay
            new_lr = lr * (1.0 - (current_step - warmup_steps) / (total_steps - warmup_steps))
            new_lr = max(new_lr, 0.0) # Ensure LR doesn't go below 0
        
        # Apply the new learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        # --- End LR Update ---

        # --- Embedding Lookup ---
        input_emb = embedding(src_indices) # Embed source indices
        target_emb_input = embedding(tgt_indices_input) # Embed target input indices
        target_emb_loss = embedding(target_indices_loss) # Embed target loss indices for MSE

        # 2. Pass them through the positional encodings.
        input_emb_pos = pos_enc(input_emb)
        target_emb_input_pos = pos_enc(target_emb_input)

        # 3. Create the src_mask and tgt_mask based on indices.
        src_key_padding_mask = (src_indices == pad_idx)
        tgt_key_padding_mask = (tgt_indices_input == pad_idx)
        tgt_len = target_emb_input_pos.size(1) # Target sequence length
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

        # 4. Pass the pos-encoded embeddings through the model.
        output_emb = model(
            src=input_emb_pos,
            tgt=target_emb_input_pos,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        # 5. Pass the output embeddings through the decoder.
        output_logits = decoder(output_emb)

        # 6. Calculate the MSE loss between the output embeddings and the
        # target embeddings for loss (target_emb_loss has no positional encoding).
        mse_mask = (target_indices_loss != pad_idx).unsqueeze(-1).expand_as(target_emb_loss)
        # Ensure MSE calculation uses embeddings without positional encoding
        mse_loss = mse_criterion(output_emb * mse_mask, target_emb_loss * mse_mask)

        # 7. Calculate the CE loss between the output logits and the target loss indices.
        ce_loss = ce_criterion(
            output_logits.reshape(-1, num_tokens), # Shape: (Batch*SeqLen, NumTokens)
            target_indices_loss.reshape(-1) # Shape: (Batch*SeqLen)
        )

        # 8. Add the MSE and CE losses (with weighting) and backpropagate.
        total_loss = ce_loss + mse_loss_weight * mse_loss # Weight the MSE loss
        optimizer.zero_grad()
        total_loss.backward()

        # Add Gradient Clipping
        torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip_norm)

        # 9. Update the parameters.
        optimizer.step()

        # Accumulate losses
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total_batches += 1

        # Calculate accuracy (use target_indices_loss for expected words)
        with torch.no_grad():
            output_text, expected_text = decode_output(output_logits, target_indices_loss, idx_to_char)
            sentence_correct, char_count, char_correct = compare_outputs(output_text, expected_text)
            total_sentence_correct += sentence_correct
            total_char_count += char_count
            total_char_correct += char_correct
            total_num_samples += len(output_text)

    # display the decoded outputs only for the last step of each epoch
    if total_num_samples > 0:
        rand_idx = [_.item() for _ in torch.randint(0, len(output_text),
                                                          (min(3, len(output_text)),))]
        print("\n" + "----"*40)
        for i in rand_idx:
            out_ = output_text[i]
            exp_ = expected_text[i]
            print(f"Train Output:   \"{out_}\"", flush=True)
            print(f"Train Expected: \"{exp_}\"", flush=True)
            print("----"*40, flush=True)

    avg_sentence_acc = total_sentence_correct / total_num_samples if total_num_samples > 0 else 0
    avg_char_acc = total_char_correct / total_char_count if total_char_count > 0 else 0

    return (
        avg_mse_loss / total_batches if total_batches > 0 else 0,
        avg_ce_loss / total_batches if total_batches > 0 else 0,
        avg_sentence_acc, 
        avg_char_acc
    )

@torch.no_grad()
def validate(epoch, num_epochs, val_loader, model, embedding, decoder, pos_enc, ce_criterion, mse_criterion):
    avg_mse_loss = 0
    avg_ce_loss = 0
    total_batches = 0
    total_sentence_correct = 0
    total_char_correct = 0
    total_char_count = 0
    total_num_samples = 0

    model.eval()
    for (
        src_indices, # Source indices for encoder input
        tgt_indices_input, # Target indices for decoder input (shifted)
        target_indices_loss # Target indices for loss calculation
    ) in tqdm(val_loader, leave=False, desc=f"Val epoch {epoch+1}/{num_epochs}"):

        # --- Embedding Lookup ---
        input_emb = embedding(src_indices)
        target_emb_input = embedding(tgt_indices_input)
        target_emb_loss = embedding(target_indices_loss) # For MSE loss calc

        # --- Loss Calculation (using teacher forcing like in training) ---
        input_emb_pos = pos_enc(input_emb)
        target_emb_input_pos = pos_enc(target_emb_input)
        src_key_padding_mask = (src_indices == pad_idx)
        tgt_key_padding_mask = (tgt_indices_input == pad_idx)
        tgt_len = target_emb_input_pos.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)

        output_emb = model(
            src=input_emb_pos,
            tgt=target_emb_input_pos,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        output_logits_loss = decoder(output_emb)

        mse_mask = (target_indices_loss != pad_idx).unsqueeze(-1).expand_as(target_emb_loss)
        # Use target_emb_loss (no pos enc) for MSE target
        mse_loss = mse_criterion(output_emb * mse_mask, target_emb_loss * mse_mask)
        ce_loss = ce_criterion(
            output_logits_loss.reshape(-1, num_tokens),
            target_indices_loss.reshape(-1)
        )
        avg_mse_loss += mse_loss.item()
        avg_ce_loss += ce_loss.item()
        total_batches += 1
        # --- End Loss Calculation ---

        # --- Autoregressive Generation (for Accuracy) ---
        batch_size_current = src_indices.size(0) # Use src_indices for batch size
        # Increase generation length buffer slightly more
        max_len = src_indices.size(1) + 30 # Max length based on source + buffer

        # Start with <sos> token index
        generated_indices = torch.full((batch_size_current, 1),
                                       char_to_idx['<sos>'],
                                       dtype=torch.long, device=device)

        # Pre-calculate encoder output to avoid recomputing it in the loop
        # Need pos encoded input embeddings here
        input_emb_pos_gen = pos_enc(embedding(src_indices))
        src_key_padding_mask_gen = (src_indices == pad_idx)
        memory = model.encoder(input_emb_pos_gen, src_key_padding_mask=src_key_padding_mask_gen)

        # Flag to track sequences that finished with <eos>
        finished_sequences = torch.zeros(batch_size_current, dtype=torch.bool, device=device)

        for _ in range(max_len - 1): # Max generation length
            # Embed the currently generated sequence and apply pos encoding
            tgt_emb_pos = pos_enc(embedding(generated_indices))
            current_tgt_len = generated_indices.size(1)
            tgt_mask_gen = nn.Transformer.generate_square_subsequent_mask(current_tgt_len).to(device)
            # No target padding mask needed here as we build the sequence

            # Use pre-computed memory from encoder
            output_step = model.decoder(
                tgt=tgt_emb_pos,
                memory=memory,
                tgt_mask=tgt_mask_gen,
                memory_key_padding_mask=src_key_padding_mask_gen # Use mask from encoder pass
            )

            # Get logits for the last token only
            last_token_logits = decoder(output_step[:, -1, :])
            predicted_idx = last_token_logits.argmax(-1).unsqueeze(1)

            # Append predicted token index
            generated_indices = torch.cat([generated_indices, predicted_idx], dim=1)

            # Update finished sequences mask
            finished_sequences |= (predicted_idx.squeeze(1) == char_to_idx['<eos>'])

            # Stop if all sequences have generated <eos>
            if finished_sequences.all():
                break

        # Decode the generated indices (excluding the initial <sos>)
        # Need to create fake logits tensor for decode_output structure
        # We only care about the first argument (generated indices) here
        fake_logits_for_decode = torch.zeros(generated_indices.size(0), generated_indices.size(1), num_tokens, device=device)
        for i in range(generated_indices.size(0)):
            for j in range(generated_indices.size(1)):
                idx = generated_indices[i, j].item()
                if idx < num_tokens: # Avoid index out of bounds if something went wrong
                  fake_logits_for_decode[i, j, idx] = 1.0

        # Pass generated indices (via fake_logits) and target_indices_loss
        output_text, expected_text = decode_output(fake_logits_for_decode[:, 1:, :], target_indices_loss, idx_to_char)
        sentence_correct, char_count, char_correct = compare_outputs(output_text, expected_text)
        total_sentence_correct += sentence_correct
        total_char_count += char_count
        total_char_correct += char_correct
        total_num_samples += len(output_text)
        # --- End Autoregressive Generation ---

    # display the decoded outputs only for the last step of each epoch
    if total_num_samples > 0:
        rand_idx = [_.item() for _ in torch.randint(0, len(output_text),
                                                          (min(3, len(output_text)),))]
        print("\n" + "----"*40)
        for i in rand_idx:
            out_ = output_text[i]
            exp_ = expected_text[i]
            print(f"Val Output:   \"{out_}\"", flush=True)
            print(f"Val Expected: \"{exp_}\"", flush=True)
            print("----"*40, flush=True)

    avg_sentence_acc = total_sentence_correct / total_num_samples if total_num_samples > 0 else 0
    avg_char_acc = total_char_correct / total_char_count if total_char_count > 0 else 0

    return (
        avg_mse_loss / total_batches if total_batches > 0 else 0,
        avg_ce_loss / total_batches if total_batches > 0 else 0,
        avg_sentence_acc,
        avg_char_acc
    )

# --- Training Loop ---
# Lists for THIS run's results (cleared before loop)
train_mse_loss_list = []
train_ce_loss_list = []
train_acc_list = []
val_mse_loss_list = []
val_ce_loss_list = []
val_acc_list = []
val_char_acc_list = []

print("Starting training...")
for epoch in trange(start_epoch, num_epochs, initial=start_epoch, total=num_epochs, desc="Overall Training"):
    train_mse_loss, train_ce_loss, train_sentence_acc, train_char_acc = train_one_epoch(
        epoch, num_epochs, train_loader, model, embedding, decoder, pos_enc, optimizer, ce_criterion, mse_criterion, params, grad_clip_norm, lr, warmup_steps # Pass necessary args
    )
    val_mse_loss, val_ce_loss, val_sentence_acc, val_char_acc = validate(
        epoch, num_epochs, val_loader, model, embedding, decoder, pos_enc, ce_criterion, mse_criterion # Pass necessary args
    )

    # Update best validation accuracy based on CHARACTER accuracy and save best model
    if val_char_acc > best_val_char_acc:
        print(f"Epoch {epoch+1}: New best validation character accuracy: {val_char_acc:.4f} (improved from {best_val_char_acc:.4f})")
        best_val_char_acc = val_char_acc
        # --- Save the Best Model Checkpoint (Updated Format) --- 
        os.makedirs('results', exist_ok=True)
        model_save_path = "results/q2_model.pt"
        save_dict = {
            'transformer': model.state_dict(), # Use 'transformer' key
            'decoder': decoder.state_dict(),
            'embeddings': embedding.state_dict(), # Use 'embeddings' key
            # Save optimizer state for momentum
            'optimizer_state_dict': optimizer.state_dict(),
            # 'epoch': epoch + 1, # Keep epoch commented to force restart from 0
            'best_val_char_acc': best_val_char_acc # <<< UNCOMMENT THIS
        }
        torch.save(save_dict, model_save_path)
        print(f"  -> Saved best model checkpoint to {model_save_path}")
        # --- End Save Best Model --- 

    train_mse_loss_list.append(train_mse_loss)
    train_ce_loss_list.append(train_ce_loss)
    train_acc_list.append(train_sentence_acc)
    val_mse_loss_list.append(val_mse_loss)
    val_ce_loss_list.append(val_ce_loss)
    val_acc_list.append(val_sentence_acc)
    val_char_acc_list.append(val_char_acc) # Store val char acc

    # Print epoch results including best validation CHAR accuracy and current LR
    print(f"Epoch {epoch+1}/{num_epochs} Results (LR: {optimizer.param_groups[0]['lr']:.2e}):")
    # Only print CE loss if MSE weight is 0
    train_loss_str = f"{(train_mse_loss + train_ce_loss):.4f}" if mse_loss_weight > 0 else f"{train_ce_loss:.4f}"
    val_loss_str = f"{(val_mse_loss + val_ce_loss):.4f}" if mse_loss_weight > 0 else f"{val_ce_loss:.4f}"
    print(f"  Train Loss: {train_loss_str}, Acc: Sent={train_sentence_acc:.4f}, Char={train_char_acc:.4f}")
    print(f"  Val   Loss: {val_loss_str}, Acc: Sent={val_sentence_acc:.4f}, Char={val_char_acc:.4f} (Best Char: {best_val_char_acc:.4f})")

# Ensure plots directory exists
os.makedirs('plots', exist_ok=True)

train_mse_loss_list = np.array(train_mse_loss_list)
train_ce_loss_list = np.array(train_ce_loss_list)
train_acc_list = np.array(train_acc_list)*100
val_mse_loss_list = np.array(val_mse_loss_list)
val_ce_loss_list = np.array(val_ce_loss_list)
val_acc_list = np.array(val_acc_list)*100
val_char_acc_list = np.array(val_char_acc_list)*100

fig, axs = plt.subplots(2, 2, figsize=(10, 10))

axs[0, 0].plot(np.arange(num_epochs), train_ce_loss_list + train_mse_loss_list * mse_loss_weight, label="Train")
axs[0, 0].plot(np.arange(num_epochs), val_ce_loss_list + val_mse_loss_list * mse_loss_weight, label="Val")
axs[0, 0].legend()
axs[0, 0].set_title("Total Weighted Loss")
axs[0, 0].set_xlabel("Epoch")
axs[0, 0].set_ylabel("Loss")
axs[0, 0].set_yscale("log")

axs[0, 1].plot(np.arange(num_epochs), train_acc_list, label="Train")
axs[0, 1].plot(np.arange(num_epochs), val_acc_list, label="Val Sentence Acc")
axs[0, 1].plot(np.arange(len(val_char_acc_list)), val_char_acc_list, label="Val Char Acc")
axs[0, 1].legend()
axs[0, 1].set_title("Accuracy")
axs[0, 1].set_xlabel("Epoch")
axs[0, 1].set_ylabel("Accuracy (%)")

axs[1, 0].plot(np.arange(num_epochs), train_mse_loss_list, label="Train")
axs[1, 0].plot(np.arange(num_epochs), val_mse_loss_list, label="Val")
axs[1, 0].legend()
axs[1, 0].set_title("MSE Loss")
axs[1, 0].set_xlabel("Epoch")
axs[1, 0].set_ylabel("Loss")
axs[1, 0].set_yscale("log")

axs[1, 1].plot(np.arange(num_epochs), train_ce_loss_list, label="Train")
axs[1, 1].plot(np.arange(num_epochs), val_ce_loss_list, label="Val")
axs[1, 1].legend()
axs[1, 1].set_title("CE Loss")
axs[1, 1].set_xlabel("Epoch")
axs[1, 1].set_ylabel("Loss")
axs[1, 1].set_yscale("log")

fig.tight_layout()
fig.savefig("plots/q2_results.png", dpi=300)
plt.close()

print("Final sentence accuracy")
print(f"Train: {train_acc_list[-1]:.4f}")
# print(f"Val: {val_acc_list[-1]:.4f}")
print(f"Best Val Char Acc during training: {best_val_char_acc:.4f}") # Report best char acc
print("Final losses (last epoch)")
if mse_loss_weight > 0:
    print(f"  Train MSE: {train_mse_loss_list[-1]:.4f}") # Use 4 decimal places
    print(f"  Val MSE: {val_mse_loss_list[-1]:.4f}") # Clarify this is last epoch
print(f"  Train CE: {train_ce_loss_list[-1]:.4f}") # Use 4 decimal places
print(f"  Val CE: {val_ce_loss_list[-1]:.4f}") # Clarify this is last epoch

# --- Generate Test Predictions --- 
print("\nLoading best model (based on val char acc) for test prediction...")
model_load_path = "results/q2_model.pt"
best_model_val_acc_loaded = 0.0 # Keep track for printing
if os.path.exists(model_load_path):
    try:
        # Load checkpoint with the new structure
        checkpoint = torch.load(model_load_path, map_location=device)
        model.load_state_dict(checkpoint['transformer'])
        decoder.load_state_dict(checkpoint['decoder'])
        embedding.load_state_dict(checkpoint['embeddings'])
        # Optional: Load epoch/acc if they were saved
        loaded_epoch = checkpoint.get('epoch', 'unknown') # Example if saved
        best_model_val_acc_loaded = checkpoint.get('best_val_char_acc', 0.0) # Example if saved
        print(f"Loaded model checkpoint. (Saved at epoch {loaded_epoch}, ValCharAcc: {best_model_val_acc_loaded:.4f} if saved)")
    except Exception as e:
        print(f"Error loading checkpoint {model_load_path}: {e}. Cannot generate test predictions.")
        exit() # Exit if loading fails
else:
    print(f"Warning: Best model checkpoint {model_load_path} not found. Cannot generate test predictions.")
    exit() # Exit if no model found

print(f"Generating test predictions (using loaded model)...")
model.eval() # Ensure model is in eval mode
test_predictions = []

with torch.no_grad():
    for (src_indices, _, _) in tqdm(test_loader, leave=False, desc="Test Prediction"):

        # --- Embedding Lookup ---
        # No need to move src_indices again, collate_fn handles it.

        # --- Autoregressive Generation --- (Similar to validate function)
        batch_size_current = src_indices.size(0)
        # Define a reasonable max length for generation if target isn't available
        # Increase buffer slightly more here too
        max_len_test = src_indices.size(1) + 30 # Heuristic: source length + buffer

        generated_indices = torch.full((batch_size_current, 1),
                                       char_to_idx['<sos>'],
                                       dtype=torch.long, device=device)

        # --- Precompute encoder output --- 
        input_emb_pos_gen = pos_enc(embedding(src_indices))
        src_key_padding_mask_gen = (src_indices == pad_idx)
        memory = model.encoder(input_emb_pos_gen, src_key_padding_mask=src_key_padding_mask_gen)
        # --- End Precompute --- 

        finished_sequences = torch.zeros(batch_size_current, dtype=torch.bool, device=device)

        for _ in range(max_len_test - 1):
            # Embed current generated sequence
            tgt_emb_pos = pos_enc(embedding(generated_indices))
            current_tgt_len = generated_indices.size(1)
            tgt_mask_gen = nn.Transformer.generate_square_subsequent_mask(current_tgt_len).to(device)

            output_step = model.decoder(
                tgt=tgt_emb_pos,
                memory=memory,
                tgt_mask=tgt_mask_gen,
                memory_key_padding_mask=src_key_padding_mask_gen
            )

            last_token_logits = decoder(output_step[:, -1, :])
            predicted_idx = last_token_logits.argmax(-1).unsqueeze(1)
            generated_indices = torch.cat([generated_indices, predicted_idx], dim=1)
            finished_sequences |= (predicted_idx.squeeze(1) == char_to_idx['<eos>'])
            if finished_sequences.all():
                break
        # --- End Autoregressive Generation ---

        # Decode generated indices
        batch_out_indices = generated_indices[:, 1:].detach().cpu().numpy() # Skip <sos>
        for i in range(batch_size_current):
            current_out = []
            for idx in batch_out_indices[i]:
                if idx == char_to_idx['<eos>'] or idx == pad_idx:
                    break
                if idx in idx_to_char:
                    current_out.append(idx_to_char[idx])
                else:
                     current_out.append('?') # Handle potential unknown indices
            test_predictions.append("".join(current_out))

# Save test predictions to file
test_output_path = "results/q2_test.txt"
with open(test_output_path, 'w') as f:
    for line in test_predictions:
        f.write(line + '\n')

print(f"Test predictions saved to {test_output_path}")
