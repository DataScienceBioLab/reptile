from tqdm import trange, tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from yelp_dataset import YelpDataset
import seaborn as sns
import os
import numpy as np
import logging # Import logging

# Ensure necessary directories exist
os.makedirs('plots', exist_ok=True)
os.makedirs('results', exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
emb_dim = 50
batch_size = 128
rnn_dropout = 0.3
num_rnn_layers = 2
lr = 1e-3
num_epochs = 20
weight_decay = 1e-4

# --- Dataset Loading ---
data_dir = "data"
glove_full_path = "glove/modified_glove_50d.pt" # Path relative to script
print(f"Attempting to load GloVe from: {os.path.abspath(glove_full_path)}")
# Create dataset instances (vocab/vectors will be loaded/prepared by the Dataset class)
train_dataset = YelpDataset("train", glove_full_path=glove_full_path, data_dir=data_dir)
val_dataset = YelpDataset("val", glove_full_path=glove_full_path, data_dir=data_dir)
test_dataset = YelpDataset("test", glove_full_path=glove_full_path, data_dir=data_dir)

# --- Embedding Layer Initialization ---
# Use vocab_size, pad_idx, and vectors directly from the loaded train_dataset instance
vocab_size = train_dataset.vocab_size
# Ensure pad_idx is correctly fetched using the pad_token attribute
pad_idx = train_dataset.word_indices.get(train_dataset.pad_token)
if pad_idx is None:
    logging.warning(f"Padding token '{train_dataset.pad_token}' not found in final word_indices. Padding may not work correctly.")
    pad_idx = 0 # Fallback or handle error

glove_vectors = train_dataset.vectors.float()

print(f"Using vocab size: {vocab_size}")
print(f"Using pad index: {pad_idx}")
print(f"Using GloVe vectors shape: {glove_vectors.shape}")

# Create nn.Embedding instance from pre-trained vectors
# The vocab_size and vectors shape should match due to handling in YelpDataset
if vocab_size != glove_vectors.shape[0]:
    # This error should ideally be caught by the assertion in YelpDataset init
    raise ValueError(f"Mismatch persists after YelpDataset init: vocab {vocab_size} != vectors {glove_vectors.shape[0]}")

embeddings = nn.Embedding.from_pretrained(glove_vectors,
                                        freeze=False,
                                        padding_idx=pad_idx)

# Ensure padding index vector is zeroed out (good practice, might be redundant)
if pad_idx is not None and pad_idx < embeddings.num_embeddings:
    with torch.no_grad():
        embeddings.weight[pad_idx].fill_(0)

embeddings = embeddings.to(device)
print(f"Embedding layer created with shape: {embeddings.weight.shape} and fine-tuning enabled.")

# Define the padding index using the loaded vocabulary
# This line is now redundant as pad_idx is defined above
# pad_idx = train_dataset.word_indices.get('<pad>', 0)

def collate_fn(batch):
    # Check if batch contains tuples (indices, stars) or just indices (test set)
    is_test = isinstance(batch[0], torch.Tensor)
    # Use the pad_idx determined during dataset initialization and confirmed above
    pad_value = pad_idx

    if is_test:
        reviews_indices = batch
        stars = None
    else:
        reviews_indices = [item[0] for item in batch]
        stars = [item[1] for item in batch]

    # Store original lengths BEFORE padding
    lengths = torch.tensor([len(seq) for seq in reviews_indices], dtype=torch.long)

    # Pad review index sequences
    reviews_padded = torch.nn.utils.rnn.pad_sequence(reviews_indices, batch_first=True, padding_value=pad_value)

    # Move padded indices to device before embedding lookup
    reviews_padded_device = reviews_padded.to(device)
    embedded_reviews = embeddings(reviews_padded_device)

    # Pack the BATCH of embedded sequences
    # NOTE: We need lengths on CPU for pack_padded_sequence
    # Sort by lengths in descending order for packing (important!)
    lengths_sorted, perm_idx = lengths.sort(dim=0, descending=True)
    embedded_reviews_sorted = embedded_reviews[perm_idx]

    packed_embedded_reviews = torch.nn.utils.rnn.pack_padded_sequence(
        embedded_reviews_sorted, lengths_sorted.cpu(), batch_first=True
    )

    if stars:
        stars_tensor = torch.tensor(stars, dtype=torch.long).to(device)
        # Need to reorder stars according to the sort permutation
        stars_sorted = stars_tensor[perm_idx]
        return packed_embedded_reviews, stars_sorted
    else: # Test set
        # Need to return permutation index to reorder predictions later
        return packed_embedded_reviews, perm_idx

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                         shuffle=False, collate_fn=collate_fn)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                          shuffle=False, collate_fn=collate_fn)

# Create the RNN model
model = nn.RNN(
    input_size=emb_dim,
    hidden_size=emb_dim, # Output and hidden size are 50
    num_layers=num_rnn_layers,
    dropout=rnn_dropout if num_rnn_layers > 1 else 0, # Apply dropout only if layers > 1
    batch_first=True, # Crucial for packed sequence input
    bidirectional=False # Standard RNN as per diagram
)
model = model.to(device)

# Create the linear classifier
classifier = nn.Linear(
    in_features=emb_dim, # Input is RNN hidden state
    out_features=5 # 5 star ratings
)
classifier = classifier.to(device)

# Combine parameters from all three components
params = list(embeddings.parameters()) + list(model.parameters()) + list(classifier.parameters())
optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

# Create the loss function
criterion = nn.CrossEntropyLoss()

train_loss_list = []
train_acc_list = []
val_loss_list = []
val_acc_list = []
best_val_acc = 0.0 # Track best validation accuracy

print("Starting training...")
for epoch in range(num_epochs):
    # Training
    model.train()
    classifier.train()
    embeddings.train() # Ensure embeddings are trainable

    avg_train_loss = 0
    num_train_steps = 0
    correct_train = 0
    total_train_samples = 0

    pbar_train = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs} Train")
    for packed_batch, stars_sorted in pbar_train:
        # Packed batch is already on the correct device from collate_fn

        # Forward Pass
        optimizer.zero_grad()

        # Pass packed sequence through RNN
        packed_output, hidden_state = model(packed_batch)

        # Unpack sequence
        # output_padded shape: (batch, seq_len, hidden_size)
        # lengths_unpacked shape: (batch,) - contains original lengths
        output_padded, lengths_unpacked = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Extract the last relevant output
        # We need the output from the actual last time step for each sequence
        # The lengths tensor tells us the index of the last valid step (lengths - 1)
        # Need lengths_unpacked on the correct device if output_padded is on GPU
        lengths_unpacked = lengths_unpacked.to(device)
        last_output_indices = (lengths_unpacked - 1).view(-1, 1).expand(-1, output_padded.size(2))
        # Need to reshape indices for gather: (batch, 1, hidden_size)
        last_output_indices = last_output_indices.unsqueeze(1)

        # Use gather to select the output at the last time step for each sequence
        # output_padded shape: (batch, seq_len, hidden_size)
        # last_output_indices shape: (batch, 1, hidden_size)
        # gathered_output shape: (batch, 1, hidden_size)
        last_outputs = torch.gather(output_padded, 1, last_output_indices).squeeze(1)
        # last_outputs shape: (batch, hidden_size)

        # Pass through classifier
        logits = classifier(last_outputs)

        # Loss Calculation
        # Ratings are now 0-4 from the dataset
        loss = criterion(logits, stars_sorted)

        # Backward Pass & Optimization
        loss.backward()
        optimizer.step()

        # Accumulate Metrics
        avg_train_loss += loss.item()
        total_train_samples += stars_sorted.size(0)
        correct_train += (torch.argmax(logits, dim=1) == stars_sorted).sum().item()
        num_train_steps += 1
        pbar_train.set_postfix({"Loss": f"{loss.item():.4f}"})

    avg_train_loss /= num_train_steps
    train_accuracy = 100 * correct_train / total_train_samples
    train_loss_list.append(avg_train_loss)
    train_acc_list.append(train_accuracy)

    # Validation
    model.eval()
    classifier.eval()
    embeddings.eval() # Should be eval during validation

    avg_val_loss = 0
    num_val_steps = 0
    correct_val = 0
    total_val_samples = 0
    all_preds = []
    all_stars = []

    pbar_val = tqdm(val_loader, leave=False, desc=f"Epoch {epoch+1}/{num_epochs} Val")
        with torch.no_grad():
        for packed_batch, stars_sorted in pbar_val:
            # Packed batch is already on the correct device

            packed_output, hidden_state = model(packed_batch)
            output_padded, lengths_unpacked = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

            lengths_unpacked = lengths_unpacked.to(device)
            last_output_indices = (lengths_unpacked - 1).view(-1, 1).expand(-1, output_padded.size(2))
            last_output_indices = last_output_indices.unsqueeze(1)
            last_outputs = torch.gather(output_padded, 1, last_output_indices).squeeze(1)

            logits = classifier(last_outputs)
            # Ratings are now 0-4 from the dataset
            loss = criterion(logits, stars_sorted)

            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_stars.append(stars_sorted) # Store 0-4 labels

            avg_val_loss += loss.item()
            total_val_samples += stars_sorted.size(0)
            correct_val += (preds == stars_sorted).sum().item()
            num_val_steps += 1

    avg_val_loss /= num_val_steps
    val_accuracy = 100 * correct_val / total_val_samples
    val_loss_list.append(avg_val_loss)
    val_acc_list.append(val_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
          f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")

    # --- Confusion Matrix Data Calculation (remains after loop) ---
    all_preds_tensor = torch.cat(all_preds).cpu()
    all_stars_tensor = torch.cat(all_stars).cpu()
    confusion_matrix = torch.zeros(5, 5, dtype=torch.int)
    for t, p in zip(all_stars_tensor, all_preds_tensor):
        # Ensure indices are within bounds before incrementing
        if 0 <= t.long() < 5 and 0 <= p.long() < 5:
             confusion_matrix[t.long(), p.long()] += 1
        else:
             logging.warning(f"Skipping confusion matrix update for out-of-bounds prediction/target: T={t}, P={p}")

    # --- Save Best Model AND Best Confusion Matrix ---
    if val_accuracy > best_val_acc:
        print(f"  -> New best validation accuracy ({val_accuracy:.2f}%), saving models and best confusion matrix...")
        best_val_acc = val_accuracy
        # Save models
        torch.save(model.state_dict(), "results/q1_model.pt")
        torch.save(classifier.state_dict(), "results/q1_classifier.pt")
        torch.save(embeddings.state_dict(), "results/q1_embedding.pt")

        # Save the confusion matrix plot for the BEST epoch
        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=range(1, 6), yticklabels=range(1, 6)) # Use 1-5 labels
        plt.xlabel("Predicted Star Rating")
        plt.ylabel("Actual Star Rating")
        # Title now reflects it's the best one found so far
        plt.title(f"Best Confusion Matrix (Epoch {epoch+1}, Val Acc: {val_accuracy:.2f}%)")
        plt.tight_layout()
        plt.savefig("plots/q1_confusion_matrix_best.png", dpi=150) # Use fixed filename
        plt.close()

# Plotting Final Results
print("Training finished. Plotting final results...")
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(range(1, num_epochs + 1), train_loss_list, label="Train Loss")
axs[0].plot(range(1, num_epochs + 1), val_loss_list, label="Val Loss")
    axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Cross-Entropy Loss")
axs[0].set_title("Loss vs. Epoch")
    axs[0].legend()
axs[0].grid(True)

axs[1].plot(range(1, num_epochs + 1), train_acc_list, label="Train Accuracy")
axs[1].plot(range(1, num_epochs + 1), val_acc_list, label="Val Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy (%)")
axs[1].set_title("Accuracy vs. Epoch")
    axs[1].legend()
axs[1].grid(True)

    fig.tight_layout()
fig.savefig("plots/q1_final_plot.png", dpi=300)
    plt.close()
print("Final plots saved to plots/q1_final_plot.png")

# Generate Test Predictions
print("\nLoading best models for test prediction...")
model_load_path = "results/q1_model.pt"
classifier_load_path = "results/q1_classifier.pt"
embedding_load_path = "results/q1_embedding.pt"

if not all(os.path.exists(p) for p in [model_load_path, classifier_load_path, embedding_load_path]):
    print("Error: One or more best model files not found. Cannot generate test predictions.")
else:
    try:
        model.load_state_dict(torch.load(model_load_path, map_location=device))
        classifier.load_state_dict(torch.load(classifier_load_path, map_location=device))
        embeddings.load_state_dict(torch.load(embedding_load_path, map_location=device))
        print("Best models loaded successfully.")

        model.eval()
        classifier.eval()
        embeddings.eval()

        test_predictions = []
        # Store original indices to reorder predictions
        # We need a way to track original order since test loader shuffles by sort
        test_pred_dict = {}

        print("Generating test predictions...")
        with torch.no_grad():
            # Assuming test_loader provides original indices or we adapt it
            # Current collate_fn returns packed_batch, perm_idx for test
            for batch_idx, (packed_batch, perm_idx) in enumerate(tqdm(test_loader, desc="Test Prediction")):
                packed_output, _ = model(packed_batch)
                output_padded, lengths_unpacked = torch.nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

                lengths_unpacked = lengths_unpacked.to(device)
                last_output_indices = (lengths_unpacked - 1).view(-1, 1).expand(-1, output_padded.size(2))
                last_output_indices = last_output_indices.unsqueeze(1)
                last_outputs = torch.gather(output_padded, 1, last_output_indices).squeeze(1)

                logits = classifier(last_outputs)
                preds = torch.argmax(logits, dim=1) + 1 # Convert back to 1-5 rating

                # Unsort the predictions based on perm_idx
                _, unperm_idx = perm_idx.sort(dim=0) # Get inverse permutation
                preds_unsorted = preds[unperm_idx]

                # Store predictions corresponding to their original batch index
                # Calculate original indices based on batch_idx and batch_size
                start_original_idx = batch_idx * test_loader.batch_size
                for i, pred in enumerate(preds_unsorted.cpu().tolist()):
                    original_idx = start_original_idx + i
                    test_pred_dict[original_idx] = pred

        # Sort predictions by original index and create final list
        test_predictions = [test_pred_dict[i] for i in sorted(test_pred_dict.keys())]

        # Save test predictions
        test_output_path = "results/q1_test.txt"
        try:
            with open(test_output_path, 'w') as f:
                for pred in test_predictions:
                    f.write(str(pred) + '\n')
            print(f"Test predictions saved to {test_output_path}")
        except Exception as e:
            print(f"Error writing test predictions to {test_output_path}: {e}")

    except Exception as e:
        print(f"Error during test prediction: {e}")
