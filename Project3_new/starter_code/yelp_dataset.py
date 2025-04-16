import json
import torch
from torch.utils.data import Dataset
import os
import logging
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YelpDataset(Dataset):
    def __init__(self, split, glove_full_path="glove/modified_glove_50d.pt", data_dir="data"):
        self.split = split
        self.word_indices = {}
        self.idx_to_word = []
        self.vectors = None
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'
        self.max_len = 40

        # --- Load GloVe (expecting dict[str, tensor]) ---
        if not os.path.exists(glove_full_path):
            logging.error(f"GloVe file not found at: {glove_full_path}")
            raise FileNotFoundError(f"GloVe file not found at: {glove_full_path}")

        try:
            glove_data_dict = torch.load(glove_full_path, map_location=torch.device('cpu'))
            if not isinstance(glove_data_dict, dict):
                raise TypeError(f"Loaded GloVe data is not a dictionary. Got {type(glove_data_dict)}")

            logging.info(f"Loaded GloVe dictionary with {len(glove_data_dict)} word entries.")

            # --- Build vocab and vectors from dict --- 
            glove_words = list(glove_data_dict.keys())
            # Start vocab with special tokens
            self.idx_to_word = [self.pad_token, self.unk_token]
            self.idx_to_word.extend(glove_words) # Add words from GloVe

            self.word_indices = {word: idx for idx, word in enumerate(self.idx_to_word)}
            self.vocab_size = len(self.idx_to_word)

            # Determine embedding dim from first item
            first_word = next(iter(glove_data_dict))
            emb_dim = glove_data_dict[first_word].shape[0] # Assuming shape [emb_dim] or [1, emb_dim]
            if len(glove_data_dict[first_word].shape) > 1:
                 emb_dim = glove_data_dict[first_word].shape[1]
            logging.info(f"Detected embedding dimension: {emb_dim}")

            # Create the vectors tensor, adding vectors for pad and unk
            self.vectors = torch.zeros((self.vocab_size, emb_dim), dtype=torch.float)
            # Initialize <unk> vector (e.g., small random)
            self.vectors[self.word_indices[self.unk_token]] = torch.randn(emb_dim) * 0.01
            # <pad> vector remains zeros (index 0)

            # Fill in vectors from the loaded dictionary
            for word, vec_tensor in glove_data_dict.items():
                 if word in self.word_indices:
                      idx = self.word_indices[word]
                      try:
                          self.vectors[idx] = vec_tensor.squeeze().float() # Ensure correct shape and type
                      except Exception as assign_e:
                          logging.error(f"Error assigning vector for word '{word}' at index {idx}: {assign_e}")
                          raise
                 else:
                     logging.warning(f"Word '{word}' from GloVe dict not in constructed vocab? Skipping.")

            logging.info(f"Final vocabulary size: {self.vocab_size}")
            logging.info(f"Final embedding vector shape: {self.vectors.shape}")
            assert self.vocab_size == self.vectors.shape[0], "Mismatch between final vocab size and vector count!"

        except Exception as e:
            logging.error(f"Error loading or processing GloVe file {glove_full_path}: {e}")
            raise

        # --- Load Yelp JSON Lines data --- 
        data_file = os.path.join(data_dir, f"yelp_dataset_{split}.json")
        if not os.path.exists(data_file):
            logging.error(f"Yelp data file not found: {data_file}")
            raise FileNotFoundError(f"Yelp data file not found: {data_file}")
        
        self.reviews = []
        self.stars = []
        try:
            # Read JSON Lines file line by line
            with open(data_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line: continue # Skip empty lines
                    try:
                        item = json.loads(line) # Use json.loads for each line
                    except json.JSONDecodeError:
                        logging.warning(f"Skipping invalid JSON on line {line_num+1} in {data_file}")
                        continue

                    text_key = 'text' if 'text' in item else 'review'
                    stars_key = 'stars'
                    if text_key in item and (split == 'test' or stars_key in item):
                        self.reviews.append(item[text_key].lower())
                        if split != 'test':
                            try:
                                self.stars.append(int(item[stars_key]))
                            except (ValueError, TypeError):
                                logging.warning(f"Skipping item due to invalid stars: {item.get(stars_key)}")
                                self.reviews.pop()
                    else:
                        logging.warning(f"Skipping item missing keys: {item}")
            logging.info(f"Loaded {len(self.reviews)} reviews for split '{split}'.")
            if split != 'test': assert len(self.reviews) == len(self.stars)
        except Exception as e:
            logging.error(f"Error loading Yelp data {data_file}: {e}")
            raise

    def __len__(self):
        return len(self.reviews)

    def __getitem__(self, idx):
        review_text = self.reviews[idx]
        pad_idx = self.word_indices[self.pad_token]
        unk_idx = self.word_indices[self.unk_token]

        word_indices_list = []
        words = review_text.split(" ")[:self.max_len]
        for word in words:
            word_indices_list.append(self.word_indices.get(word, unk_idx))

        if not word_indices_list:
            word_indices_list.append(pad_idx)

        indices_tensor = torch.tensor(word_indices_list, dtype=torch.long)

        if self.split == "test":
            return indices_tensor
        else:
            # Convert 1-5 stars to 0-4 index for CrossEntropyLoss
            star_rating = torch.tensor(self.stars[idx] - 1, dtype=torch.long)
            return indices_tensor, star_rating

