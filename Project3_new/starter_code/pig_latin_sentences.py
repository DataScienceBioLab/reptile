import json
import torch
import logging

class PigLatinSentences(torch.utils.data.Dataset):
    def __init__(self, split, char_to_idx):
        self.char_to_idx = char_to_idx
        self.english_sentences = []
        self.pig_latin_sentences = []

        # Reverted: Load data from the clean JSON file
        file_path = f"data/pig_latin_{split}.json"
        print(f"Loading data from: {file_path}")
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # Expecting 'english' key now based on preprocessor output
                    if 'english' in item and 'pig_latin' in item:
                        self.english_sentences.append(item['english'])
                        self.pig_latin_sentences.append(item['pig_latin'])
                    else:
                         logging.warning(f"Skipping item in {file_path}: Missing 'english' or 'pig_latin' key.")
            print(f"Loaded {len(self.english_sentences)} sentences from {file_path}")
        except FileNotFoundError:
            logging.error(f"Error: Data file not found at {file_path}. Did you run preprocess_data.py?")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error: Invalid JSON format in {file_path}.")
            raise

    def __len__(self):
        return len(self.english_sentences)

    def __getitem__(self, idx):
        english_sentence = self.english_sentences[idx]
        pig_latin_sentence = self.pig_latin_sentences[idx] # This might be None for test set

        # --- Process English Sentence --- 
        eng_tokens = ['<sos>'] + list(english_sentence) + ['<eos>']
        try:
            eng_word_idx = torch.tensor([self.char_to_idx[token] for token in eng_tokens], dtype=torch.long)
        except KeyError as e:
            logging.error(f"Unknown character in English sentence idx {idx}: {e}. Sentence: '{english_sentence}'")
            # Return dummy tensors or skip? For now, let's return dummy to avoid crashing loader
            eng_word_idx = torch.tensor([self.char_to_idx['<pad>']], dtype=torch.long)

        # --- Process Pig Latin Sentence (if it exists) --- 
        if pig_latin_sentence is not None:
            pig_latin_tokens = ['<sos>'] + list(pig_latin_sentence) + ['<eos>']
            try:
                pig_latin_word_idx = torch.tensor([self.char_to_idx[token] for token in pig_latin_tokens], dtype=torch.long)
            except KeyError as e:
                logging.error(f"Unknown character in Pig Latin sentence idx {idx}: {e}. Sentence: '{pig_latin_sentence}'")
                # Return dummy tensor
                pig_latin_word_idx = torch.tensor([self.char_to_idx['<pad>']], dtype=torch.long)
        else:
            # For test set or missing data, return None for pig latin part
            pig_latin_word_idx = None

        return eng_word_idx, pig_latin_word_idx
        