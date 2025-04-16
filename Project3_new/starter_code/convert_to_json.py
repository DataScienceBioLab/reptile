import json
import os
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def convert_txt_to_json(split):
    """
    Reads the TXT file containing line-by-line JSON, parses it,
    and writes a clean JSON list file.
    Handles potential absence of 'pig_latin' key in the test set.
    """
    input_txt_path = os.path.join("data", f"reviews_pig_latin_data_{split}.txt")
    output_json_path = os.path.join("data", f"pig_latin_{split}.json")

    if not os.path.exists(input_txt_path):
        logging.error(f"Input file not found: {input_txt_path}")
        return

    logging.info(f"Converting {input_txt_path} -> {output_json_path}")

    data_list = []
    processed_lines = 0
    skipped_lines = 0

    try:
        with open(input_txt_path, 'r', encoding='utf-8') as infile:
            for i, line in enumerate(infile):
                line = line.strip()
                if not line:
                    skipped_lines += 1
                    continue # Skip empty lines

                try:
                    data = json.loads(line)
                    entry = {}
                    if 'original' in data:
                        entry['english'] = data['original'] # Rename key
                        # Pig Latin key might be missing, especially in test set
                        entry['pig_latin'] = data.get('pig_latin', None) # Use .get() for safety
                        data_list.append(entry)
                        processed_lines += 1
                    else:
                        logging.warning(f"Skipping line {i+1}: Missing 'original' key.")
                        skipped_lines += 1
                except json.JSONDecodeError:
                    logging.warning(f"Skipping line {i+1}: Invalid JSON format on line: {line[:100]}...")
                    skipped_lines += 1
                except Exception as e:
                     logging.warning(f"Skipping line {i+1}: Unexpected error parsing line ({e}): {line[:100]}...")
                     skipped_lines += 1

    except Exception as e:
        logging.error(f"Error reading {input_txt_path}: {e}")
        return

    if not data_list:
        logging.error(f"No valid data extracted from {input_txt_path}. Output file not created.")
        return

    try:
        # Ensure the data directory exists
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(data_list, outfile, indent=2) # Use indent for readability
        logging.info(f"Successfully created {output_json_path} with {processed_lines} entries.")
        if skipped_lines > 0:
            logging.warning(f"Skipped {skipped_lines} lines during conversion.")
    except Exception as e:
        logging.error(f"Error writing {output_json_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Pig Latin TXT data (line-JSON) into a single JSON list file.")
    parser.add_argument("split", choices=["train", "val", "test"], help="The data split to process (train, val, test).")
    args = parser.parse_args()

    convert_txt_to_json(args.split) 