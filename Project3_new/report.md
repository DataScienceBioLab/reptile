# Project 3: Sequence Modeling

**Name:** Kevin Mok
**Course:** CSE 849 Deep Learning (Spring 2025)
**Instructor:** Zijun Cui
**Date:** 04/11/2025

## Introduction

This report details the implementation and results for Project 3, focusing on sequence modeling techniques. Question 1 involved predicting Yelp review ratings using a Recurrent Neural Network (RNN) with pre-trained GloVe embeddings. Question 2 focused on translating English sentences into Pig Latin using a Transformer architecture.

---

## Question 1: Review Rating Prediction

### Methodology

The goal was to predict a star rating (1-5) given the text of a Yelp review. The approach involved:

1.  **Data Loading:** Utilizing a custom `YelpDataset` class to load reviews and star ratings from JSON Lines files (`data/yelp_dataset_{split}.json`). A vocabulary (`word_indices`) and embedding tensor (`vectors`) were derived from the provided `glove/modified_glove_50d.pt` file (loaded as a `dict[str, tensor]`), handling `<pad>` (index 0) and `<unk>` tokens. Reviews were limited to a maximum length of 40 words.
2.  **Embeddings:** Using an `nn.Embedding` layer initialized with the 50-dimensional GloVe vectors via `from_pretrained`. These embeddings were fine-tuned during training (`freeze=False`). The padding index was explicitly set and zeroed out.
3.  **Model:** An `nn.RNN` with 2 layers and a hidden dimension of 50 was used, incorporating dropout (0.3) between layers and `batch_first=True`.
4.  **Collate Function:** A custom `collate_fn` was implemented to handle variable-length sequences. It pads the word index sequences, looks up embeddings *within the collate function*, sorts the batch by sequence length, and uses `pack_padded_sequence` to prepare input for the RNN. It handles test vs. train/val batches correctly.
5.  **Classification:** The output vector corresponding to the *last actual time step* (before padding) of the RNN sequence was extracted using `torch.gather` and fed into a `nn.Linear` classifier layer (50 input features, 5 output classes).
6.  **Training:** The embedding layer, RNN, and classifier were trained jointly using `AdamW` optimizer (LR=1e-3, WD=1e-4) and `CrossEntropyLoss`. The dataset provides 0-4 indexed targets for the loss. The best model components were saved based on validation accuracy.

### Results

**Training Curves:**

The following plot shows the training and validation loss and accuracy curves over the 20 training epochs:

![Q1 Loss and Accuracy Curves](starter_code/plots/q1_final_plot.png)

*Discussion:* The training loss decreased steadily, indicating the model was learning from the training data. Validation loss decreased rapidly in the first few epochs, reaching its minimum around epoch 4, before starting to increase slightly, indicating the onset of overfitting. Similarly, training accuracy increased consistently throughout, while validation accuracy peaked early at epoch 4 and then plateaued or slightly decreased. The widening gap between training and validation metrics after epoch 4 confirms that the model began overfitting the training data.

**Best Validation Accuracy:**

The best validation accuracy achieved during training was **57.26%** (at epoch 4).

**Confusion Matrix:**

The confusion matrix below corresponds to the epoch where the best validation accuracy was achieved (Epoch 4). It shows the distribution of predicted vs. actual star ratings on the validation set.

![Q1 Best Confusion Matrix](starter_code/plots/q1_confusion_matrix_best.png)

*Analysis:* The diagonal indicates the model performs significantly better than random chance, achieving its peak generalization at epoch 4. As seen in the confusion matrix from that epoch, it appears most accurate at predicting 5-star reviews. The main confusion occurs between adjacent star ratings (e.g., predicting 4 stars for 5-star reviews, or 2 stars for 1-star reviews), which is intuitive as the textual differences might be subtle. There is less confusion between ratings at opposite ends of the scale (e.g., predicting 1 star for a 5-star review is rare).

---

## Question 2: Pig Latin Translation

### Methodology

The goal was to translate English sentences into Pig Latin character by character using a sequence-to-sequence model.

1.  **Data Loading:** Using `PigLatinSentences` dataset class and a character-level vocabulary including `<sos>`, `<eos>`, and `<pad>`. Sentences were processed from JSON Lines files.
2.  **Embeddings:** Employing a trainable `nn.Embedding` layer (dimension 100) to represent characters.
3.  **Positional Encoding:** Adding positional encodings to both encoder and decoder inputs. The `PositionalEncoding` class was modified to correctly handle `batch_first=True` tensor dimensions, which proved crucial for successful training.
4.  **Model:** A standard `nn.Transformer` architecture was used with the parameters specified in the project description: 2 encoder layers, 2 decoder layers, 2 attention heads (`nhead=2`), and a feedforward dimension of 128 (`dim_feedforward=128`). Dropout (0.1) was used (`transformer_dropout=0.1`). `batch_first=True` was used throughout.
5.  **Output Layer:** A `nn.Linear` layer decoded the Transformer's output embeddings back into logits over the character vocabulary (30 tokens).
6.  **Training:** The model was trained using the `AdamW` optimizer (WD=1e-2) and `CrossEntropyLoss`. Initial runs experimented with label smoothing, but the final successful configuration used `label_smoothing=0.0`. A linear warm-up (2000 steps) followed by a linear decay learning rate schedule was implemented, with a maximum learning rate of `5e-4`. Gradient clipping (max_norm=1.0) was applied. Training used teacher forcing.
7.  **Evaluation:** Validation accuracy was measured using autoregressive generation. Both character-level and sentence-level accuracy were tracked. The best model checkpoints were saved based on character accuracy during the main run (`q2.py`).
8.  ~~**(Optional) Fine-tuning:** Fine-tuning steps with a lower constant learning rate were considered but ultimately the main training run achieved the target performance.~~

### Results

**Training Curves:**

The following plot shows the training and validation loss and accuracy curves over the main training run (`q2.py` after fixing the Positional Encoding issue and disabling label smoothing):

![Q2 Loss and Accuracy Curves](starter_code/plots/q2_results.png)

*Discussion:* After correcting the `PositionalEncoding` to handle `batch_first=True`, the model trained effectively with the specified 2/2/128 architecture. The warm-up/decay learning rate schedule led to stable and rapid convergence. Character accuracy improved quickly, exceeding 99.7% and peaking at 99.81%. Sentence accuracy also increased significantly, surpassing the 95% target and reaching its peak of 95.64% by the end of training, closely following the character accuracy. Disabling label smoothing appeared beneficial in achieving the final high sentence accuracy. Overfitting appears minimal based on the validation curves closely tracking the training curves.

**(Optional) Fine-tuning Curves:**

<!-- This section is removed as fine-tuning was not performed separately
[If you performed fine-tuning and generated the dual save plots, include them]

```markdown
![Q2 Fine-tuning Curves](starter_code/plots/q2_finetune_dual_save_results.png)

*Discussion:* The fine-tuning phase with a low constant learning rate allowed for minor adjustments. While character accuracy remained very high, sentence accuracy [mention if it improved, e.g., "improved slightly, crossing the 95% threshold" or "remained stable around its peak from the main run"]. This suggests the model was already close to its optimal performance for exact sentence matches.
```
-->

**Best Validation Accuracy:**

*   The best validation **character** accuracy achieved during the training run was **99.81%**.
*   The best validation **sentence** accuracy achieved during the training run was **95.64%**.

**Challenges & Observations:**

Initial training attempts were significantly hindered by an implementation detail regarding the interaction between `nn.Transformer(batch_first=True)` and the standard `PositionalEncoding` module. The positional encodings were being applied incorrectly, preventing the model from learning sequence order effectively, despite trying larger architectures and various hyperparameters. Modifying the `PositionalEncoding` class to correctly add encodings along the sequence dimension (dim=1) when `batch_first=True` was the critical fix. Once resolved, the specified architecture trained efficiently using a warm-up/decay schedule (max LR 5e-4) and no label smoothing, confirming the instructor's guidance that the architecture was sufficient and achieving high performance within the 70 epochs.

---

## Conclusion

This project successfully demonstrated the application of sequence models to NLP tasks. For Question 1, the RNN model achieved its best validation accuracy of **57.26%** early in training (epoch 4) in predicting Yelp ratings, showing proficiency in capturing sequence information before overfitting occurred. For Question 2, after addressing implementation challenges with positional encoding, the Transformer model effectively learned the character-level rules of Pig Latin translation, achieving a high validation character accuracy of **99.81%** and a sentence accuracy of **95.64%**. Both models met the performance expectations and grading requirements outlined in the project description.
