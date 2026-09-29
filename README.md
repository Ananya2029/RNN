# Recurrent Neural Networks (RNN) — From Numbers to Sentences

Step-by-step notebooks that build Simple RNNs in TensorFlow/Keras, starting with the smallest possible sequence task and finishing with sequence-to-sequence text generation and a hands-on look at vanishing and exploding gradients.

| # | Notebook | What it does |
|---|---|---|
| 1 | `01_rnn_next_number.ipynb` | Predicts the next number in a sequence from a sliding window — the core idea of an RNN on the simplest data |
| 2 | `02_rnn_character_level.ipynb` | Learns the alphabet: given 3 letters (e.g. `A B C`), predicts the next one (`D`) |
| 3 | `03_rnn_word_level.ipynb` | Word-level next-word prediction on a repeating word sequence, using word-to-index encoding |
| 4 | `04_rnn_sentences_and_gradients.ipynb` | Four tasks on real sentences (below) |

**Notebook 4 — tasks**

1. **Next-word prediction** — 2-to-2 and 3-to-1 word models (e.g. `"The dog is"` → `"sleeping"`, `"I want some"` → `"tea"`).
2. **Sequence-to-sequence** — variable-length input → variable-length output with padding, masking, `RepeatVector` and `TimeDistributed` layers.
3. **Missing middle word** — reconstructs a sentence such as `"I like learning"` → `"I like machine learning"`.
4. **Vanishing & exploding gradients** — shows how repeated multiplication makes gradients explode, then compares training with and without **gradient clipping**.

## Key concepts covered

Sliding-window sequence framing · token-to-index encoding · `SimpleRNN` input shape `(samples, timesteps, features)` · many-to-one vs many-to-many · padding and masking · softmax output with sparse categorical cross-entropy · gradient clipping

## Run

```bash
pip install tensorflow numpy matplotlib
jupyter notebook
```

Each notebook is self-contained (no external data) and runs top to bottom.

## Tech Stack

Python · TensorFlow / Keras · NumPy · Matplotlib
