# Email Phishing Detector 

AI-powered web app that detects phishing emails using neural networks.

## What it does 

Paste an email into the web interface and the AI tells you if it's phishing or safe. Uses a neural network trained on email datasets to spot suspicious patterns in English emails.

## How the AI Works:

Building the AI for this phishing detector involved several key steps to ensure it could accurately understand email content and identify malicious patterns:

1.  **Gathering Diverse Data:** I started by combining multiple datasets. An initial dataset contained clearly labeled phishing and legitimate emails. Realizing the "legitimate" emails were mostly formal, I added the large **Enron Email Dataset** to teach the model about *normal, informal* emails. This resulted in a balanced dataset of over 43,000 emails.

2.  **Cleaning the Text:** Raw email text is messy. I created a cleaning pipeline using NLTK:
    * Converted all text to **lowercase**.
    * Removed **punctuation, numbers, and special symbols**.
    * Removed common English **"stop words"** (like "the," "is," "and") to focus the model on important keywords.

3.  **Tokenization (Text-to-Numbers):** Neural networks need numbers. I used a `Tokenizer` from TensorFlow/Keras to build a dictionary of all unique words in the training data (over 126,000 words!). Each email was converted into a sequence of numbers based on this dictionary.

4.  **Padding Sequences:** Networks require fixed-size inputs. I set a maximum length of **200 words**. Shorter emails were "padded" with zeros at the end; longer ones were truncated.

5.  **Injecting "World Knowledge" (GloVe Embeddings):** Instead of learning word meanings from scratch, I used **GloVe**, a pre-trained dictionary from Stanford containing "meaning vectors" for 400,000 English words learned from billions of words online. I built an "embedding matrix" linking the project's words to their GloVe meanings.

6.  **Building the Neural Network:** I designed a sequence-aware network:
    * **Embedding Layer:** Replaced word numbers with their 100-dimension GloVe meaning vectors. Crucially, I set this layer to `trainable=True` to allow **fine-tuning** of GloVe meanings for phishing detection.
    * **Bidirectional LSTM Layer:** The core memory unit reads the word sequence forwards *and* backwards to understand context and word order.
    * **Dense Layers & Dropout:** Final decision-making layers. `Dropout` was used to prevent overfitting (memorization).
    * **Output Layer:** A single neuron outputting a score between 0 (Safe) and 1 (Phishing).

7.  **Training:** The model was trained on ~35,000 emails (80% of data) for 5 epochs using a GPU, learning to minimize prediction errors.

8.  **Evaluation:** The model achieved ~98-99% accuracy on unseen test emails. Crucially, fine-tuning allowed it to correctly classify informal emails ("Hi mom") as safe while still catching phishing, including misspelled attempts.

This combination of diverse data, text processing, pre-trained embeddings, fine-tuning, and a sequence-aware network allows the AI to effectively analyze English emails.

## Running it 

```bash
pip install -r requirements.txt
python app.py