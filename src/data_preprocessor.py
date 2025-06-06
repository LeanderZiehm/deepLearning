import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import Counter
import re
import pickle
from tqdm import tqdm

# Download required NLTK data
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")


class TextPreprocessor:
    """Text preprocessing utilities for hierarchical document structure."""

    def __init__(self, maximum_vocabulary_size=50000, minimum_word_frequency=2):
        self.maximum_vocabulary_size = maximum_vocabulary_size
        self.minimum_word_frequency = minimum_word_frequency
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.vocabulary_size = 0

    def clean_text(self, text):
        """Clean and normalize text."""
        if pd.isna(text):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove extra whitespace and newlines
        text = re.sub(r"\s+", " ", text)

        # Remove special characters but keep punctuation for sentence structure
        text = re.sub(r"[^\w\s.,!?;:]", " ", text)

        return text.strip()

    def build_vocabulary(self, texts):
        """Build vocabulary from training texts."""
        print("Building vocabulary...")
        word_counts = Counter()

        for text in tqdm(texts, desc="Processing texts"):
            cleaned_text = self.clean_text(text)
            sentences = sent_tokenize(cleaned_text)

            for sentence in sentences:
                words = word_tokenize(sentence)
                word_counts.update(words)

        # Create word-to-index mapping
        # Reserve indices: 0=padding, 1=unknown
        self.word_to_idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx_to_word = {0: "<PAD>", 1: "<UNK>"}

        # Add words that meet frequency threshold
        valid_words = [
            word
            for word, count in word_counts.items()
            if count >= self.minimum_word_frequency
        ]

        # Sort by frequency and take top words
        valid_words = sorted(valid_words, key=lambda x: word_counts[x], reverse=True)
        valid_words = valid_words[
            : self.maximum_vocabulary_size - 2
        ]  # -2 for PAD and UNK

        for idx, word in enumerate(valid_words, start=2):
            self.word_to_idx[word] = idx
            self.idx_to_word[idx] = word

        self.vocabulary_size = len(self.word_to_idx)
        print(f"Vocabulary size: {self.vocabulary_size}")

    def text_to_hierarchical_indices(
        self, text, max_sentences=20, max_words_per_sentence=50
    ):
        """Convert text to hierarchical structure of word indices."""
        cleaned_text = self.clean_text(text)
        sentences = sent_tokenize(cleaned_text)

        # Limit number of sentences
        sentences = sentences[:max_sentences]

        hierarchical_doc = []
        for sentence in sentences:
            words = word_tokenize(sentence)
            # Limit words per sentence
            words = words[:max_words_per_sentence]

            # Convert words to indices
            word_indices = []
            for word in words:
                idx = self.word_to_idx.get(word, 1)  # 1 is <UNK>
                word_indices.append(idx)

            if word_indices:  # Only add non-empty sentences
                hierarchical_doc.append(word_indices)

        return hierarchical_doc if hierarchical_doc else [[1]]  # Return <UNK> if empty

    def save_vocabulary(self, filepath):
        """Save vocabulary to file."""
        vocab_data = {
            "word_to_idx": self.word_to_idx,
            "idx_to_word": self.idx_to_word,
            "vocabulary_size": self.vocabulary_size,
        }
        with open(filepath, "wb") as f:
            pickle.dump(vocab_data, f)

    def load_vocabulary(self, filepath):
        """Load vocabulary from file."""
        with open(filepath, "rb") as f:
            vocab_data = pickle.load(f)

        self.word_to_idx = vocab_data["word_to_idx"]
        self.idx_to_word = vocab_data["idx_to_word"]
        self.vocabulary_size = vocab_data["vocabulary_size"]


class FakeNewsDataset(Dataset):
    """Dataset class for fake news classification."""

    def __init__(
        self, texts, labels, preprocessor, max_sentences=20, max_words_per_sentence=50
    ):
        self.texts = texts
        self.labels = labels
        self.preprocessor = preprocessor
        self.max_sentences = max_sentences
        self.max_words_per_sentence = max_words_per_sentence

        # Convert texts to hierarchical format
        self.hierarchical_docs = []
        for text in tqdm(texts, desc="Convertng text to hierarchical format..."):
            doc = preprocessor.text_to_hierarchical_indices(
                text, max_sentences, max_words_per_sentence
            )
            self.hierarchical_docs.append(doc)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.hierarchical_docs[idx], self.labels[idx]


def create_padding_for_hierarchical_sequences(
    documents, max_sentence_length=None, max_word_length=None, pad_token=0
):

    if max_sentence_length is None:
        max_sentence_length = max(len(doc) for doc in documents)
    if max_word_length is None:
        max_word_length = max(len(sent) for doc in documents for sent in doc)

    num_docs = len(documents)
    padded_docs = np.full(
        (num_docs, max_sentence_length, max_word_length), pad_token, dtype=np.int64
    )
    word_lengths = np.zeros((num_docs, max_sentence_length), dtype=np.int64)
    sentence_lengths = np.zeros(num_docs, dtype=np.int64)

    for i, doc in enumerate(documents):
        sentence_lengths[i] = min(len(doc), max_sentence_length)
        for j, sent in enumerate(doc[:max_sentence_length]):
            word_len = min(len(sent), max_word_length)
            word_lengths[i, j] = word_len
            padded_docs[i, j, :word_len] = sent[:word_len]

    return padded_docs, word_lengths, sentence_lengths


def collate_fn(batch):
    """Custom collate function for hierarchical data."""
    docs, labels = zip(*batch)

    # Pad hierarchical sequences
    padded_docs, word_lengths, sentence_lengths = create_padding_for_hierarchical_sequences(list(docs))

    return (
        torch.tensor(padded_docs, dtype=torch.long),
        torch.tensor(word_lengths, dtype=torch.long),
        torch.tensor(sentence_lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )
