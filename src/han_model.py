import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class WordAttention(nn.Module):
    """
    Word-level attention mechanism as described in the HAN paper.
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dimmentions,
        word_gru_hidden_units,
        word_gru_layers,
        word_attention_dimmentions,
    ):
        super(WordAttention, self).__init__()

        # Word embedding layer
        self.embedding = nn.Embedding(
            vocabulary_size, embedding_dimmentions, padding_idx=0
        )

        # Bidirectional GRU for word-level encoding
        self.word_gru = nn.GRU(
            embedding_dimmentions,
            word_gru_hidden_units,
            num_layers=word_gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if word_gru_layers > 1 else 0,
        )

        # Word attention mechanism
        self.word_attention = nn.Linear(
            2 * word_gru_hidden_units, word_attention_dimmentions
        )
        self.word_context_vector = nn.Linear(word_attention_dimmentions, 1, bias=False)

        self.dropout = nn.Dropout(0.1)

    def forward(self, sentences, word_lengths):

        batch_size, max_sentence_length, max_word_length = sentences.size()

        # Reshape to process all sentences together
        sentences = sentences.view(-1, max_word_length)
        word_lengths = word_lengths.view(-1)

        # Remove sentences with zero length
        valid_sentences = word_lengths > 0
        if not valid_sentences.any():
            # Handle case where all sentences are empty
            zero_output = torch.zeros(
                batch_size,
                max_sentence_length,
                2 * self.word_gru.hidden_size,
                device=sentences.device,
            )
            zero_weights = torch.zeros(
                batch_size,
                max_sentence_length,
                max_word_length,
                device=sentences.device,
            )
            return zero_output, zero_weights

        # Process only valid sentences
        valid_sentences_data = sentences[valid_sentences]
        valid_word_lengths = word_lengths[valid_sentences]

        # Word embeddings
        embedded = self.embedding(valid_sentences_data)
        embedded = self.dropout(embedded)

        # Pack sequences for efficient RNN processing
        packed_embedded = pack_padded_sequence(
            embedded, valid_word_lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Bidirectional GRU
        packed_output, _ = self.word_gru(packed_embedded)
        word_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Word attention mechanism
        attention_weights = torch.tanh(self.word_attention(word_outputs))
        attention_weights = self.word_context_vector(attention_weights).squeeze(-1)
        # Create attention mask for padding tokens
        word_mask = (
            torch.arange(word_outputs.size(1), device=word_outputs.device)[None, :]
            < valid_word_lengths[:, None]
        )
        attention_weights = attention_weights.masked_fill(~word_mask, -float("inf"))
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights
        sentence_vectors = torch.sum(
            word_outputs * attention_weights.unsqueeze(-1), dim=1
        )

        # Reconstruct full batch including zero vectors for invalid sentences
        full_sentence_vectors = torch.zeros(
            batch_size * max_sentence_length,
            sentence_vectors.size(-1),
            device=sentences.device,
        )
        full_attention_weights = torch.zeros(
            batch_size * max_sentence_length, max_word_length, device=sentences.device
        )

        full_sentence_vectors[valid_sentences] = sentence_vectors
        full_attention_weights[valid_sentences] = attention_weights

        # Reshape back to original dimensions
        sentence_vectors = full_sentence_vectors.view(
            batch_size, max_sentence_length, -1
        )
        word_attention_weights = full_attention_weights.view(
            batch_size, max_sentence_length, max_word_length
        )

        return sentence_vectors, word_attention_weights


class SentenceAttention(nn.Module):
    """
    Sentence-level attention mechanism as described in the HAN paper.
    """

    def __init__(
        self,
        word_gru_hidden_units,
        sentence_gru_hidden_units,
        sentence_gru_layers,
        sentence_attention_dimmention,
        number_of_classes,
    ):
        super(SentenceAttention, self).__init__()

        # Bidirectional GRU for sentence-level encoding
        self.sentence_gru = nn.GRU(
            2 * word_gru_hidden_units,
            sentence_gru_hidden_units,
            num_layers=sentence_gru_layers,
            bidirectional=True,
            batch_first=True,
            dropout=0.1 if sentence_gru_layers > 1 else 0,
        )

        # Sentence attention mechanism
        self.sentence_attention = nn.Linear(
            2 * sentence_gru_hidden_units, sentence_attention_dimmention
        )
        self.sentence_context_vector = nn.Linear(
            sentence_attention_dimmention, 1, bias=False
        )

        # Final classification layer
        self.classifier = nn.Linear(2 * sentence_gru_hidden_units, number_of_classes)

        self.dropout = nn.Dropout(0.1)

    def forward(self, sentence_vectors, sentence_lengths):

        sentence_vectors = self.dropout(sentence_vectors)

        # Pack sequences for efficient RNN processing
        packed_sentences = pack_padded_sequence(
            sentence_vectors,
            sentence_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        # Bidirectional GRU
        packed_output, _ = self.sentence_gru(packed_sentences)
        sentence_outputs, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Sentence attention mechanism
        attention_weights = torch.tanh(self.sentence_attention(sentence_outputs))
        attention_weights = self.sentence_context_vector(attention_weights).squeeze(-1)

        # Create attention mask for padding sentences
        sentence_mask = (
            torch.arange(sentence_outputs.size(1), device=sentence_outputs.device)[
                None, :
            ]
            < sentence_lengths[:, None]
        )
        attention_weights = attention_weights.masked_fill(~sentence_mask, -float("inf"))
        attention_weights = F.softmax(attention_weights, dim=1)

        # Apply attention weights to get document representation
        doc_vectors = torch.sum(
            sentence_outputs * attention_weights.unsqueeze(-1), dim=1
        )

        # Final classification
        logits = self.classifier(doc_vectors)

        return logits, attention_weights


class HierarchicalAttentionNetwork(nn.Module):
    """
    Complete Hierarchical Attention Network implementation.
    """

    def __init__(
        self,
        vocabulary_size,
        embedding_dimmentions,
        word_gru_hidden_units,
        word_gru_layers,
        word_attention_dimmentions,
        sentence_gru_hidden_units,
        sentence_gru_layers,
        sentence_attention_dimmention,
        number_of_classes,
        pretrained_embeddings=None,
    ):
        super(HierarchicalAttentionNetwork, self).__init__()

        self.word_attention = WordAttention(
            vocabulary_size,
            embedding_dimmentions,
            word_gru_hidden_units,
            word_gru_layers,
            word_attention_dimmentions,
        )

        self.sentence_attention = SentenceAttention(
            word_gru_hidden_units,
            sentence_gru_hidden_units,
            sentence_gru_layers,
            sentence_attention_dimmention,
            number_of_classes,
        )

        # Initialize embeddings with pre-trained vectors if provided
        if pretrained_embeddings is not None:
            self.word_attention.embedding.weight.data.copy_(
                torch.from_numpy(pretrained_embeddings)
            )
            # Freeze embedding layer if desired
            # self.word_attention.embedding.weight.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "embedding" not in name:  # Don't reinitialize embeddings
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.zeros_(param)

    def forward(self, documents, word_lengths, sentence_lengths):
        # Word-level attention
        sentence_vectors, word_attention_weights = self.word_attention(
            documents, word_lengths
        )

        # Sentence-level attention
        logits, sentence_attention_weights = self.sentence_attention(
            sentence_vectors, sentence_lengths
        )

        return logits, word_attention_weights, sentence_attention_weights


# Example usage and training utilities
class HANTrainer:
    """Training utilities for Hierarchical Attention Network."""

    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()

    def trainset_step(self, batch, optimizer):
        """Single training step."""
        self.model.train()
        documents, word_lengths, sentence_lengths, labels = batch

        # Move to device
        documents = documents.to(self.device)
        word_lengths = word_lengths.to(self.device)
        sentence_lengths = sentence_lengths.to(self.device)
        labels = labels.to(self.device)

        optimizer.zero_grad()

        # Forward pass
        logits, _, _ = self.model(documents, word_lengths, sentence_lengths)

        # Compute loss
        loss = self.criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

        optimizer.step()

        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()

        return loss.item(), accuracy.item()

    def evaluate(self, dataloader):
        """Evaluate model on validation/test set."""
        self.model.eval()
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                documents, word_lengths, sentence_lengths, labels = batch

                # Move to device
                documents = documents.to(self.device)
                word_lengths = word_lengths.to(self.device)
                sentence_lengths = sentence_lengths.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                logits, _, _ = self.model(documents, word_lengths, sentence_lengths)

                # Compute loss and accuracy
                loss = self.criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)
                accuracy = (predictions == labels).float().mean()

                total_loss += loss.item()
                total_accuracy += accuracy.item()
                num_batches += 1

        return total_loss / num_batches, total_accuracy / num_batches


# Creates the Hierarchical Attention Network
def create_han_model(
    vocabulary_size=10000, embedding_dimmentions=200, number_of_classes=5
):

    model = HierarchicalAttentionNetwork(
        vocabulary_size=vocabulary_size,
        embedding_dimmentions=embedding_dimmentions,
        word_gru_hidden_units=50,
        word_gru_layers=1,
        word_attention_dimmentions=100,
        sentence_gru_hidden_units=50,
        sentence_gru_layers=1,
        sentence_attention_dimmention=100,
        number_of_classes=number_of_classes,
    )
    return model


# Data preprocessing utilities
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


# Example training loop
def trainset_han_model(model, trainset_loader, validationset_loader, number_of_epochs=10, lr=0.001):
    """
    Example training loop for HAN model.
    """
    trainer = HANTrainer(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    highest_validation_accuracy = 0.0

    for epoch in range(number_of_epochs):
        # Training
        model.train()
        trainset_loss = 0.0
        trainset_acc = 0.0
        num_batches = 0

        for batch in trainset_loader:
            loss, acc = trainer.trainset_step(batch, optimizer)
            trainset_loss += loss
            trainset_acc += acc
            num_batches += 1

        # Validation
        validationset_loss, validationset_accuracy = trainer.evaluate(validationset_loader)

        # Update learning rate
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch+1}/{number_of_epochs}:")
        print(
            f"  Trainingset Loss: {trainset_loss/num_batches:.4f}, Trainingset Accuracy: {trainset_acc/num_batches:.4f}"
        )
        print(f"  Validationset Loss: {validationset_loss:.4f}, Validationset Accuracy: {validationset_accuracy:.4f}")

        # Save best model
        if validationset_accuracy > highest_validation_accuracy:
            highest_validation_accuracy = validationset_accuracy
            torch.save(model.state_dict(), "best_han_model.pth")

    print(f"Best validation accuracy: {highest_validation_accuracy:.4f}")


if __name__ == "__main__":

    # Model parameters
    vocabulary_size = 10000
    embedding_dimmentions = 200
    number_of_classes = 5

    # Create model
    model = create_han_model(vocabulary_size, embedding_dimmentions, number_of_classes)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    batch_size = 2
    max_sentence_length = 3
    max_word_length = 5

    # Create dummy data
    documents = torch.randint(
        1, vocabulary_size, (batch_size, max_sentence_length, max_word_length)
    )
    word_lengths = torch.randint(
        1, max_word_length + 1, (batch_size, max_sentence_length)
    )
    sentence_lengths = torch.randint(1, max_sentence_length + 1, (batch_size,))

    print("\nTesting forward pass")
    with torch.no_grad():
        logits, word_att, sentence_att = model(
            documents, word_lengths, sentence_lengths
        )

    print(f"Output logits shape: {logits.shape}")
    print(f"Word attention weights shape: {word_att.shape}")
    print(f"Sentence attention weights shape: {sentence_att.shape}")
    print("\nHierarchical Attention Network implementation complete")
