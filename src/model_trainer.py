import torch
from torch.serialization import add_safe_globals
import sklearn.preprocessing._label as _l
from tqdm import tqdm

# Import the HAN model (assuming it's saved as han_model.py)
from src.han_model import HierarchicalAttentionNetwork, HANTrainer


class ModelTrainer:
    """Complete training pipeline for HAN model."""

    def __init__(self, vocabulary_size, number_of_classes, label_encoder, device=None):
        self.vocabulary_size = vocabulary_size
        self.number_of_classes = number_of_classes
        self.label_encoder = label_encoder
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # Model hyperparameters
        self.model_config = {
            "vocabulary_size": vocabulary_size,
            "embedding_dimmentions": 200,
            "word_gru_hidden_units": 50,
            "word_gru_layers": 1,
            "word_attention_dimmentions": 100,
            "sentence_gru_hidden_units": 50,
            "sentence_gru_layers": 1,
            "sentence_attention_dimmention": 100,
            "number_of_classes": number_of_classes,
        }

        # Create model
        self.model = HierarchicalAttentionNetwork(**self.model_config)
        self.trainer = HANTrainer(self.model, self.device)

        print(
            f"Model created with {sum(p.numel() for p in self.model.parameters()):,} parameters"
        )
        print(f"Using device: {self.device}")

    def train(
        self,
        trainset_loader,
        validationset_loader,
        number_of_epochs=10,
        learning_rate=0.001,
        save_path="best_han_model.pth",
        patience=3,
    ):
        # Training parameters
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)

        # Tracking best validation accuracy
        highest_validation_accuracy = 0.0
        patience_counter = 0
        trainingset_history = []
        validationset_history = []

        print("Training started")
        print(f"Training samples:   {len(trainset_loader.dataset)}")
        print(f"Validation samples: {len(validationset_loader.dataset)}")

        for epoch in range(number_of_epochs):
            # 1) Training phase
            self.model.train()
            trainset_loss = 0.0
            trainset_acc = 0.0
            num_batches = 0

            progress_bar = tqdm(trainset_loader, desc=f"Epoch {epoch+1}/{number_of_epochs}")
            for batch in progress_bar:
                loss, acc = self.trainer.trainset_step(batch, optimizer)
                trainset_loss += loss
                trainset_acc += acc
                num_batches += 1

                progress_bar.set_postfix({"Loss": f"{loss:.4f}", "Acc": f"{acc:.4f}"})

            avg_trainset_loss = trainset_loss / num_batches
            avg_trainset_acc = trainset_acc / num_batches

            # 2) Validation phase
            validationset_loss, validationset_accuracy = self.trainer.evaluate(validationset_loader)

            # 3) Scheduler step
            scheduler.step()

            # 4) Record history
            trainingset_history.append(
                {"loss": avg_trainset_loss, "accuracy": avg_trainset_acc}
            )
            validationset_history.append({"loss": validationset_loss, "accuracy": validationset_accuracy})

            # 5) Print epoch metrics
            print(f"\nEpoch {epoch+1}/{number_of_epochs}:")
            print(f"  Train Loss: {avg_trainset_loss:.4f}, Train Acc: {avg_trainset_acc:.4f}")
            print(f"  Val   Loss: {validationset_loss:.4f}, Val   Acc: {validationset_accuracy:.4f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # 6) Check for improvement
            if validationset_accuracy > highest_validation_accuracy:
                highest_validation_accuracy = validationset_accuracy
                patience_counter = 0

                # Save model checkpoint only if a valid path is provided
                if save_path is not None:
                    checkpoint = {
                        "model_state_dict": self.model.state_dict(),
                        "model_config": self.model_config,
                        "highest_validation_accuracy": highest_validation_accuracy,
                        "epoch": epoch + 1,
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "label_encoder": self.label_encoder,  # ensure this exists on self
                    }
                    torch.save(checkpoint, save_path)
                    print(f"New best model saved Val Acc: {validationset_accuracy:.4f}")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{patience}")

            # 7) Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

            print()

        print(
            f"Training completed with the best validation accuracy being: {highest_validation_accuracy:.4f}"
        )

        # 8) Load best model only if save_path is not None
        if save_path is not None:
            self.load_model(save_path)

        return {
            "highest_validation_accuracy": highest_validation_accuracy,
            "trainingset_history": trainingset_history,
            "validationset_history": validationset_history,
            "total_epochs": epoch + 1,
        }

    def evaluate(self, test_loader):
        """Evaluate model on test set."""
        print("Evaluating on test set...")
        testset_loss, testset_accuracy = self.trainer.evaluate(test_loader)
        print(
            f"Testset Loss: {testset_loss:.4f}, Testset Accuracy: {testset_accuracy:.4f}"
        )
        return testset_loss, testset_accuracy

    def load_model(self, model_path):
        """Load model from checkpoint."""
        checkpoint = torch.load(
            model_path, map_location=self.device, weights_only=False
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Model loaded from {model_path}")
        return checkpoint

    def save_model(self, save_path, additional_data=None):
        """Save current model state."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "model_config": self.model_config,
        }

        if additional_data:
            checkpoint.update(additional_data)

        torch.save(checkpoint, save_path)
        print(f"Model saved to {save_path}")


def load_trained_model(model_path="best_han_model.pth", device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    add_safe_globals([_l.LabelEncoder])
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Create model with saved configuration
    model = HierarchicalAttentionNetwork(**checkpoint["model_config"])

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded from {model_path}")
    print(
        f"Highest validation accuracy: {checkpoint.get('highest_validation_accuracy', 'N/A')}"
    )

    return model, checkpoint["model_config"], checkpoint


def predict_single_text(
    model,
    preprocessor,
    label_encoder,
    text,
    device=None,
    max_sentences=20,
    max_words_per_sentence=50,
):
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Import here to avoid circular imports
    from src.data_preprocessor import create_padding_for_hierarchical_sequences

    # Preprocess text
    hierarchical_doc = preprocessor.text_to_hierarchical_indices(
        text, max_sentences, max_words_per_sentence
    )

    # Pad sequences
    padded_docs, word_lengths, sentence_lengths = create_padding_for_hierarchical_sequences(
        [hierarchical_doc]
    )

    # Convert to tensors
    documents = torch.tensor(padded_docs, dtype=torch.long).to(device)
    word_lengths = torch.tensor(word_lengths, dtype=torch.long).to(device)
    sentence_lengths = torch.tensor(sentence_lengths, dtype=torch.long).to(device)

    with torch.no_grad():
        logits, word_attention_weights, sentence_attention_weights = model(
            documents, word_lengths, sentence_lengths
        )

        # Get prediction
        probabilities = torch.softmax(logits, dim=1)
        predicted_label_idx = torch.argmax(probabilities, dim=1).item()
        predicted_label = label_encoder.inverse_transform([predicted_label_idx])[0]
        confidence = probabilities[0, predicted_label_idx].item()

    return {
        "predicted_label": predicted_label,
        "confidence": confidence,
        "probabilities": probabilities[0].cpu().numpy(),
        "word_attention": word_attention_weights[0].cpu().numpy(),
        "sentence_attention": sentence_attention_weights[0].cpu().numpy(),
        "all_labels": label_encoder.classes_,
    }


def get_model_predictions(model, data_loader, label_encoder, device=None):
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    all_predictions = []
    all_true_labels = []
    all_probabilities = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Getting predictions"):
            documents, word_lengths, sentence_lengths, labels = batch

            # Move to device
            documents = documents.to(device)
            word_lengths = word_lengths.to(device)
            sentence_lengths = sentence_lengths.to(device)

            # Forward pass
            logits, _, _ = model(documents, word_lengths, sentence_lengths)

            # Get predictions and probabilities
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_true_labels.extend(labels.numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # Convert to label names
    predicted_labels = label_encoder.inverse_transform(all_predictions)
    true_labels = label_encoder.inverse_transform(all_true_labels)

    return {
        "predictions": predicted_labels,
        "true_labels": true_labels,
        "probabilities": all_probabilities,
        "prediction_indices": all_predictions,
        "true_label_indices": all_true_labels,
    }
