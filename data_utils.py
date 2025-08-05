# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Copyright (c) 2025 Lumina Mescuwa
# This file is licensed under the PolyForm Noncommercial License.
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
import sys
from pathlib import Path


def load_imdb_dataset() -> Dataset:
    """Loads the IMDb dataset from the Hugging Face hub."""
    return load_dataset("imdb")


def get_tokenizer(model_name: str = "bert-base-uncased") -> AutoTokenizer:
    """Initializes and returns a tokenizer from the Hugging Face hub.

    Args:
        model_name (str): The name of the pre-trained model whose tokenizer
            should be used.

    Returns:
        An `AutoTokenizer` instance.
    """
    return AutoTokenizer.from_pretrained(model_name)


def preprocess_data(dataset: Dataset, tokenizer: AutoTokenizer, max_length: int = 128) -> Dataset:
    """Tokenizes, pads/truncates, and formats a dataset for PyTorch.

    Args:
        dataset: A `datasets.Dataset` object (e.g., a train or test split).
        tokenizer: A `transformers` tokenizer instance.
        max_length: The maximum sequence length for padding and truncation.

    Returns:
        A processed `datasets.Dataset` object ready for a PyTorch DataLoader.
    """
    def tokenize_function(examples):
        # Tokenize the text, truncating to max_length and padding to max_length.
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    # Apply the tokenization function in a batched manner for efficiency.
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    # Remove the original 'text' column as it's no longer needed.
    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    # Set the format to 'torch' to get PyTorch tensors when iterating.
    tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    return tokenized_dataset


def get_dataloaders(batch_size: int, max_length: int, model_name: str, num_workers: int = 0
                    ) -> tuple[DataLoader, DataLoader, int, int]:
    """Loads and preprocesses the IMDb dataset and returns PyTorch DataLoaders.

    This function handles loading, splitting, tokenizing, and wrapping the
    IMDb dataset into training and validation DataLoaders. It uses a stratified
    split to ensure class balance in the validation set.

    Args:
        batch_size (int): The batch size for the DataLoaders.
        max_length (int): The maximum sequence length for tokenization.
        model_name (str): The name of the tokenizer model to use.
        num_workers (int): The number of worker processes for data loading.

    Returns:
        A tuple containing:
        - The training DataLoader.
        - The validation DataLoader.
        - The vocabulary size of the tokenizer.
        - The pad token ID of the tokenizer.
    """
    print("Loading IMDb dataset...")
    raw_datasets = load_imdb_dataset()

    print("Splitting training data into train/validation subsets...")
    # Use a stratified split to preserve the same class balance (positive/negative reviews)
    # in both the training and validation sets. This leads to more reliable validation metrics.
    train_test_split = raw_datasets['train'].train_test_split(
        test_size=0.1,
        shuffle=True,
        seed=42,
        stratify_by_column="label"
    )
    train_dataset_raw = train_test_split["train"]
    val_dataset_raw = train_test_split["test"]

    print("Initializing tokenizer...")
    tokenizer = get_tokenizer(model_name)

    print("Preprocessing datasets...")
    train_dataset_processed = preprocess_data(train_dataset_raw, tokenizer, max_length)
    validation_dataset_processed = preprocess_data(val_dataset_raw, tokenizer, max_length)

    print(f"Creating DataLoaders with num_workers={num_workers}...")
    # Use pinned memory if a GPU is available for faster CPU-to-GPU data transfers.
    pin_memory = torch.cuda.is_available()
    # Use persistent workers if multi-processing to avoid re-initializing workers each epoch.
    persistent_workers = num_workers > 0

    train_dataloader = DataLoader(
        train_dataset_processed,
        batch_size=batch_size,
        shuffle=True,  # Shuffle the training data each epoch.
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    validation_dataloader = DataLoader(
        validation_dataset_processed,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle validation data.
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )

    print("DataLoaders created successfully.")
    return train_dataloader, validation_dataloader, tokenizer.vocab_size, tokenizer.pad_token_id


# This block allows the script to be run directly to pre-tokenize and save the dataset,
# which can speed up subsequent training runs by avoiding repeated preprocessing.
if __name__ == '__main__':
    if "--save_tokenized" in sys.argv:
        try:
            save_dir_str = sys.argv[sys.argv.index("--save_tokenized") + 1]
            save_dir = Path(save_dir_str)
            print(f"Attempting to save tokenized datasets to: {save_dir}")

            # Use reasonable defaults for offline tokenization.
            DEFAULT_MAX_LEN = 512
            DEFAULT_TOKENIZER = "bert-base-uncased"
            print(f"Using defaults: max_length={DEFAULT_MAX_LEN}, tokenizer={DEFAULT_TOKENIZER}")

            # Perform the same loading and splitting as in get_dataloaders.
            ds = load_imdb_dataset()
            train_val = ds["train"].train_test_split(test_size=0.1, shuffle=True, seed=42, stratify_by_column="label")
            tok = get_tokenizer(DEFAULT_TOKENIZER)

            # Preprocess and save each split.
            train_ds_tokenized = preprocess_data(train_val["train"], tok, DEFAULT_MAX_LEN)
            val_ds_tokenized = preprocess_data(train_val["test"], tok, DEFAULT_MAX_LEN)

            save_dir.mkdir(parents=True, exist_ok=True)
            train_ds_tokenized.save_to_disk(save_dir / "train")
            val_ds_tokenized.save_to_disk(save_dir / "val")
            print(f"Successfully saved tokenized datasets to -> {save_dir}")
        except IndexError:
            print("Error: --save_tokenized requires a directory path argument.")
        except Exception as e:
            print(f"An error occurred during offline tokenization: {e}")
    else:
        # Example usage to demonstrate the dataloader functionality.
        print("Running example usage of get_dataloaders...")
        train_loader, val_loader, vocab_size, _ = get_dataloaders(batch_size=8, max_length=128, model_name="bert-base-uncased")
        print(f"Number of training batches: {len(train_loader)}")
        print(f"Number of validation batches: {len(val_loader)}")

        # Inspect a single batch to verify its contents.
        sample_batch = next(iter(train_loader))
        print("\nSample batch shapes:")
        print("  Input IDs:", sample_batch["input_ids"].shape)
        print("  Attention Mask:", sample_batch["attention_mask"].shape)
        print("  Labels:", sample_batch["label"].shape) 