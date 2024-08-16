# Standard library imports
import os
import json
import gzip
import re
import random

# Third-party library imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from umap import UMAP
from sklearn.model_selection import train_test_split

# PyTorch and Transformers imports
import torch
from torch.utils.data import Dataset
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, get_linear_schedule_with_warmup
from torch.optim import AdamW

# Other utility imports
from collections import Counter

# Constants
DOMAINS = ['Sports_and_Outdoors']#, 'Movies_and_TV', 'Neutral', 'Pet_Supplies', 'Automotive', 'Electronics']
NUM_DOMAINS = len(DOMAINS)
SCORES = [1, 2, 3, 4, 5]
NUM_SCORES = len(SCORES)

# General Parameters
LIMIT_READ = 80
STAT_SIG = 1
MAX_LENGTH = 512

# Training
NUM_EPOCHS = 1
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.01
LEARNING_RATE = 5e-5

# Logging and Checkpoints
LOGGING_STEPS = 50
EVAL_STEPS = 50

# Paths
dataset_base_path = r"G:\My Drive\MSc\Semester 2\NLP\Final Project\Datasets"
trained_model_save_directory = r"G:\My Drive\MSc\Semester 2\NLP\Final Project\Trained_models"

# Determine the device and print it for the user
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class ReviewsDataset(Dataset):
    """
    Custom Dataset class for handling tokenized text data and corresponding labels.
    Inherits from torch.utils.data.Dataset.
    """

    def __init__(self, encodings, scores):
        """
        Initializes the DataLoader class with encodings and labels.

        Args:
            encodings (dict): A dictionary containing tokenized input text data
                              (e.g., 'input_ids', 'token_type_ids', 'attention_mask').
            scores (list): A list of integer labels for the input text data.
        """

        self.encodings = encodings
        self.labels = scores

    def __getitem__(self, idx):
        """
        Returns a dictionary containing tokenized data and the corresponding label for a given index.

        Args:
            idx (int): The index of the data item to retrieve.

        Returns:
            item (dict): A dictionary containing the tokenized data and the corresponding label.
        """
        # Retrieve tokenized data for the given index
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # Add the label for the given index to the item dictionary
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        """
        Returns the number of data items in the dataset.

        Returns:
            (int): The number of data items in the dataset.
        """
        return len(self.labels)


def read_gzip_json(file_path, limit, num_scores=5):
    reviews, scores = [], []
    score_counts = Counter()
    max_reviews_per_score = limit // num_scores  # Calculate the maximum number of reviews per score

    with gzip.open(file_path, 'rb') as f:
        for line in tqdm(f, desc=f"Loading {file_path}", unit=" lines"):
            data = json.loads(line.decode('utf-8'))
            if 'reviewText' in data and 'overall' in data:
                review = data['reviewText']
                score = data['overall'] - 1  # Adjust score to be zero-indexed
                if score < 0 or score >= NUM_SCORES:
                    raise ValueError(f"Score out of range")

                if score_counts[score] < max_reviews_per_score:
                    reviews.append(review)
                    scores.append(score)
                    score_counts[score] += 1

                if len(reviews) >= limit:
                    break

    return reviews, scores


def create_reviews_and_labels():
    reviews_and_labels_local = {}

    for domain in DOMAINS:
        runs = {}
        for run in range(STAT_SIG):
            slot = {
                "reviews": [],
                "scores": []
            }
            runs[run] = slot
        reviews_and_labels_local[domain] = runs

    for domain in DOMAINS:

        if domain == 'Neutral':
            continue

        dataset_path = os.path.join(dataset_base_path,
                                    f'{domain}_5.json.gz')  # Update this with your actual file naming convention
        reviews, scores = read_gzip_json(dataset_path, LIMIT_READ * STAT_SIG)

        combined = list(zip(reviews, scores))
        random.shuffle(combined)
        reviews[:], scores[:] = zip(*combined)

        idx = 0
        for run in range(STAT_SIG):
            for _ in range(LIMIT_READ):
                if idx >= len(reviews):
                    break
                reviews_and_labels_local[domain][run]["reviews"].append(reviews[idx])
                reviews_and_labels_local[domain][run]["scores"].append(scores[idx])
                idx += 1

    return reviews_and_labels_local


def create_layers_weigths():
    weights_local = {}
    for domain in DOMAINS:
        for run in range(STAT_SIG):
            model = fine_tune_bert(domain, run)

            model_weights = {}
            pattern = r'^bert\.encoder\.layer\.\d+\.attention\.self\.value\.weight$'
            for name, param in model.named_parameters():
                if re.match(pattern, name):
                    model_weights[name] = param.detach().cpu().numpy()

            # Flatten weights and add them to the DomainWeights instance
            for layer, atten_vals in model_weights.items():
                flattened_weights = atten_vals.reshape(1, -1)  # shape as (sample, features)
                if layer not in weights_local:
                    weights_local[layer] = {}
                if domain not in weights_local[layer]:
                    weights_local[layer][domain] = {}
                weights_local[layer][domain][run] = flattened_weights
    return weights_local


def check_label_distribution(scores, domain, run, threshold=10):
    label_counts = Counter(scores)
    total_samples = len(scores)
    label_distribution = {label: count / total_samples * 100 for label, count in label_counts.items()}

    print(f"Label distribution for domain '{domain}', run '{run}':")
    for label, percentage in label_distribution.items():
        print(f"Label {label}: {percentage:.2f}% ({label_counts[label]} samples)")

    # Plotting the distribution
    labels = list(label_distribution.keys())
    percentages = list(label_distribution.values())
    bar_width = 0.5  # Adjust this value as needed

    fig, ax = plt.subplots()
    bars = ax.bar(labels, percentages, width=bar_width)

    # Set x-ticks to be integers
    ax.set_xticks(labels)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_xlabel('Label')
    ax.set_ylabel('Percentage of Samples')
    ax.set_title(f'Distribution of Labels for {domain} - Run {run}')
    ax.grid(True, which='both', linestyle='--', linewidth=0.7)

    # Optionally, add value labels on top of the bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval:.2f}%', va='bottom')  # va: vertical alignment

    plt.show()

    max_percentage = max(label_distribution.values())
    min_percentage = min(label_distribution.values())
    is_balanced = max_percentage - min_percentage <= threshold
    if is_balanced:
        print(
            f"The dataset is balanced. The difference between the most and least frequent labels is {max_percentage - min_percentage:.2f}%.")
    else:
        print(
            f"The dataset is imbalanced. The difference between the most and least frequent labels is {max_percentage - min_percentage:.2f}%.")


def compute_metrics(pred):
    """
    Args:
        pred (obj): An object containing label_ids and predictions attributes.
            - label_ids (array-like): A 1D array of true class labels.
            - predictions (array-like): A 2D array where each row represents
              an observation, and each column represents the probability of
              that observation belonging to a certain class.
    """
    # Extract true labels from the input object
    labels = pred.label_ids

    # Obtain predicted class labels by finding the column index with the maximum probability
    preds = pred.predictions.argmax(-1)

    # Compute macro precision, recall, and F1 score using sklearn's precision_recall_fscore_support function
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')

    # Calculate the accuracy score using sklearn's accuracy_score function
    acc = accuracy_score(labels, preds)

    # Return the computed metrics as a dictionary
    return {
        'Accuracy': acc,
        'F1': f1,
        'Precision': precision,
        'Recall': recall
    }


def fine_tune_bert(domain, run):
    model_save_path = os.path.join(trained_model_save_directory, f'bert_fine_tuned_on_{domain}_run_{run}')

    if os.path.exists(model_save_path):
        print(f"Loading existing model from {model_save_path}")
        model = BertForSequenceClassification.from_pretrained(model_save_path)
    else:
        print(f"Fine-tuning BERT model on {domain}, run {run}...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', max_length=MAX_LENGTH)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=NUM_SCORES, problem_type="single_label_classification")
        model.to(device)

        reviews, scores = reviews_and_labels[domain][run].values()
        if len(set(scores)) != len(set(SCORES)):
            raise AssertionError("len(set(scores)) != len(set(SCORES))")

        # Check the distribution of the scores
        check_label_distribution(scores, domain, run)

        reviews_train, reviews_val, scores_train, scores_val = train_test_split(reviews, scores, test_size=0.2)

        train_encodings = tokenizer(reviews_train, truncation=True, padding=True)
        val_encodings = tokenizer(reviews_val, truncation=True, padding=True)

        train_dataloader = ReviewsDataset(train_encodings, scores_train)
        val_dataloader = ReviewsDataset(val_encodings, scores_val)

        # Initialize the optimizer
        optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

        training_args = TrainingArguments(
            # The output directory where the model predictions and checkpoints will be written
            output_dir=trained_model_save_directory,
            do_train=True,
            do_eval=True,
            #  The number of epochs, defaults to 3.0
            num_train_epochs=NUM_EPOCHS,
            per_device_train_batch_size=TRAIN_BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            # Number of steps used for a linear warmup
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
            logging_strategy='steps',
            # TensorBoard log directory
            logging_dir=trained_model_save_directory + '/multi-class-logs',
            logging_steps=LOGGING_STEPS,
            eval_strategy="steps",
            eval_steps=EVAL_STEPS,
            save_strategy="steps",
            fp16=True,
            load_best_model_at_end=True,
        )

        trainer = Trainer(
            # the pre-trained model that will be fine-tuned
            model=model,
            # training arguments that we defined above
            args=training_args,
            train_dataset=train_dataloader,
            eval_dataset=val_dataloader,
            compute_metrics=compute_metrics,
            optimizers=(optimizer, None),
        )

        trainer.train()

        trainer.save_model(model_save_path)
        print(f"Model saved successfully to {model_save_path}")

    return model


# Create data structure to hold all layers weights
reviews_and_labels = create_reviews_and_labels()
weights = create_layers_weigths()
