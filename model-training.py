# Step 1: Install Required Libraries
#!pip install transformers torch scikit-learn

# Step 2: Import Libraries and Load Data
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Load the data
file_path = "resume_job_matching_data.csv"
df = pd.read_csv(file_path)

# Check the data structure
print(df.head())

# Step 3: Prepare the Dataset
class ResumeJobDataset(Dataset):
    def __init__(self, job_descriptions, resumes, labels, tokenizer, max_length=128):
        self.job_descriptions = job_descriptions
        self.resumes = resumes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        job_desc = str(self.job_descriptions[idx])
        resume = str(self.resumes[idx])
        label = self.labels[idx]

        # Tokenize job description and resume together
        inputs = self.tokenizer(
            job_desc,
            resume,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Flatten to remove extra dimension
        input_ids = inputs["input_ids"].squeeze()
        attention_mask = inputs["attention_mask"].squeeze()

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Initialize the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Prepare the data
job_descriptions = df["Job Description"].tolist()
resumes = df["Resume Excerpt"].tolist()
labels = df["Match Label"].tolist()

# Split data into training and validation sets
train_descriptions, val_descriptions, train_resumes, val_resumes, train_labels, val_labels = train_test_split(
    job_descriptions, resumes, labels, test_size=0.2, random_state=42
)

# Create dataset instances
train_dataset = ResumeJobDataset(train_descriptions, train_resumes, train_labels, tokenizer)
val_dataset = ResumeJobDataset(val_descriptions, val_resumes, val_labels, tokenizer)

# Step 4: Set Up the Model and Training Arguments
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Disable W&B logging to avoid needing an API key
import os
os.environ["WANDB_DISABLED"] = "true"

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    eval_strategy="epoch",  # Changed to eval_strategy to avoid warning
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    run_name="resume_matching"  # Custom name for the run
)

# Define a compute_metrics function to calculate accuracy, precision, recall, and F1 score
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# Step 5: Initialize the Trainer with compute_metrics and Start Training
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics  # Added this line
)

trainer.train()

# Step 6: Evaluate the Model
results = trainer.evaluate()
print("Evaluation Results:", results)  # Updated to print all metrics

# Step 7: Save the Model
model.save_pretrained("/content/saved_model")
tokenizer.save_pretrained("/content/saved_model")

# Updated predict_match function to handle device compatibility
def predict_match(job_desc, resume):
    # Move model to the appropriate device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Tokenize input and move tensors to the same device as the model
    inputs = tokenizer(
        job_desc,
        resume,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=128
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Perform prediction
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    return "Match" if prediction == 1 else "No Match"

# Example usage
job_desc = "Software developer with experience in Python and machine learning."
resume = "Experienced Python developer skilled in data analysis and ML frameworks."
print(predict_match(job_desc, resume))
