import torch
import torch.nn as nn
import torch.optim as optim
import math
from transformers import GPTNeoForCausalLM, AutoTokenizer
from datasets import load_dataset #if this doesn't load try pip install datasets

# Set pad token as eos token to handle padding correctly
tokenizer.pad_token = tokenizer.eos_token

# Load GPT-Neo 1.3B model and tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)  # FP16 helps save VRAM

# Move the model to GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load a small sample of the OpenWebText dataset (100 samples)
dataset = load_dataset("openwebtext", split="train")
data = [sample["text"] for sample in dataset.select(range(100))]

# Function to calculate perplexity (a measure of text generation quality)
def calculate_perplexity(model, data, max_seq_length=64):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    
    with torch.no_grad():  # No need to track gradients for validation
        for text in data:
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length).to(device)
            labels = inputs["input_ids"].clone()  # Use input_ids as labels for causal LM
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss  # Compute the loss (negative log-likelihood)
            total_loss += loss.item()  # Add to total loss

    avg_loss = total_loss / len(data)  # Calculate average loss
    perplexity = math.exp(avg_loss)  # Convert loss to perplexity
    return perplexity

# Calculate perplexity before fine-tuning
perplexity_before = calculate_perplexity(model, data)
print(f"📌 Perplexity Before Fine-Tuning: {perplexity_before:.4f}")

# Fine-tuning function
from tqdm import tqdm  # Import tqdm for a progress bar

# Set pad token for consistency
tokenizer.pad_token = tokenizer.eos_token

def fine_tune(model, data, epochs=1, lr=3e-5, batch_size=1, gradient_accumulation_steps=8, max_seq_length=64):
    model.train()  # Set model to training mode
    optimizer = optim.AdamW(model.parameters(), lr=lr)  # Initialize optimizer (AdamW)
    
    # Loop over epochs for fine-tuning
    for epoch in range(epochs):
        total_loss = 0
        optimizer.zero_grad()  # Reset gradients at the start of each epoch

        # Loop through each batch of data
        progress_bar = tqdm(enumerate(data), total=len(data), desc=f"Epoch {epoch+1}/{epochs}")

        for i, text in progress_bar:
            inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=max_seq_length).to(device)
            labels = inputs["input_ids"].clone()  # Use input_ids as labels for causal LM
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps  # Scale loss by gradient accumulation steps

            total_loss += loss.item()  # Add loss for this batch
            loss.backward()  # Compute gradients

            # Update model weights after every gradient_accumulation_steps
            if (i + 1) % gradient_accumulation_steps == 0 or i == len(data) - 1:
                optimizer.step()  # Perform a weight update
                optimizer.zero_grad()  # Reset gradients

                avg_loss = total_loss / gradient_accumulation_steps  # Compute average loss
                progress_bar.set_postfix(loss=f"{avg_loss:.4f}")  # Display current loss
                total_loss = 0  # Reset total loss for next accumulation

    return model

# Fine-tune GPT-Neo model
fine_tune(model, data, epochs=1, batch_size=1, gradient_accumulation_steps=8, max_seq_length=128)

# Calculate perplexity after fine-tuning
perplexity_after = calculate_perplexity(model, data)
print(f"✅ Perplexity After Fine-Tuning: {perplexity_after:.4f}")
