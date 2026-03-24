import torch
import torch.nn as nn
import torch.optim as optim
from src.config import LEARNING_RATE, NUM_EPOCHS

def train_model(model, train_loader):
    """Trains the model and returns the loss history across epochs."""
    
    # Binary Cross Entropy is the standard for Sigmoid output
    criterion = nn.BCELoss()
    
    # Adam optimizer works optimally generally
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_history = []
    
    # Ensure model is in training mode (enables dropout)
    model.train()
    
    print(f"Starting Training for {NUM_EPOCHS} Epochs...")
    
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            
            # Clear old gradients
            optimizer.zero_grad()
            
            # Make prediction
            outputs = model(inputs)
            
            # Calculate loss against true labels
            loss = criterion(outputs, labels)
            
            # Find Errors (Backpropagation)
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        loss_history.append(avg_loss)
        
        # Output progress cleanly
        if (epoch + 1) % 10 == 0:
            print(f"  > Epoch [{epoch + 1:3d}/{NUM_EPOCHS}], Training Loss: {avg_loss:.4f}")
            
    print("Model Training Successfully Completed!")
    return loss_history
