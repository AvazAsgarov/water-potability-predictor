import torch

def evaluate_model(model, test_loader):
    """Calculates and prints the accuracy of the model on unseen test data."""
    # Ensure model is in evaluation mode (disables dropout)
    model.eval()
    
    correct = 0
    total = 0
    
    # Turn off gradient tracking to speed up processing and save memory
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            
            # Output >= 0.5 means predicting '1' (Potable), otherwise '0'
            predicted = (outputs >= 0.5).float()
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"=====================================")
    print(f"  Final Evaluation Accuracy: {accuracy:.2f}%")
    print(f"=====================================")
    
    return accuracy
