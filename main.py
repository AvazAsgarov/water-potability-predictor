from src.dataset import get_dataloaders
from src.visualize import generate_all_visuals, plot_training_loss
from src.model import WaterModel
from src.train import train_model
from src.evaluate import evaluate_model

def main():
    print("--- Water Potability Deep Learning Pipeline ---")
    
    # 1. Provide DataLoaders and the raw Dataframe
    train_loader, test_loader, raw_df = get_dataloaders()
    print("Data loaded and scaled successfully.")
    
    # 2. Generate Exploratory Visualizations
    generate_all_visuals(raw_df)
    
    # 3. Initialize Model
    model = WaterModel()
    
    # 4. Train Model
    loss_history = train_model(model, train_loader)
    
    # 5. Plot Training Trajectory
    plot_training_loss(loss_history)
    print("Training loss curve saved to assets.")
    
    # 6. Evaluate final model
    evaluate_model(model, test_loader)
    
if __name__ == "__main__":
    main()
