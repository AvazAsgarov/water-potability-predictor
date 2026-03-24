import seaborn as sns
import matplotlib.pyplot as plt
from src.config import ASSETS_DIR

def plot_class_balance(df):
    """Plots and saves the potability class distribution."""
    plt.figure(figsize=(8, 6))
    
    # Using a modern palette and styling
    sns.set_theme(style="whitegrid")
    sns.countplot(x='Potability', data=df, palette='Set2', hue='Potability')
    
    plt.title('Distribution of Potability (0 = No, 1 = Yes)', fontsize=14, pad=15)
    plt.xlabel('Potability', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'potability_balance.png', dpi=300)
    plt.close()

def plot_correlation_heatmap(df):
    """Plots and saves the feature correlation heatmap."""
    plt.figure(figsize=(10, 8))
    
    # Warm/cool color scheme for clear correlations
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', 
                linewidths=0.5, annot_kws={"size": 10})
    
    plt.title('Feature Correlation Heatmap', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'correlation_heatmap.png', dpi=300)
    plt.close()

def plot_feature_distributions(df):
    """Plots and saves a grid of feature histograms."""
    # Exclude Potability from distributions to focus purely on features
    features_df = df.drop('Potability', axis=1)
    
    features_df.hist(figsize=(14, 12), bins=30, color='#4C72B0', edgecolor='black')
    plt.suptitle('Distribution of Water Quality Features', fontsize=18, y=1.02)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'feature_distributions.png', dpi=300)
    plt.close()

def plot_training_loss(loss_history):
    """Plots and saves the loss trajectory during training."""
    plt.figure(figsize=(10, 6))
    
    plt.plot(loss_history, marker='o', linestyle='-', color='#C44E52')
    plt.title('Training Loss over Epochs', fontsize=16)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Binary Cross Entropy)', fontsize=12)
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / 'training_loss.png', dpi=300)
    plt.close()

def generate_all_visuals(df):
    """Helper method to generate exploratory data analysis visuals."""
    print("Generating exploratory visualizations in assets/...")
    plot_class_balance(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    print("Visualizations successfully saved.")
