import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = None):
    """Enhanced version that includes metrics on the plot
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        title: Optional title for the plot
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=4, alpha=0.7)
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], 
             linestyle='--', color='red')
    plt.xlabel("Measured Values (Destructive)")
    plt.ylabel("Predicted Values")
    
    # Calculate metrics
    metrics = calculate_metrics(y_true, y_pred, plot=False)
    
    # Create text box with metrics
    textstr = '\n'.join((
        f'RMSE: {metrics["rmse"]:.3f}',
        f'R²: {metrics["r2"]:.3f}',
        f'SEP: {metrics["sep"]:.3f}',
        f'Bias: {metrics["bias"]:.3f}',
        f'RPD: {metrics["rpd"]:.3f}'
    ))
    
    # Place text box in upper left
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    plt.gca().text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
                  verticalalignment='top', bbox=props)
    
    if title:
        plt.title(title)
    
    plt.tight_layout()
    plt.show()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, plot: bool = True) -> dict:
    """Calculate SEP, RMSE, Bias, and RPD of predictions
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        plot: Whether to show the prediction plot
        
    Returns:
        Dictionary of calculated metrics
    """
    if plot:
        plot_predictions(y_true, y_pred)
        
    n = y_true.shape[0]
    rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
    y_error = y_true - y_pred
    mean_error = np.mean(y_error)
    std_error = np.sqrt(np.square(y_error - mean_error).sum() / (n-1))
    std_true = np.sqrt(np.square(y_true - y_true.mean()).sum() / (n-1))
    
    return {
        "r2": metrics.r2_score(y_true, y_pred),
        "rmse": rmse,
        "sep": std_error,
        "bias": mean_error,
        "rpd": std_true / std_error,
    }

    
# def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, plot: bool = True) -> dict:
#     """Calculate SEP, RMSE, Bias, and RPD of predictions

#     """
#     if plot:
#         plot_predictions(y_true, y_pred)
#     n = y_true.shape[0]
#     rmse = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
#     y_error = y_true - y_pred
#     mean_error = np.mean(y_error)
#     std_error = np.sqrt(np.square(y_error - mean_error).sum() / (n-1))
#     std_true = np.sqrt(np.square(y_true - y_true.mean()).sum() / (n-1))
#     return {
#         # calculate r-squared (R2)
#         "r2": metrics.r2_score(y_true, y_pred),

#         # calculate root mean square error (RMSE)
#         "rmse": rmse,

#         # calculate standard error of prediction (SEP)
#         "sep": std_error,

#         # calculate bias
#         "bias": mean_error,

#         # calculate ratio of performance to deviation (RPD)
#         "rpd": std_true / std_error,

#     }

# def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray):
#     plt.scatter(y_true, y_pred, s = 4, alpha=0.7)
#     plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], linestyle='--', color='red')
#     plt.xlabel("Measured Values (Destructive)")
#     plt.ylabel("Predicted Values")
#     plt.show()    



def plot_model_history(history: dict):
    plt.figure(figsize=(8, 4))
    plt.plot(history["loss"], label="Calibration set loss")
    if "val_loss" in history:
        plt.plot(history["val_loss"], label="Tunning set loss")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.legend()
    ax2 = plt.gca().twinx()
    ax2.plot(history["lr"], color="r", ls="--")
    ax2.set_ylabel("learning rate", color="r")
    plt.tight_layout()
    plt.show()
    
    
