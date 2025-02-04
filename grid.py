import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Paths to saved models and scaler
lstm_model_path = 'D:/Tugas Akhir/Hasil2/lstm_model.h5'
scaler_path = 'D:/Tugas Akhir/Hasil2/scaler.pkl'
svdd_model_path = 'D:/Tugas Akhir/Hasil2/svdd_model.pkl'
pca_model_path = 'D:/Tugas Akhir/Hasil2/pca_model.pkl'
adaptive_threshold_path = 'D:/Tugas Akhir/Hasil2/adaptive_threshold.npy'
results_save_path = 'D:/Tugas Akhir/Hasil2/inference/inference_results.xlsx'
figures_save_dir = 'D:/Tugas Akhir/Hasil2/inference/figures'
os.makedirs(figures_save_dir, exist_ok=True)

# Load models and scaler
lstm_model = tf.keras.models.load_model(lstm_model_path)
scaler = joblib.load(scaler_path)
svdd_model = joblib.load(svdd_model_path)
pca = joblib.load(pca_model_path)
adaptive_threshold = np.load(adaptive_threshold_path)

# Define features and sequence length (based on training configuration)
features = ['CHL_RW_TEMP_1', 'CHL_SW_TEMP_1', 'CHL_SWCD_TEMP_1', 
            'CHL_RWCD_TEMP_1', 'CT_RW_TEMP_1', 'CDWL_RW_TEMP', 'CDWL_SW_TEMP', 
            'CT_SW_TEMP_1']
time_step = 30

# Function to create sequences for LSTM
def create_sequences(data, time_step=30):
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
    return np.array(X)

# Function to run inference on new data
def detect_anomalies(data):
    # Scale and sequence the data
    scaled_data = scaler.transform(data[features])
    X_sequences = create_sequences(scaled_data, time_step)

    # LSTM model predictions
    y_pred_scaled = lstm_model.predict(X_sequences)
    
    # Reverse scaling to get residuals
    y_pred_rescaled = scaler.inverse_transform(y_pred_scaled)
    y_actual_rescaled = scaler.inverse_transform(scaled_data[time_step:])

    # Calculate residuals
    residuals = np.abs(y_actual_rescaled - y_pred_rescaled)

    # Apply PCA transformation to residuals
    residuals_pca = pca.transform(residuals)

    # SVDD prediction
    svdd_predictions = svdd_model.predict(residuals_pca)
    svdd_anomalies = np.where(svdd_predictions == -1, 1, 0)

    # Adaptive threshold
    max_residuals = np.max(residuals, axis=1)
    threshold_anomalies = np.where(max_residuals > adaptive_threshold[:len(max_residuals)], 1, 0)

    # Combine SVDD and threshold anomalies (AND operation)
    combined_anomalies = (svdd_anomalies & threshold_anomalies).astype(int)

    # Output anomaly results and index of detected anomalies
    anomaly_indices = np.where(combined_anomalies == 1)[0] + time_step  # Adjust for sequence start
    return svdd_anomalies, threshold_anomalies, combined_anomalies, anomaly_indices, residuals, max_residuals

# Function to evaluate and save results
def evaluate_and_save_results(data, svdd_anomalies, threshold_anomalies, combined_anomalies, residuals, max_residuals, output_path):
    # Ground truth (assuming a column 'Anomali' exists in the test data)
    ground_truth = data['Anomali'].values[time_step:]

    # Calculate metrics for SVDD, threshold, and combined methods
    evaluation_results = {
        "Method": ["SVDD", "Adaptive Threshold", "Combined"],
        "Accuracy": [
            accuracy_score(ground_truth, svdd_anomalies),
            accuracy_score(ground_truth, threshold_anomalies),
            accuracy_score(ground_truth, combined_anomalies)
        ],
        "Precision": [
            precision_score(ground_truth, svdd_anomalies),
            precision_score(ground_truth, threshold_anomalies),
            precision_score(ground_truth, combined_anomalies)
        ],
        "Recall": [
            recall_score(ground_truth, svdd_anomalies),
            recall_score(ground_truth, threshold_anomalies),
            recall_score(ground_truth, combined_anomalies)
        ],
        "F1-Score": [
            f1_score(ground_truth, svdd_anomalies),
            f1_score(ground_truth, threshold_anomalies),
            f1_score(ground_truth, combined_anomalies)
        ]
    }

    # Save evaluation results to Excel
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(evaluation_results).to_excel(writer, sheet_name='Evaluation Metrics', index=False)

        # Confusion Matrices
        conf_matrix_svdd = confusion_matrix(ground_truth, svdd_anomalies)
        conf_matrix_threshold = confusion_matrix(ground_truth, threshold_anomalies)
        conf_matrix_combined = confusion_matrix(ground_truth, combined_anomalies)

        # Save confusion matrices to separate sheets
        pd.DataFrame(conf_matrix_svdd, index=['Actual_Normal', 'Actual_Anomaly'],
                     columns=['Predicted_Normal', 'Predicted_Anomaly']).to_excel(writer, sheet_name='Confusion_Matrix_SVDD')
        pd.DataFrame(conf_matrix_threshold, index=['Actual_Normal', 'Actual_Anomaly'],
                     columns=['Predicted_Normal', 'Predicted_Anomaly']).to_excel(writer, sheet_name='Confusion_Matrix_Threshold')
        pd.DataFrame(conf_matrix_combined, index=['Actual_Normal', 'Actual_Anomaly'],
                     columns=['Predicted_Normal', 'Predicted_Anomaly']).to_excel(writer, sheet_name='Confusion_Matrix_Combined')

    print(f"Evaluation results saved successfully to {output_path}")

    # Plot residuals for each feature
    plt.figure(figsize=(14, 6))
    for i, feature in enumerate(features):
        plt.plot(residuals[:, i], label=f'Residuals - {feature}')
    plt.xlabel('Sample Index')
    plt.ylabel('Residual Value')
    plt.title('Residuals for Each Feature')
    plt.legend()
    plt.grid(True)
    residuals_plot_path = os.path.join(figures_save_dir, 'residuals.png')
    plt.savefig(residuals_plot_path)
    plt.close()
    print(f"Residuals plot saved to {residuals_plot_path}")

    # Plot max residuals vs adaptive threshold
    plt.figure(figsize=(10, 6))
    plt.plot(max_residuals, label='Max Residuals', color='blue')
    plt.plot(adaptive_threshold[:len(max_residuals)], label='Adaptive Threshold', color='red', linestyle='dashed')
    plt.title('Max Residuals vs Adaptive Threshold')
    plt.xlabel('Sample Index')
    plt.ylabel('Residual Value')
    plt.legend()
    plt.grid(True)
    threshold_plot_path = os.path.join(figures_save_dir, 'max_residuals_vs_threshold.png')
    plt.savefig(threshold_plot_path)
    plt.close()
    print(f"Threshold plot saved to {threshold_plot_path}")

    # Plot confusion matrices for each method
    for method, conf_matrix in zip(["SVDD", "Threshold", "Combined"],
                                   [conf_matrix_svdd, conf_matrix_threshold, conf_matrix_combined]):
        plt.figure(figsize=(6, 5))
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {method}')
        conf_matrix_path = os.path.join(figures_save_dir, f'conf_matrix_{method.lower()}.png')
        plt.savefig(conf_matrix_path)
        plt.close()
        print(f"{method} Confusion Matrix saved to {conf_matrix_path}")

# Example usage
# Load test data for inference
test_data_path = 'D:/Tugas Akhir/Dataset/February/Revised_Combined_Dataset_with_Anomalies.csv'
test_data = pd.read_csv(test_data_path)

# Detect anomalies in the test data
svdd_anomalies, threshold_anomalies, combined_anomalies, anomaly_indices, residuals, max_residuals = detect_anomalies(test_data)

# Evaluate and save results
evaluate_and_save_results(test_data, svdd_anomalies, threshold_anomalies, combined_anomalies, residuals, max_residuals, results_save_path)
