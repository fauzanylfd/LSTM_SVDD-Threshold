import sys
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
import joblib

# Paths and Directories
file_path = 'D:/Tugas Akhir/Dataset/ChillerPlant_January_anomali.csv'
save_dir = 'D:/Tugas Akhir/Hasil14'
os.makedirs(save_dir, exist_ok=True)  # Ensure save directory exists

# Load the training data
train_data = pd.read_csv(file_path)

# Define features and preprocess the training data
features = ['CHL_RW_TEMP_1', 'CHL_SW_TEMP_1', 'CHL_SWCD_TEMP_1', 
            'CHL_RWCD_TEMP_1', 'CT_RW_TEMP_1', 'CDWL_RW_TEMP', 'CDWL_SW_TEMP', 
            'CT_SW_TEMP_1']
ground_truth_label = 'Anomali'

# Separate normal data
train_data_normal = train_data[train_data[ground_truth_label] == 0]

# Scaling the features using StandardScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data_normal[features])

# Function to create sequences for LSTM
def create_sequences(data, time_step=25):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step])
    return np.array(X), np.array(y)

time_step = 25
X_train, y_train = create_sequences(scaled_train_data, time_step)

# 5. Build and compile the LSTM model
lstm_model = Sequential([
    LSTM(150, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    LSTM(100, return_sequences=False),
    Dropout(0.3),
    Dense(50),
    Dense(25),
    Dense(len(features))  # Output layer
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the LSTM model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = lstm_model.fit(
    X_train, y_train, 
    epochs=100, 
    batch_size=32, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss Over Epochs', fontsize=15)
plt.xlabel('Epoch', fontsize=15)
plt.ylabel('Loss', fontsize=15)
plt.xticks(fontsize=12)  # Increase font size for x-axis tick labels
plt.yticks(fontsize=12)  # Increase font size for y-axis tick labels
plt.legend(fontsize=12)  # Set legend font size for better readability
plt.grid(True, linestyle="--", alpha=0.7)  # Optional: dashed grid lines for easier reading


# Save the training loss plot
training_loss_plot_filename = os.path.join(save_dir, 'training_validation_loss.png')
plt.savefig(training_loss_plot_filename)
plt.close()

print(f"Training and validation loss plot saved successfully to {training_loss_plot_filename}")

# Load and preprocess test data
test_file_path = 'D:/Tugas Akhir/Dataset/February/Anomaly_Applied_Data.csv'
test_data = pd.read_csv(test_file_path)
scaled_test_data = scaler.transform(test_data[features])
X_test, y_test = create_sequences(scaled_test_data, time_step)

# Predict the test data using the LSTM model and calculate residuals
y_pred = lstm_model.predict(X_test)
y_test_rescaled = scaler.inverse_transform(y_test)
y_pred_rescaled = scaler.inverse_transform(y_pred)
residuals_test = np.abs(y_test_rescaled - y_pred_rescaled)

# Train OneClassSVM on training data residuals with PCA
y_train_pred = lstm_model.predict(X_train)
y_train_rescaled = scaler.inverse_transform(y_train)
y_train_pred_rescaled = scaler.inverse_transform(y_train_pred)
residuals_train = np.abs(y_train_rescaled - y_train_pred_rescaled)

# Plot residuals before PCA for each feature
plt.figure(figsize=(12, 6))
for i, feature in enumerate(features):
    plt.plot(residuals_train[:, i], label=f'Residuals - {feature}')
plt.title('Residuals for Train Data (Before PCA)')
plt.xlabel('Sample Index')
plt.ylabel('Residual Value')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'residuals_train_before_pca.png'))
plt.close()

# Apply PCA
pca = PCA(n_components=0.95)  # Retain 95% variance
residuals_train_pca = pca.fit_transform(residuals_train)
residuals_test_pca = pca.transform(residuals_test)

# Plot explained variance ratio for each principal component
explained_variance = pca.explained_variance_ratio_
plt.figure(figsize=(10, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, color='blue')
plt.title('Explained Variance by Principal Components')
plt.xlabel('Principal Component')
plt.ylabel('Variance Ratio')
plt.grid(True)
plt.savefig(os.path.join(save_dir, 'explained_variance_pca.png'))
plt.close()

# Visualize data after PCA
plt.figure(figsize=(10, 6))
if residuals_train_pca.shape[1] >= 2:
    plt.scatter(residuals_train_pca[:, 0], residuals_train_pca[:, 1], alpha=0.5, label='Training Data (PCA)')
    plt.scatter(residuals_test_pca[:, 0], residuals_test_pca[:, 1], alpha=0.5, label='Test Data (PCA)', marker='x')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Data Visualization After PCA')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'data_after_pca.png'))
    plt.close()
else:
    print("PCA did not reduce dimensions to 2 or more components for scatter plot.")

# Fit OneClassSVM
ocsvm_multivariate = OneClassSVM(kernel='rbf', gamma='scale', nu=0.005)
ocsvm_multivariate.fit(residuals_train_pca)

# Generate grid for decision boundary visualization (training data)
xx, yy = np.meshgrid(
    np.linspace(residuals_train_pca[:, 0].min() - 1, residuals_train_pca[:, 0].max() + 1, 500),
    np.linspace(residuals_train_pca[:, 1].min() - 1, residuals_train_pca[:, 1].max() + 1, 500)
)

# Decision function values for the grid (training data)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = ocsvm_multivariate.decision_function(grid).reshape(xx.shape)

# Plot decision boundary and training data
plt.figure(figsize=(10, 6))
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')  # Boundary
plt.scatter(residuals_train_pca[:, 0], residuals_train_pca[:, 1], c='blue', s=20, edgecolor='k', label='Training Data')
plt.title('SVDD Decision Boundary (Training Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.grid(True)
plt.show()
train_boundary_plot_path = os.path.join(save_dir, 'svdd_training_boundary.png')
plt.savefig(train_boundary_plot_path)
plt.close()
print(f"Training boundary plot saved to {train_boundary_plot_path}")

# Predict anomalies on test data
svdd_predictions = ocsvm_multivariate.predict(residuals_test_pca)
y_pred_anomalies_svdd = np.where(svdd_predictions == -1, 1, 0)

# Generate grid for decision boundary visualization (test data)
xx, yy = np.meshgrid(
    np.linspace(residuals_test_pca[:, 0].min() - 1, residuals_test_pca[:, 0].max() + 1, 500),
    np.linspace(residuals_test_pca[:, 1].min() - 1, residuals_test_pca[:, 1].max() + 1, 500)
)

# Decision function values for the grid (test data)
grid = np.c_[xx.ravel(), yy.ravel()]
Z = ocsvm_multivariate.decision_function(grid).reshape(xx.shape)

# Plot decision boundary and test data
plt.figure(figsize=(10, 6))
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='red')  # Boundary
plt.scatter(
    residuals_test_pca[:, 0],
    residuals_test_pca[:, 1],
    c=y_pred_anomalies_svdd,
    cmap='coolwarm',
    s=20,
    edgecolor='k',
    label='Test Data (0=Normal, 1=Anomaly)'
)
plt.title('SVDD Decision Boundary (Test Data)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Anomaly Prediction (0=Normal, 1=Anomaly)')
plt.legend()
plt.grid(True)
plt.show()
test_boundary_plot_path = os.path.join(save_dir, 'svdd_test_boundary.png')
plt.savefig(test_boundary_plot_path)
plt.close()
print(f"Test boundary plot saved to {test_boundary_plot_path}")


# Define adaptive threshold using MSE
def adaptive_threshold_mse(error, window_size=20):
    threshold = []
    for i in range(len(error)):
        if i < window_size:
            # Compute MSE for initial subset
            mse = np.mean(np.square(error[:i+1]))
        else:
            # Compute MSE for the sliding window
            mse = np.mean(np.square(error[i-window_size:i]))
        
        # Define the threshold based on MSE
        threshold.append(mse + 0.25 * np.sqrt(mse))  # Adjust multiplier if needed
    
    return np.array(threshold)

# Calculate adaptive threshold based on residuals
threshold_values = adaptive_threshold_mse(residuals_test)

# Apply the adaptive threshold
y_pred_anomalies_threshold = np.where(np.max(residuals_test, axis=1) > threshold_values, 1, 0)

# Combine predictions
y_pred_anomalies_combined = (y_pred_anomalies_svdd & y_pred_anomalies_threshold).astype(int)

# Adjust y_true_anomalies to start from time_step index
y_true_anomalies = test_data[ground_truth_label].values[time_step:]
target_length = len(y_true_anomalies)

# Adjust predictions to match target length
y_pred_anomalies_svdd = y_pred_anomalies_svdd[:target_length]
y_pred_anomalies_threshold = y_pred_anomalies_threshold[:target_length]
y_pred_anomalies_combined = y_pred_anomalies_combined[:target_length]

# Flatten arrays
y_true_anomalies = y_true_anomalies.flatten()
y_pred_anomalies_svdd = y_pred_anomalies_svdd.flatten()
y_pred_anomalies_threshold = y_pred_anomalies_threshold.flatten()
y_pred_anomalies_combined = y_pred_anomalies_combined.flatten()

# Evaluation Metrics for One-Class SVM
test_accuracy_svdd = accuracy_score(y_true_anomalies, y_pred_anomalies_svdd)
test_precision_svdd = precision_score(y_true_anomalies, y_pred_anomalies_svdd)
test_recall_svdd = recall_score(y_true_anomalies, y_pred_anomalies_svdd)
test_f1_svdd = f1_score(y_true_anomalies, y_pred_anomalies_svdd)

# Evaluation Metrics for Adaptive Threshold
test_accuracy_threshold = accuracy_score(y_true_anomalies, y_pred_anomalies_threshold)
test_precision_threshold = precision_score(y_true_anomalies, y_pred_anomalies_threshold)
test_recall_threshold = recall_score(y_true_anomalies, y_pred_anomalies_threshold)
test_f1_threshold = f1_score(y_true_anomalies, y_pred_anomalies_threshold)

# Evaluation Metrics for Combined (AND operation)
test_accuracy_combined = accuracy_score(y_true_anomalies, y_pred_anomalies_combined)
test_precision_combined = precision_score(y_true_anomalies, y_pred_anomalies_combined)
test_recall_combined = recall_score(y_true_anomalies, y_pred_anomalies_combined)
test_f1_combined = f1_score(y_true_anomalies, y_pred_anomalies_combined)

# Confusion Matrices
conf_matrix_svdd = confusion_matrix(y_true_anomalies, y_pred_anomalies_svdd)
conf_matrix_threshold = confusion_matrix(y_true_anomalies, y_pred_anomalies_threshold)
conf_matrix_combined = confusion_matrix(y_true_anomalies, y_pred_anomalies_combined)

# Identifikasi baris-baris dengan anomali pada data uji berdasarkan hasil 'combined'
anomaly_indices = np.where(y_pred_anomalies_combined == 1)[0]

# List untuk menyimpan hasil deteksi sensor yang kemungkinan rusak
anomaly_detections = []

# Loop melalui setiap baris anomali dan temukan residual terbesar
for idx in anomaly_indices:
    # Ambil residual untuk baris yang terdeteksi anomali
    residuals_row = residuals_test[idx]
    
    # Temukan indeks fitur dengan residual terbesar
    max_residual_index = np.argmax(residuals_row)
    
    # Fitur (sensor) dengan residual terbesar
    sensor_name = features[max_residual_index]
    max_residual_value = residuals_row[max_residual_index]
    
    # Simpan hasil dalam bentuk dictionary
    anomaly_detections.append({
        'Index': idx,
        'Sensor': sensor_name,
        'Max Residual': max_residual_value
    })

# Konversi hasil deteksi ke DataFrame untuk kemudahan analisis
anomaly_detections_df = pd.DataFrame(anomaly_detections)

# Simpan hasil deteksi anomali ke dalam file Excel
anomaly_detection_file_path = os.path.join(save_dir, 'anomaly_detection_summary.xlsx')
anomaly_detections_df.to_excel(anomaly_detection_file_path, index=False)

print(f"Anomaly detection summary saved successfully to {anomaly_detection_file_path}")

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svdd, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={"size": 22})  # Increase the font size of the annotations
plt.title('Confusion Matrix (SVDD)', fontsize=16)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(save_dir, 'confusion_matrix_svdd.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_threshold, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={"size": 22})  # Increase the font size of the annotations
plt.title('Confusion Matrix (Adaptive Threshold)', fontsize=22)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(save_dir, 'confusion_matrix_threshold.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_combined, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'],
            annot_kws={"size": 22})  # Increase the font size of the annotations
plt.title('Confusion Matrix (Combined)', fontsize=22)
plt.xlabel('Predicted', fontsize=14)
plt.ylabel('Actual', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig(os.path.join(save_dir, 'confusion_matrix_combined.png'))
plt.close()

# Save Evaluation Results to Excel including precision and recall
evaluation_results = {
    "Method": ["SVDD", "Adaptive Threshold", "Combined (AND)"],
    "Accuracy": [test_accuracy_svdd, test_accuracy_threshold, test_accuracy_combined],
    "Precision": [test_precision_svdd, test_precision_threshold, test_precision_combined],
    "Recall": [test_recall_svdd, test_recall_threshold, test_recall_combined],
    "F1-Score": [test_f1_svdd, test_f1_threshold, test_f1_combined]
}
evaluation_df = pd.DataFrame(evaluation_results)
evaluation_file_path = 'D:/Tugas Akhir/Hasil14/evaluation_results.xlsx'
evaluation_df.to_excel(evaluation_file_path, index=False)

# Save Results to DataFrame
results_dict = {f'Actual_{feature}': y_test_rescaled[:, i].flatten() for i, feature in enumerate(features)}
results_dict.update({f'Predicted_{feature}': y_pred_rescaled[:, i].flatten() for i, feature in enumerate(features)})
results_dict.update({f'Residuals_{feature}': residuals_test[:, i].flatten() for i, feature in enumerate(features)})
results_dict['Anomaly_Prediction_SVDD'] = y_pred_anomalies_svdd
results_dict['Anomaly_Prediction_Threshold'] = y_pred_anomalies_threshold
results_dict['Anomaly_Prediction_Combined'] = y_pred_anomalies_combined
results_dict['True_Anomalies'] = y_true_anomalies

# Save as Excel
results_df = pd.DataFrame(results_dict)
results_file_path = 'D:/Tugas Akhir/Hasil14/anomaly_detection_results_combined.xlsx'
results_df.to_excel(results_file_path, index=False)

print("Evaluation metrics including precision and recall have been saved successfully.")
print("Confusion matrices and anomaly detection results have been saved successfully.")

# Define adaptive threshold function
def adaptive_threshold(error, window_size=20):
    threshold = []
    for i in range(len(error)):
        if i < window_size:
            threshold.append(np.mean(error[:i+1]) + 2 * np.std(error[:i+1]))
        else:
            threshold.append(np.mean(error[i-window_size:i]) + 2 * np.std(error[i-window_size:i]))
    return np.array(threshold)

# Prepare ground truth for training data
y_true_anomalies_train = train_data_normal[ground_truth_label].values[time_step:]
target_length_train = len(y_true_anomalies_train)

# Predict anomalies on training data using One-Class SVM
train_predictions = ocsvm_multivariate.predict(residuals_train_pca)
y_pred_anomalies_svdd_train = np.where(train_predictions == -1, 1, 0)[:target_length_train]

# Apply adaptive threshold on training data residuals
threshold_values_train = adaptive_threshold(residuals_train)
y_pred_anomalies_threshold_train = np.where(np.max(residuals_train, axis=1) > threshold_values_train, 1, 0)[:target_length_train]

# Combine predictions (AND operation) for training data
y_pred_anomalies_combined_train = (y_pred_anomalies_svdd_train & y_pred_anomalies_threshold_train).astype(int)

# Evaluation Metrics for One-Class SVM on Training Data
train_accuracy_svdd = accuracy_score(y_true_anomalies_train, y_pred_anomalies_svdd_train)
train_precision_svdd = precision_score(y_true_anomalies_train, y_pred_anomalies_svdd_train)
train_recall_svdd = recall_score(y_true_anomalies_train, y_pred_anomalies_svdd_train)
train_f1_svdd = f1_score(y_true_anomalies_train, y_pred_anomalies_svdd_train)
conf_matrix_svdd_train = confusion_matrix(y_true_anomalies_train, y_pred_anomalies_svdd_train)

# Evaluation Metrics for Adaptive Threshold on Training Data
train_accuracy_threshold = accuracy_score(y_true_anomalies_train, y_pred_anomalies_threshold_train)
train_precision_threshold = precision_score(y_true_anomalies_train, y_pred_anomalies_threshold_train)
train_recall_threshold = recall_score(y_true_anomalies_train, y_pred_anomalies_threshold_train)
train_f1_threshold = f1_score(y_true_anomalies_train, y_pred_anomalies_threshold_train)
conf_matrix_threshold_train = confusion_matrix(y_true_anomalies_train, y_pred_anomalies_threshold_train)

# Evaluation Metrics for Combined (AND operation) on Training Data
train_accuracy_combined = accuracy_score(y_true_anomalies_train, y_pred_anomalies_combined_train)
train_precision_combined = precision_score(y_true_anomalies_train, y_pred_anomalies_combined_train)
train_recall_combined = recall_score(y_true_anomalies_train, y_pred_anomalies_combined_train)
train_f1_combined = f1_score(y_true_anomalies_train, y_pred_anomalies_combined_train)
conf_matrix_combined_train = confusion_matrix(y_true_anomalies_train, y_pred_anomalies_combined_train)

# Save Confusion Matrices for Training Data
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_svdd_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix (SVDD - Training Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(save_dir, 'confusion_matrix_svdd_train.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_threshold_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix (Adaptive Threshold - Training Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(save_dir, 'confusion_matrix_threshold_train.png'))
plt.close()

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_combined_train, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
plt.title('Confusion Matrix (Combined - Training Data)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig(os.path.join(save_dir, 'confusion_matrix_combined_train.png'))
plt.close()

# Save Evaluation Results for Training Data to Excel
evaluation_results_train = {
    "Method": ["SVDD (Training)", "Adaptive Threshold (Training)", "Combined (Training)"],
    "Accuracy": [train_accuracy_svdd, train_accuracy_threshold, train_accuracy_combined],
    "Precision": [train_precision_svdd, train_precision_threshold, train_precision_combined],
    "Recall": [train_recall_svdd, train_recall_threshold, train_recall_combined],
    "F1-Score": [train_f1_svdd, train_f1_threshold, train_f1_combined]
}
evaluation_df_train = pd.DataFrame(evaluation_results_train)
evaluation_file_path_train = os.path.join(save_dir, 'evaluation_results_train.xlsx')
evaluation_df_train.to_excel(evaluation_file_path_train, index=False)

print("Training evaluation metrics including precision, recall, and confusion matrices have been saved successfully.")


# Visualisasi Hasil Prediksi One-Class SVM pada Data Latih
plt.figure(figsize=(10, 6))

# Data normal dan anomali pada data latih berdasarkan hasil prediksi One-Class SVM
train_predictions = ocsvm_multivariate.predict(residuals_train_pca)
normal_data_train = residuals_train_pca[train_predictions == 1]
anomalous_data_train = residuals_train_pca[train_predictions == -1]

# Scatter plot dari data normal (label 1) dan anomali (label -1) pada data latih
plt.scatter(normal_data_train[:, 0], normal_data_train[:, 1], c='blue', alpha=0.5, label='Normal Data (Inliers)')
plt.scatter(anomalous_data_train[:, 0], anomalous_data_train[:, 1], c='red', alpha=0.5, label='Anomalous Data (Outliers)')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVDD Results on Training Data (PCA-Transformed)')
plt.legend()
plt.grid(True)

# Simpan hasil plot ke direktori
ocsvm_train_result_plot_filename = os.path.join(save_dir, 'ocsvm_results_train_data.png')
plt.savefig(ocsvm_train_result_plot_filename)
plt.close()

print(f"SVDD training result plot saved successfully to {ocsvm_train_result_plot_filename}")


# Visualisasi Hasil Prediksi One-Class SVM pada Data Uji
plt.figure(figsize=(10, 6))

# Data normal dan anomali pada data uji berdasarkan hasil prediksi One-Class SVM
normal_data_test = residuals_test_pca[svdd_predictions == 1]
anomalous_data_test = residuals_test_pca[svdd_predictions == -1]

# Scatter plot dari data normal (label 1) dan anomali (label -1) pada data uji
plt.scatter(normal_data_test[:, 0], normal_data_test[:, 1], c='blue', alpha=0.5, label='Normal Data (Inliers)')
plt.scatter(anomalous_data_test[:, 0], anomalous_data_test[:, 1], c='red', alpha=0.5, label='Anomalous Data (Outliers)')

plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('SVDD Results on Test Data (PCA-Transformed)')
plt.legend()
plt.grid(True)

# Simpan hasil plot ke direktori
ocsvm_result_plot_filename = os.path.join(save_dir, 'ocsvm_results_test_data.png')
plt.savefig(ocsvm_result_plot_filename)
plt.close()

print(f"SVDD result plot saved successfully to {ocsvm_result_plot_filename}")

# Plot original and standardized training data
plt.figure(figsize=(14, 8))
for i, feature in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.plot(train_data_normal[feature], label=f'Original {feature}', color='blue')
    plt.plot(scaled_train_data[:, i], label=f'Normalization {feature}', color='orange')
    plt.title(f'{feature} (Original vs Normalization) - Training Data')
    plt.legend()
    plt.grid(True)

# Save the figure for original vs standardized training data
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'original_vs_Normalization_training.png'))
plt.close()

print("Original vs Normalization Training Data plot saved successfully.")

# Plot original and standardized test data
scaled_test_data = scaler.transform(test_data[features])  # Apply StandardScaler to test data

plt.figure(figsize=(14, 8))
for i, feature in enumerate(features):
    plt.subplot(4, 2, i+1)
    plt.plot(test_data[feature], label=f'Original {feature}', color='blue')
    plt.plot(scaled_test_data[:, i], label=f'Normalization {feature}', color='orange')
    plt.title(f'{feature} (Original vs Normalization) - Test Data')
    plt.legend()
    plt.grid(True)

# Save the figure for original vs standardized test data
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'original_vs_Normalization_test.png'))
plt.close()

print("Original vs Normalization Test Data plot saved successfully.")

# Plot Residuals and Adaptive Threshold for Test Data
plt.figure(figsize=(12, 6))
plt.plot(np.max(residuals_test, axis=1), label='Max Residuals (Test Data)', color='blue')
plt.plot(threshold_values, label='Adaptive Threshold', color='red', linestyle='dashed')
plt.title('Residuals vs Adaptive Threshold (Test Data)', fontsize=20)
plt.xlabel('Sample Index', fontsize=20)
plt.ylabel('Residuals / Threshold Value', fontsize=18)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)
plt.savefig(os.path.join(save_dir, 'threshold_vs_residuals_test.png'))
plt.close()

# Calculate adaptive threshold based on residuals for training data
threshold_values_train = adaptive_threshold(residuals_train)

# Plot Residuals and Adaptive Threshold for Training Data
plt.figure(figsize=(12, 6))
plt.plot(np.max(residuals_train, axis=1), label='Max Residuals (Training Data)', color='blue')
plt.plot(threshold_values_train, label='Adaptive Threshold (Training)', color='red', linestyle='dashed')
plt.title('Residuals vs Adaptive Threshold (Training Data)', fontsize=20)
plt.xlabel('Sample Index', fontsize=16)
plt.ylabel('Residuals / Threshold Value', fontsize=16)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.legend(fontsize=20)
plt.grid(True, linestyle="--", alpha=0.7)

# Save the plot
training_residual_threshold_plot_filename = os.path.join(save_dir, 'threshold_vs_residuals_train.png')
plt.savefig(training_residual_threshold_plot_filename)
plt.close()

print(f"Residuals vs Adaptive Threshold plot for training data saved successfully to {training_residual_threshold_plot_filename}")

# Save Actual vs Predicted for Test and Train Data with controlled Y-axis scale
for i in range(len(features)):
    # Set Y-axis limits with margin
    y_min_test = y_test_rescaled[:, i].min() - 0.5
    y_max_test = y_test_rescaled[:, i].max() + 0.5

    # Test Data with Y-axis adjusted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled[:, i], label=f'Actual Data - {features[i]}', color='blue')
    plt.plot(y_pred_rescaled[:, i], label=f'Predicted Data - {features[i]}', color='orange', linestyle='dashed')

    # Highlight only the largest residual in each row
    anomalies_indices = np.where(y_pred_anomalies_combined == 1)[0]
    largest_residual_indices = np.argmax(residuals_test[anomalies_indices, :], axis=1)  # Index of largest residual per anomaly
    relevant_anomalies = anomalies_indices[largest_residual_indices == i]  # Select anomalies corresponding to this feature
    plt.scatter(relevant_anomalies, y_test_rescaled[relevant_anomalies, i], color='red', marker='x', s=80, label='Max Residual Anomalies')

    plt.ylim(y_min_test, y_max_test)
    plt.title(f'Actual vs Predicted Test Data with Anomalies  - {features[i]}', fontsize=28)
    plt.xlabel('Sample Index', fontsize=23)
    plt.ylabel(f'{features[i]} Value', fontsize=23)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, f'actual_vs_predicted_with_max_anomalies_test_{features[i]}.png'))
    plt.close()

    # Train Data
    plt.figure(figsize=(12, 6))
    plt.plot(y_train_rescaled[:, i], label=f'Actual Train Data - {features[i]}', color='blue')
    plt.plot(y_train_pred_rescaled[:, i], label=f'Predicted Train Data - {features[i]}', color='orange', linestyle='dashed')

    plt.ylim(y_min_test, y_max_test)
    plt.title(f'Actual vs Predicted Training Data with  Anomalies - {features[i]}', fontsize=28)
    plt.xlabel('Sample Index', fontsize=23)
    plt.ylabel(f'{features[i]} Value', fontsize=23)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, f'actual_vs_predicted_with_max_anomalies_train_{features[i]}.png'))
    plt.close()


# Save Residuals for Test Data per sensor
for i, feature in enumerate(features):
    plt.figure(figsize=(12, 6))
    plt.plot(residuals_test[:, i], label=f'Residuals - {feature}', color='blue')
    plt.title(f'Residuals for Test Data - {feature}', fontsize=26)
    plt.xlabel('Sample Index', fontsize=23)
    plt.ylabel('Residual Value', fontsize=23)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, f'residuals_test_{feature}.png'))
    plt.close()

# Save Residuals for Train Data per sensor
for i, feature in enumerate(features):
    plt.figure(figsize=(12, 6))
    plt.plot(residuals_train[:, i], label=f'Residuals - {feature}', color='blue')
    plt.title(f'Residuals for Train Data - {feature}', fontsize=26)
    plt.xlabel('Sample Index', fontsize=23)
    plt.ylabel('Residual Value', fontsize=23)
    plt.xticks(fontsize=24)
    plt.yticks(fontsize=24)
    plt.legend(fontsize=24)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(save_dir, f'residuals_train_{feature}.png'))
    plt.close()

# Calculate correlation matrix for training data features
correlation_matrix = train_data_normal[features].corr()

# Plot the correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix of Training Data Features')
plt.tight_layout()

# Save the correlation matrix plot
correlation_matrix_plot_filename = os.path.join(save_dir, 'correlation_matrix_training_data.png')
plt.savefig(correlation_matrix_plot_filename)
plt.close()

print(f"Correlation matrix plot saved successfully to {correlation_matrix_plot_filename}")


print("All plots and results have been saved successfully.")

# Define file paths for saving each component
lstm_model_path = os.path.join(save_dir, 'lstm_model.h5')
scaler_path = os.path.join(save_dir, 'scaler.pkl')
svdd_model_path = os.path.join(save_dir, 'svdd_model.pkl')
pca_model_path = os.path.join(save_dir, 'pca_model.pkl')


# Save LSTM model
lstm_model.save(lstm_model_path)
print(f"LSTM model saved successfully to {lstm_model_path}")

# Save scaler, SVDD model, and PCA
joblib.dump(scaler, scaler_path)
print(f"Scaler saved successfully to {scaler_path}")

joblib.dump(ocsvm_multivariate, svdd_model_path)
print(f"SVDD model saved successfully to {svdd_model_path}")

joblib.dump(pca, pca_model_path)
print(f"PCA model saved successfully to {pca_model_path}")


# Plot Residuals and Adaptive Threshold for Test Data
plt.figure(figsize=(12, 6))
plt.plot(np.max(residuals_test, axis=1), label='Max Residuals (Test Data)', color='blue')
plt.plot(threshold_values, label='Adaptive Threshold', color='red', linestyle='dashed')
plt.title('Residuals vs Adaptive Threshold (Test Data)', fontsize=20)
plt.xlabel('Sample Index', fontsize=20)
plt.ylabel('Residuals / Threshold Value', fontsize=18)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()  # Tampilkan plot

# Calculate adaptive threshold based on residuals for training data
threshold_values_train = adaptive_threshold(residuals_train)

# Plot Residuals and Adaptive Threshold for Training Data
plt.figure(figsize=(12, 6))
plt.plot(np.max(residuals_train, axis=1), label='Max Residuals (Training Data)', color='blue')
plt.plot(threshold_values_train, label='Adaptive Threshold (Training)', color='red', linestyle='dashed')
plt.title('Residuals vs Adaptive Threshold (Training Data)', fontsize=16)
plt.xlabel('Sample Index', fontsize=14)
plt.ylabel('Residuals / Threshold Value', fontsize=14)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()  # Tampilkan plot

# Plot residuals for Test Data
plt.figure(figsize=(12, 6))
for i in range(len(features)):
    plt.plot(residuals_test[:, i], label=f'Residuals - {features[i]}')
plt.title('Residuals for Test Data', fontsize=23)
plt.xlabel('Sample Index', fontsize=22)
plt.ylabel('Residual Value', fontsize=22)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()  # Tampilkan plot

# Plot residuals for Train Data
plt.figure(figsize=(12, 6))
for i in range(len(features)):
    plt.plot(residuals_train[:, i], label=f'Residuals - {features[i]}')
plt.title('Residuals for Train Data', fontsize=23)
plt.xlabel('Sample Index', fontsize=22)
plt.ylabel('Residual Value', fontsize=22)
plt.xticks(fontsize=26)
plt.yticks(fontsize=26)
plt.grid(True)
plt.show()  # Tampilkan plot

# Display Actual vs Predicted for Test and Train Data with controlled Y-axis scale
for i in range(len(features)):
    # Set Y-axis limits with margin
    y_min_test = y_test_rescaled[:, i].min() - 0.5
    y_max_test = y_test_rescaled[:, i].max() + 0.5

    # Test Data with Y-axis adjusted
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_rescaled[:, i], label=f'Actual Test Data ', color='blue')
    plt.plot(y_pred_rescaled[:, i], label=f'Predicted Test Data ', color='orange', linestyle='dashed')

    # Highlight only the largest residual in each row
    anomalies_indices = np.where(y_pred_anomalies_combined == 1)[0]
    largest_residual_indices = np.argmax(residuals_test[anomalies_indices, :], axis=1)  # Index of largest residual per anomaly
    relevant_anomalies = anomalies_indices[largest_residual_indices == i]  # Select anomalies corresponding to this feature
    plt.scatter(relevant_anomalies, y_test_rescaled[relevant_anomalies, i], color='red', marker='x', s=80, label='Max Residual Anomalies')

    plt.ylim(y_min_test, y_max_test)
    plt.title(f'Actual vs Predicted Test Data - {features[i]}', fontsize=24)
    plt.xlabel('Sample Index', fontsize=20)
    plt.ylabel(f'{features[i]} Value', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()  # Display the plot instead of saving

    # Train Data
    plt.figure(figsize=(12, 6))
    plt.plot(y_train_rescaled[:, i], label=f'Actual Train Data - {features[i]}', color='blue')
    plt.plot(y_train_pred_rescaled[:, i], label=f'Predicted Train Data - {features[i]}', color='orange', linestyle='dashed')

    # Highlight only the largest residual in each row for training data
    anomalies_indices_train = np.where(y_pred_anomalies_combined_train == 1)[0]
    largest_residual_indices_train = np.argmax(residuals_train[anomalies_indices_train, :], axis=1)  # Index of largest residual per anomaly
    relevant_anomalies_train = anomalies_indices_train[largest_residual_indices_train == i]  # Select anomalies corresponding to this feature
    plt.scatter(relevant_anomalies_train, y_train_rescaled[relevant_anomalies_train, i], color='red', marker='x', s=80, label='Max Residual Anomalies')

    plt.ylim(y_min_test, y_max_test)
    plt.title(f'Actual vs Predicted Training Data  - {features[i]}', fontsize=24)
    plt.xlabel('Sample Index', fontsize=20)
    plt.ylabel(f'{features[i]} Value', fontsize=20)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()  # Display the plot instead of saving


# Hitung MSE, RMSE, dan MAE
mse_test = mean_squared_error(y_test_rescaled, y_pred_rescaled)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test_rescaled, y_pred_rescaled)

# Tampilkan hasil di konsol
print(f"Test MSE: {mse_test}, RMSE: {rmse_test}, MAE: {mae_test}")

# Simpan hasil ke file Excel
results_dict = {
    "Metric": ["MSE", "RMSE", "MAE"],
    "Value": [mse_test, rmse_test, mae_test]
}

results_df = pd.DataFrame(results_dict)

# Pastikan direktori hasil ada
os.makedirs(save_dir, exist_ok=True)

# Simpan ke file Excel
results_file_path = os.path.join(save_dir, 'test_metrics_results.xlsx')
results_df.to_excel(results_file_path, index=False)

print(f"Test metrics saved successfully to {results_file_path}")
