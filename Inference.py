from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from tensorflow.keras.models import load_model
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Path ke model dan komponen yang telah disimpan
save_dir = 'D:/Tugas Akhir/Hasil13'
lstm_model_path = 'D:/Tugas Akhir/Hasil12/lstm_model.h5'
scaler_path = 'D:/Tugas Akhir/Hasil12/scaler.pkl'
svdd_model_path = 'D:/Tugas Akhir/Hasil12/svdd_model.pkl'
pca_model_path = 'D:/Tugas Akhir/Hasil12/pca_model.pkl'


# Muat model dan komponen yang disimpan
lstm_model = load_model(lstm_model_path)
scaler = joblib.load(scaler_path)
svdd_model = joblib.load(svdd_model_path)
pca_model = joblib.load(pca_model_path)

# Fungsi untuk membuat sequence
def create_sequences(data, time_step=25):
    X = []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step)])
    return np.array(X)

# Fungsi untuk menghitung threshold adaptif
def calculate_adaptive_threshold(residuals, window_size=20, multiplier=1.5):
    threshold = []
    for i in range(len(residuals)):
        if i < window_size:
            threshold.append(np.mean(residuals[:i+1]) + multiplier * np.std(residuals[:i+1]))
        else:
            threshold.append(np.mean(residuals[i-window_size:i]) + multiplier * np.std(residuals[i-window_size:i]))
    return np.array(threshold)

# Fungsi untuk membuat sequence
def inference(data, features, save_dir, time_step=25, window_size=20, multiplier=1.5):
    os.makedirs(save_dir, exist_ok=True)  # Pastikan direktori hasil ada
    
    # Normalisasi data
    scaled_data = scaler.transform(data[features])
    
    # Buat sequence
    X = create_sequences(scaled_data, time_step)
    
    # Prediksi menggunakan LSTM
    y_pred = lstm_model.predict(X)
    
    # Transformasi balik data
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_actual_rescaled = scaler.inverse_transform(scaled_data[time_step:])
    residuals = np.abs(scaled_data[time_step:] - y_pred)
    residuals_pca = pca_model.transform(residuals)
    
    # Deteksi menggunakan SVDD
    svdd_predictions = svdd_model.predict(residuals_pca)
    anomalies_svdd = np.where(svdd_predictions == -1, 1, 0)
    
    # Hitung threshold adaptif
    max_residuals = np.max(residuals, axis=1)
    adaptive_threshold = calculate_adaptive_threshold(max_residuals, window_size, multiplier)
    anomalies_threshold = np.where(max_residuals > adaptive_threshold, 1, 0)
    
    # Kombinasi prediksi (AND operation)
    combined_anomalies = (anomalies_svdd & anomalies_threshold).astype(int)
    
    # Ground Truth
    y_true = data[ground_truth_label].values[time_step:]  # Sesuaikan panjang ground truth dengan data
    
    # Evaluasi
    evaluation_results = {
        "Method": ["SVDD", "Adaptive Threshold", "Combined (AND)"],
        "Accuracy": [
            accuracy_score(y_true, anomalies_svdd),
            accuracy_score(y_true, anomalies_threshold),
            accuracy_score(y_true, combined_anomalies)
        ],
        "Precision": [
            precision_score(y_true, anomalies_svdd),
            precision_score(y_true, anomalies_threshold),
            precision_score(y_true, combined_anomalies)
        ],
        "Recall": [
            recall_score(y_true, anomalies_svdd),
            recall_score(y_true, anomalies_threshold),
            recall_score(y_true, combined_anomalies)
        ],
        "F1-Score": [
            f1_score(y_true, anomalies_svdd),
            f1_score(y_true, anomalies_threshold),
            f1_score(y_true, combined_anomalies)
        ]
    }
    evaluation_df = pd.DataFrame(evaluation_results)
    
    # Simpan hasil evaluasi ke Excel
    evaluation_file_path = os.path.join(save_dir, 'evaluation_results.xlsx')
    evaluation_df.to_excel(evaluation_file_path, index=False)
    print(f"Evaluation results saved to {evaluation_file_path}")
    
    # Confusion Matrices
    confusion_matrices = {
        "SVDD": confusion_matrix(y_true, anomalies_svdd),
        "Threshold": confusion_matrix(y_true, anomalies_threshold),
        "Combined": confusion_matrix(y_true, combined_anomalies)
    }
    
    # Plot Confusion Matrices
    for method, cm in confusion_matrices.items():
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Anomaly'], yticklabels=['Normal', 'Anomaly'])
        plt.title(f'Confusion Matrix ({method})', fontsize=16)
        plt.xlabel('Predicted', fontsize=14)
        plt.ylabel('Actual', fontsize=14)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        confusion_plot_path = os.path.join(save_dir, f'confusion_matrix_{method.lower()}.png')
        plt.savefig(confusion_plot_path)
        plt.close()
        print(f"Confusion Matrix for {method} saved to {confusion_plot_path}")
    
    # Simpan hasil prediksi ke Excel
    results_file_path = os.path.join(save_dir, 'inference_results.xlsx')
    results_df = pd.DataFrame({
        "Max Residuals": max_residuals,
        "SVDD_Anomalies": anomalies_svdd,
        "Threshold_Anomalies": anomalies_threshold,
        "Combined_Anomalies": combined_anomalies,
        "True_Labels": y_true
    })
    results_df.to_excel(results_file_path, index=False)
    print(f"Results saved to {results_file_path}")
    
    # Visualisasi Actual vs Predicted untuk setiap fitur
    for i, feature in enumerate(features):
        y_min = min(y_actual_rescaled[:, i].min(), y_pred_rescaled[:, i].min()) - 0.5
        y_max = max(y_actual_rescaled[:, i].max(), y_pred_rescaled[:, i].max()) + 0.5

        # Plot Actual vs Predicted
        plt.figure(figsize=(12, 6))
        plt.plot(y_actual_rescaled[:, i], label=f'Actual Data - {feature}', color='blue')
        plt.plot(y_pred_rescaled[:, i], label=f'Predicted Data - {feature}', color='orange', linestyle='dashed')
        
        # Highlight Anomalies
        anomalies_indices = np.where(combined_anomalies == 1)[0]
        relevant_anomalies = anomalies_indices[np.argmax(residuals[anomalies_indices, :], axis=1) == i]
        plt.scatter(relevant_anomalies, y_actual_rescaled[relevant_anomalies, i], color='red', marker='x', s=80, label='Anomaly')

        plt.ylim(y_min, y_max)
        plt.title(f'Actual vs Predicted Data with Anomalies - {feature}', fontsize=16)
        plt.xlabel('Sample Index', fontsize=14)
        plt.ylabel(f'{feature} Value', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle="--", alpha=0.7)
        plot_path = os.path.join(save_dir, f'actual_vs_predicted_{feature}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Plot saved to {plot_path}")
    
    # Kembalikan hasil prediksi dan evaluasi
    return results_df, evaluation_df

# Contoh penggunaan inference
new_data_path = 'D:/Tugas Akhir/Dataset/February/Anomaly_Applied_Data.csv'
new_data = pd.read_csv(new_data_path)

# Fitur yang digunakan untuk prediksi
features = ['CHL_RW_TEMP_1', 'CHL_SW_TEMP_1', 'CHL_SWCD_TEMP_1', 
            'CHL_RWCD_TEMP_1', 'CT_RW_TEMP_1', 'CDWL_RW_TEMP', 'CDWL_SW_TEMP', 
            'CT_SW_TEMP_1']
ground_truth_label = 'Anomali'

# Jalankan inference
results, evaluation = inference(new_data, features, save_dir)