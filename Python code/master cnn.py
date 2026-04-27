import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. DATA LOADING - THE "LONG WINDOW" FIX
# ==========================================
data_path = r'C:\Des\Academic\5th Year\2nd\ML\Project'
file_map = {'struct_rs_R1.mat': 0, 'struct_r1b_R1.mat': 1, 
            'struct_r2b_R1.mat': 2, 'struct_r3b_R1.mat': 3, 
            'struct_r4b_R1.mat': 4}

all_signals, all_labels = [], []
# INCREASED: 10,000 samples = 0.2 seconds (Crucial for fault detection)
time_steps = 10000 

print("--- Loading Long-Window Data for High Accuracy CNN ---")

for filename, label in file_map.items():
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path): continue
    
    with h5py.File(full_path, 'r') as f:
        main_key = [k for k in f.keys() if k != '#refs#'][0]
        for t_key in f[main_key].keys():
            t_group = f[main_key][t_key]
            try:
                curr_raw = np.array(f[t_group['Ia'][0][0]]).flatten()
                vibe_raw = np.array(f[t_group['Vib_axial'][0][0]]).flatten()
                
                # Simple Normalization
                curr_raw = (curr_raw - np.mean(curr_raw)) / np.std(curr_raw)
                vibe_raw = (vibe_raw - np.mean(vibe_raw)) / np.std(vibe_raw)
                
                # Take 40 long samples per torque level
                for i in range(40):
                    start = i * time_steps
                    if (start + time_steps) > len(curr_raw): break
                    
                    segment = np.stack([curr_raw[start:start+time_steps], 
                                      vibe_raw[start:start+time_steps]], axis=1)
                    all_signals.append(segment)
                    all_labels.append(label)
            except: continue

X = np.array(all_signals)
y = np.array(all_labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# 2. THE "DEEP-WAVE" CNN ARCHITECTURE
# ==========================================
def build_final_cnn():
    model = models.Sequential([
        # Large initial kernel (50) to filter out 50Hz electricity
        layers.Conv1D(32, kernel_size=50, strides=2, activation='relu', input_shape=(time_steps, 2)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        
        layers.Conv1D(64, kernel_size=25, strides=2, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(4),
        
        layers.Conv1D(128, kernel_size=10, activation='relu'),
        layers.GlobalAveragePooling1D(), # Summarizes the 0.2s window
        
        layers.Dense(100, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(5, activation='softmax')
    ])
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

# ==========================================
# 3. TRAINING
# ==========================================
model = build_final_cnn()
print("\n--- Training Final High-Accuracy CNN ---")
# Lower batch size (16) works better for long windows
model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))

# Final Evaluation
y_pred = np.argmax(model.predict(X_test), axis=1)
acc = accuracy_score(y_test, y_pred)
print(f"\nFinal CNN Accuracy: {acc*100:.2f}%")

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['H','1b','2b','3b','4b'], yticklabels=['H','1b','2b','3b','4b'])
plt.title(f'Final CNN Result (Window: 0.2s): {acc*100:.1f}%')
plt.show()