import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
data_path = r'C:\Des\Academic\5th Year\2nd\ML\Project'
file_map = {'struct_rs_R1.mat': 0, 'struct_r1b_R1.mat': 1, 'struct_r2b_R1.mat': 2, 
            'struct_r3b_R1.mat': 3, 'struct_r4b_R1.mat': 4}

all_signals = []
all_labels = []

# For CNN, we use a fixed segment size (e.g., 1000 time steps)
time_steps = 1000 

print("--- Loading Raw Data for CNN Training ---")

for filename, label in file_map.items():
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path): continue
    
    with h5py.File(full_path, 'r') as f:
        main_key = [k for k in f.keys() if k != '#refs#'][0]
        main_struct = f[main_key]
        
        for t_key in main_struct.keys():
            t_group = main_struct[t_key]
            try:
                # Follow pointers to raw signals
                curr_raw = np.array(f[t_group['Ia'][0][0]]).flatten()
                vibe_raw = np.array(f[t_group['Vib_axial'][0][0]]).flatten()
                
                # Normalize raw signals (Crucial for Neural Networks)
                curr_raw = (curr_raw - np.mean(curr_raw)) / np.std(curr_raw)
                vibe_raw = (vibe_raw - np.mean(vibe_raw)) / np.std(vibe_raw)
                
                # Slicing raw waves into segments
                num_samples = 100
                for i in range(num_samples):
                    start = i * time_steps
                    if (start + time_steps) > len(curr_raw) or (start + time_steps) > len(vibe_raw):
                        break
                    
                    # Combine Current and Vibration as 2 "Channels" (like an image)
                    segment = np.stack([curr_raw[start:start+time_steps], 
                                      vibe_raw[start:start+time_steps]], axis=1)
                    all_signals.append(segment)
                    all_labels.append(label)
            except: continue

X = np.array(all_signals) # Shape: (Samples, 1000, 2)
y = np.array(all_labels)

# ==========================================
# 2. BUILDING THE 1D-CNN ARCHITECTURE
# ==========================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models.Sequential([
    # First Convolution Layer: Scans the wave for patterns
    layers.Conv1D(filters=32, kernel_size=10, activation='relu', input_shape=(time_steps, 2)),
    layers.MaxPooling1D(pool_size=2),
    
    # Second Convolution Layer: Scans for complex combinations
    layers.Conv1D(filters=64, kernel_size=5, activation='relu'),
    layers.MaxPooling1D(pool_size=2),
    layers.Dropout(0.2), # Prevents memorization
    
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(5, activation='softmax') # 5 Classes (0,1,2,3,4 bars)
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# ==========================================
# 3. TRAINING
# ==========================================
print("\nTraining CNN...")
history = model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# ==========================================
# 4. RESULTS
# ==========================================
y_pred = np.argmax(model.predict(X_test), axis=1)
cnn_acc = accuracy_score(y_test, y_pred)
print(f"\nCNN Final Accuracy: {cnn_acc*100:.2f}%")

# Plot Confusion Matrix
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Oranges',
            xticklabels=['H','1b','2b','3b','4b'], yticklabels=['H','1b','2b','3b','4b'])
plt.title(f'CNN Fault Detection Results (Acc: {cnn_acc*100:.1f}%)')
plt.show()