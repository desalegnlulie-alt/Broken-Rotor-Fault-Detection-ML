import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import skew
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ==========================================
# 1. SETTINGS & PATHS
# ==========================================
# This matches your exact folder path
data_path = r'C:\Des\Academic\5th Year\2nd\ML\Project'
fs_current = 50000 
fs_vibe = 7600      

file_map = {
    'struct_rs_R1.mat': 0,   # Healthy
    'struct_r1b_R1.mat': 1,  # 1 Broken Bar
    'struct_r2b_R1.mat': 2,  # 2 Broken Bars
    'struct_r3b_R1.mat': 3,  # 3 Broken Bars
    'struct_r4b_R1.mat': 4   # 4 Broken Bars
}

# ==========================================
# 2. FEATURE EXTRACTION (The "Engineering" Part)
# ==========================================
def extract_signal_features(signal):
    # Time Domain
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    skw = skew(signal.flatten())
    peak_factor = peak / rms if rms != 0 else 0
    
    # Synthetic Envelope (Hilbert Transform) - REQ in PDF Section 3
    analytic = hilbert(signal)
    env = np.abs(analytic)
    env_mean = np.mean(env)
    env_std = np.std(env)
    
    # Frequency Domain (FFT Energy)
    fft_vals = np.abs(np.fft.fft(signal))
    freq_mean = np.mean(fft_vals)
    
    # Statistical Shape
    kurt = pd.Series(signal.flatten()).kurtosis()
    
    return [rms, peak, skw, peak_factor, env_mean, env_std, freq_mean, kurt]

# ==========================================
# 3. DATA PROCESSING LOOP (HDF5 DEREFERENCING)
# ==========================================
all_features = []
all_labels = []

print("--- Starting Random Forest Data Processing ---")

for filename, label in file_map.items():
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path):
        print(f"Skipping {filename}: Not found.")
        continue
    
    print(f"Deep-scanning {filename}...")
    
    with h5py.File(full_path, 'r') as f:
        # Get the internal struct name (rs, r1b, etc)
        main_key = [k for k in f.keys() if k != '#refs#'][0]
        main_struct = f[main_key]
        
        # Loop through all 8 load conditions (torque05, torque10, etc.)
        for t_key in main_struct.keys():
            t_group = main_struct[t_key]
            
            try:
                # DEREFERENCE: Follow HDF5 pointers to locate the 1.4GB signals
                current_raw = np.array(f[t_group['Ia'][0][0]]).flatten()
                vibe_raw = np.array(f[t_group['Vib_axial'][0][0]]).flatten()
            except Exception as e:
                continue

            # WINDOWING: 80 samples per torque = 3,200 total samples
            num_samples = 80 
            win_c = int(fs_current * 0.4) # 0.4 seconds window
            win_v = int(fs_vibe * 0.4)
            
            for i in range(num_samples):
                idx_c = i * (win_c // 2) # 50% overlap for better resolution
                idx_v = i * (win_v // 2)
                
                if (idx_c + win_c) > len(current_raw) or (idx_v + win_v) > len(vibe_raw):
                    break
                    
                # Extract 8 features from Current and 8 from Vibration
                f_c = extract_signal_features(current_raw[idx_c : idx_c + win_c])
                f_v = extract_signal_features(vibe_raw[idx_v : idx_v + win_v])
                
                # Sensor Fusion: Creating a combined 16-feature vector
                all_features.append(f_c + f_v)
                all_labels.append(label)

# ==========================================
# 4. MACHINE LEARNING PIPELINE
# ==========================================
X = np.array(all_features)
y = np.array(all_labels)

# Step A: Scaling (Standardize units between Current and Vibration)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step B: ANOVA selection (Top 15 features as per PDF Section 5)
selector = SelectKBest(f_classif, k=min(15, X_scaled.shape[1]))
X_selected = selector.fit_transform(X_scaled, y)

# Step C: Split 80% Train / 20% Test
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# Step D: Train Random Forest
print("Training Optimized Random Forest...")
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

# ==========================================
# 5. RESULTS & PLOT
# ==========================================
y_pred = model.predict(X_test)
final_acc = accuracy_score(y_test, y_pred)

print("\n" + "#"*40)
print(f"FINAL PROJECT ACCURACY: {final_acc*100:.2f}%")
print("#"*40)

# Confusion Matrix Heatmap
plt.figure(figsize=(10,7))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Healthy','1 Bar','2 Bars','3 Bars','4 Bars'],
            yticklabels=['Healthy','1 Bar','2 Bars','3 Bars','4 Bars'])
plt.title(f'Random Forest Results (Accuracy: {final_acc*100:.1f}%)')
plt.ylabel('Actual Fault State')
plt.xlabel('Predicted Fault State')
plt.show()