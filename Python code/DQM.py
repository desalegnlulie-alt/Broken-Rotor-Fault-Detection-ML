import os
import h5py
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.stats import skew
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import seaborn as sns
import random

# ==========================================
# 1. SETTINGS & FEATURE EXTRACTION
# ==========================================
data_path = r'C:\Des\Academic\5th Year\2nd\ML\Project'
file_map = {'struct_rs_R1.mat': 0, 'struct_r1b_R1.mat': 1, 
            'struct_r2b_R1.mat': 2, 'struct_r3b_R1.mat': 3, 
            'struct_r4b_R1.mat': 4}

fs_current = 50000 
fs_vibe = 7600      

def extract_signal_features(signal):
    rms = np.sqrt(np.mean(signal**2))
    peak = np.max(np.abs(signal))
    skw = skew(signal.flatten())
    analytic = hilbert(signal)
    env = np.abs(analytic)
    fft_vals = np.abs(np.fft.fft(signal))
    kurt = pd.Series(signal.flatten()).kurtosis()
    return [rms, peak, skw, np.mean(env), np.std(env), np.mean(fft_vals), kurt]

# ==========================================
# 2. DATA LOADING & DEREFERENCING (Creates X and y)
# ==========================================
all_features, all_labels = [], []
print("--- Extracting Features for Reinforcement Learning ---")

for filename, label in file_map.items():
    full_path = os.path.join(data_path, filename)
    if not os.path.exists(full_path): continue
    
    with h5py.File(full_path, 'r') as f:
        main_key = [k for k in f.keys() if k != '#refs#'][0]
        main_struct = f[main_key]
        for t_key in main_struct.keys():
            t_group = main_struct[t_key]
            try:
                curr_raw = np.array(f[t_group['Ia'][0][0]]).flatten()
                vibe_raw = np.array(f[t_group['Vib_axial'][0][0]]).flatten()
                
                win_c, win_v = int(fs_current * 0.4), int(fs_vibe * 0.4)
                for i in range(50): # 50 samples per torque level
                    start_c, start_v = i * (win_c // 2), i * (win_v // 2)
                    if (start_c + win_c) > len(curr_raw): break
                    f_c = extract_signal_features(curr_raw[start_c : start_c + win_c])
                    f_v = extract_signal_features(vibe_raw[start_v : start_v + win_v])
                    all_features.append(f_c + f_v)
                    all_labels.append(label)
            except: continue

# Create the missing variables (X_train, y_train)
X = np.array(all_features)
y = np.array(all_labels)
X_scaled = StandardScaler().fit_transform(X)
X_selected = SelectKBest(f_classif, k=min(14, X_scaled.shape[1])).fit_transform(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# ==========================================
# 3. THE RL AGENT (DQN)
# ==========================================
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = 1.0   
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self._build_model()

    def _build_model(self):
        model = models.Sequential([
            layers.Dense(64, input_dim=self.state_size, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=optimizers.Adam(learning_rate=0.001))
        return model

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])

# ==========================================
# 4. TRAINING THE AGENT
# ==========================================
agent = DQNAgent(X_train.shape[1], 5)
batch_size = 32
epochs = 30 

print("\n--- RL Agent is now learning from Motor Signals ---")

for e in range(epochs):
    indices = np.random.choice(len(X_train), batch_size)
    batch_x, batch_y = X_train[indices], y_train[indices]
    
    for i in range(batch_size):
        state = batch_x[i].reshape(1, -1)
        action = agent.act(state)
        reward = 1 if action == batch_y[i] else -1
        
        target = reward 
        target_f = agent.model.predict(state, verbose=0)
        target_f[0][action] = target
        agent.model.fit(state, target_f, epochs=1, verbose=0)
        
    if agent.epsilon > agent.epsilon_min:
        agent.epsilon *= agent.epsilon_decay
    print(f"Epoch {e+1}/{epochs} - Confidence: {(1-agent.epsilon)*100:.1f}%")

# ==========================================
# 5. EVALUATION
# ==========================================
y_pred = [agent.act(x.reshape(1, -1)) for x in X_test]
print(f"\nFinal RL Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Purples',
            xticklabels=['H','1b','2b','3b','4b'], yticklabels=['H','1b','2b','3b','4b'])
plt.title('DQN Reinforcement Learning: Final Results')
plt.show()