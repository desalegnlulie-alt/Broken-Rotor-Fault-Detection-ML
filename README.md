# 🛠️ AI Motor Doctor: Broken Rotor Fault Detection
**Addis Ababa Science and Technology University (AASTU)**  
**Department of Electromechanical Engineering**

This project implements an intelligent, automated diagnostic system to detect and classify **Broken Rotor Bar (BRB)** faults in AC induction motors using Multi-Sensor Data Fusion.

---

## 📖 Project Overview
Broken rotor bars are "hidden" internal faults. Our system acts as a **Digital Doctor**, using Machine Learning to "listen" to the motor's heartbeat (Current) and "feel" its movement (Vibration) to diagnose damage severity from Healthy to 4 Broken Bars.

### 🌟 Achievement
*   **Best Model:** Random Forest (Manual Feature Engineering)
*   **Peak Accuracy:** **96.56%**
*   **Validation:** Cross-platform verified in both **Python** and **MATLAB**.

---

## 🏗️ Technical Architecture
*   **Multi-Sensor Fusion:** Fusing Stator Current ($I_a$) and Axial Vibration data.
*   **Physics-Aware Processing:** Applying **Hilbert Transform** to extract Synthetic Envelopes and **FFT** for frequency sideband analysis.
*   **Big Data Handling:** Processing **7GB of HDF5 data** using Lazy Loading and custom dereferencing logic.
*   **Feature Selection:** Utilizing **One-way ANOVA** to pick the top 15 most significant physical indicators.

---

## 📊 Model Comparison Results
| Model | Final Accuracy | Logic Type |
| :--- | :--- | :--- |
| **Random Forest** | **96.56%** | Hand-crafted Features (Expert) |
| **Enhanced 1D-CNN** | **91.70%** | Automated Feature Learning (Deep) |
| **DQN (Reinforcement Learning)** | **88.50%** | Adaptive Reward Learning |

---

## 📂 Dataset Info
Due to GitHub's 100MB file limit, the 7GB IEEE Dataset (.mat files) is hosted on Google Drive:
👉 **[Download Dataset Here][PASTE_YOUR_LINK_HERE](https://drive.google.com/file/d/1Z1vIDGscYepWEBAvQt27sWS7GF_rFUhA/view?usp=sharing)**

---

## 💻 Tech Stack
*   **Languages:** Python 3.11, MATLAB R2023a
*   **AI Libraries:** Scikit-Learn, TensorFlow, Keras
*   **Data Handling:** h5py, Pandas, Numpy, SciPy
*   **Visualization:** Matplotlib, Seaborn

---

## 🛠️ Installation
1. **Enable Windows Long Path Support** via PowerShell.
2. **Install requirements:**
   ```bash
   pip install h5py scikit-learn tensorflow pandas numpy matplotlib seaborn scipy
