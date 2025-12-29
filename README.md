# AI-Driven Insider Threat Detection System 🛡️🤖

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An intelligent, automated security system designed to detect malicious insider activities (data exfiltration, sabotage, espionage) by analyzing user behavior logs. This project moves beyond static rules by using **Behavioral Analytics** and **Hybrid Machine Learning** to flag anomalies with high precision.

---

## 📌 Project Overview

Insider threats are security risks that originate from within an organization—employees, contractors, or partners who have authorized access. Traditional firewalls and rule-based systems often fail to catch them because they look like normal users.

This project solves this by building a **"Behavioral Fingerprint"** for every user. It fuses data from multiple log sources (**Logon, File, Email, HTTP**) and uses a dual-model approach to detect threats:
1.  **XGBoost (Supervised):** Detects known threat patterns (e.g., massive file copying).
2.  **LSTM Autoencoder (Unsupervised):** Detects novel, unknown anomalies in sequential activity (e.g., unusual login times).

**Key Result:** The system achieved a **Recall of 97.7%** on the CERT v4.2 test dataset.

---

## 🚀 Features

* **Heterogeneous Data Fusion:** Merges Logon, File, Email, and Device logs into a unified user timeline.
* **Social Graph Analysis:** Uses **NetworkX** to detect suspicious email communication cliques or isolated interactions.
* **Hybrid AI Engine:** Combines the classification power of XGBoost with the sequence learning capabilities of LSTMs.
* **Heuristic Labeling:** Automatically generates ground-truth labels for training from unlabeled raw data using expert rules.
* **Automated Alerting:** Assigns a risk score (0-1) to every user-day.

---

## 🛠️ Tech Stack

* **Language:** Python 3.8+
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn, XGBoost
* **Deep Learning:** TensorFlow / Keras (LSTM)
* **Graph Analytics:** NetworkX
* **Visualization:** Matplotlib, Seaborn

---

## 📂 Dataset

This project uses the **CERT Insider Threat Dataset v4.2** created by the Software Engineering Institute at Carnegie Mellon University.
* *Note: Due to licensing, the dataset is not included in this repo. You can download it [here](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099).*

---

## ⚙️ Installation & Usage

1.  **Clone the Repository**
    ```bash
    git clone [https://github.com/yourusername/insider-threat-detection.git](https://github.com/yourusername/insider-threat-detection.git)
    cd insider-threat-detection
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Preprocess Data**
    Place your CERT dataset CSVs in the `data/` folder and run:
    ```bash
    python data_preprocess.py
    ```

4.  **Train Models**
    ```bash
    python train_xgboost.py
    python train_lstm.py
    ```

5.  **Run Evaluation**
    To see the metrics and generate ROC curves:
    ```bash
    python evaluate_and_visualize.py
    ```

6.  **Live Demo Script**
    To scan random users and see prediction scores:
    ```bash
    python demo_threat.py
    ```

---

## 📊 Results

| Metric | Score | Note |
| :--- | :--- | :--- |
| **Recall (Sensitivity)** | **97.7%** | Ensures almost zero missed threats. |
| **ROC-AUC** | 0.95 | High capability to distinguish threat vs safe. |
| **Precision** | ~9.5% | Focuses on minimizing False Negatives (Safety First). |

The system effectively flags malicious insiders while filtering out the vast majority of normal employees.

---

## 👥 Contributors

* **Poorna Chandra Tejaswi** (1EW23AI040)
* **Pavan S** (1EW23AI039)
* **Nikhith Gowda R** (1EW23AI036)
* **Abhayadithya N** (1EW23AI001)

**Guide:** Dr. Kavitha A S (HOD, Dept of AIML, EWIT)

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.