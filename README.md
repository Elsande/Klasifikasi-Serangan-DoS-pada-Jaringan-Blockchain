# 🛡️ DoS Attack Classification on Blockchain Network using PPO

Model klasifikasi serangan **Denial of Service (DoS)** pada jaringan blockchain menggunakan algoritma **Proximal Policy Optimization (PPO)** berbasis arsitektur **Actor–Critic**.

Penelitian ini memformulasikan tugas klasifikasi sebagai masalah **Reinforcement Learning (RL)** untuk meningkatkan kemampuan adaptif terhadap pola serangan dinamis pada jaringan blockchain.

---

## 📌 Background

Blockchain bersifat terdesentralisasi, namun tetap rentan terhadap serangan **Denial of Service (DoS)** yang menargetkan aspek *availability* melalui pembanjiran lalu lintas jaringan ke node.

Pendekatan supervised learning konvensional memiliki keterbatasan dalam menangani pola serangan baru. Oleh karena itu, penelitian ini mengusulkan pendekatan **Reinforcement Learning berbasis PPO** yang:

- Adaptif terhadap pola lalu lintas dinamis
- Stabil dalam proses pembaruan kebijakan
- Efisien dalam optimisasi parameter
- Tidak bergantung pada signature-based detection

---

## 🧠 Dataset

**Nama Dataset:** Blockchain Network Attack Traffic (BNaT)  
**Lingkungan:** Ethereum Private Network  

### Statistik Dataset
- Total awal: 210.000 sampel
- Setelah filtering & cleaning: 93.670 sampel
- Jumlah fitur: 21 fitur prediktor + 1 label
- Kelas:
  - `0` → Normal
  - `1` → DoS

### Distribusi Akhir
- Normal: 84%
- DoS: 16%

---

## ⚙️ Data Preprocessing

1. Penggabungan 3 file hasil capture dari Ethereum full nodes
2. Penghapusan 71.330 data duplikat
3. Identifikasi fitur:
   - 18 fitur numerik
   - 3 fitur kategorikal (`protocol_type`, `service`, `flag`)
4. Normalisasi fitur numerik menggunakan **Z-score (StandardScaler)**
5. Encoding fitur kategorikal menggunakan **Ordinal Encoding**
6. Stratified Train-Test Split (80:20)

---

## 🏗️ Model Architecture

Model menggunakan pendekatan **Actor–Critic PPO** dengan:

- 2 Hidden Layers (ReLU Activation)
- Clipped Surrogate Objective
- Hybrid Loss Function:
  - Policy Loss
  - Value Loss
  - Entropy Loss
  - Cross-Entropy Loss (Supervised-RL Hybrid)

Pendekatan ini menjaga keseimbangan antara:
- Eksplorasi
- Eksploitasi
- Stabilitas pembaruan kebijakan

Framework: **PyTorch**

---

## 🔧 Best Hyperparameters

| Parameter | Optimal Value |
|------------|---------------|
| Learning Rate | 0.0003 |
| Clip Epsilon | 0.12 |
| Entropy Coefficient | 0.005 |
| Batch Size | 256 |
| Value Coefficient | 0.7 |
| K-Fold | 2 |

Total eksperimen: **64 kombinasi hyperparameter**

---

## 📊 Evaluation Results

### 🔹 Overall Performance (Test Set)

| Metric | Score |
|--------|--------|
| Accuracy | **0.9965** |
| Precision | 0.9965 |
| Recall | 0.9965 |
| F1-Score | **0.9965** |
| AUC | **0.9999** |
| Average Precision | 0.9993 |

### 🔹 Per-Class Performance

| Label | Precision | Recall | F1-Score |
|--------|-----------|--------|----------|
| Normal | 0.9964 | 0.9994 | 0.9979 |
| DoS | 0.9969 | 0.9807 | 0.9887 |

### 📈 Additional Evaluation

- Confusion Matrix
- ROC Curve (AUC = 0.9999)
- Precision-Recall Curve (AP = 0.9993)
- Training Accuracy Convergence
- Loss Stabilization Analysis
- Policy Entropy Dynamics

Model menunjukkan:
- False Positive Rate sangat rendah
- Separabilitas kelas sangat tinggi
- Tidak terdapat indikasi overfitting
- Konvergensi pelatihan stabil

---

## 🖥️ Project Structure (Example)

```
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   └── ppo_model.py
├── training/
│   └── train.py
├── evaluation/
│   └── evaluate.py
├── utils/
├── results/
├── requirements.txt
└── README.md
```

---

## 🚀 Installation

Clone repository:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Training

```bash
python train.py
```

---

## 📈 Evaluation

```bash
python evaluate.py
```

---

## 🔮 Future Work

- Multi-class attack classification (BP, FoT, MitM, DoS)
- Integration with Graph Neural Network (GNN)
- Transformer-based architecture
- Automated hyperparameter optimization
- Real-time blockchain traffic evaluation
- Deployment-ready intrusion detection pipeline

---

## 📚 References

T. V. Khoa et al.,  
**Collaborative Learning for Cyberattack Detection in Blockchain Networks**,  
IEEE Transactions on Systems, Man, and Cybernetics, 2024.

---

## 👤 Author

**Iffo Elsande Pratama Putra**  
Informatics Engineering  
Universitas Negeri Surabaya  

---

## 📜 License

This project is intended for academic and research purposes.
