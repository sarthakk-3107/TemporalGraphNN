🫀 Temporal Graph Neural Network for Heart Disease Prediction
📘 Overview


This project implements a Temporal Graph Neural Network (TGN) using PyTorch Geometric to model evolving relationships among patients and clinical features. The model predicts heart disease progression by leveraging temporal dependencies and relational structures in Electronic Health Record (EHR) data.

🧠 Key Features
Temporal modeling of patient states using TGN memory modules.
Graph construction from heterogeneous EHR data (ECG, cholesterol, blood pressure, BMI, glucose).
GPU-optimized training for large temporal graphs.
Evaluation using AUC-ROC, Precision, Recall, and F1-Score.
Comparative analysis with Graph Attention Networks (GATs) and Temporal CNNs.


├── data/
│   ├── features.csv        # Node features (clinical attributes)
│   ├── edges.csv           # Temporal relationships between patients
│   ├── labels.csv          # Heart disease outcomes
├── src/
│   ├── preprocess.py       # Data cleaning & feature normalization
│   ├── model.py            # TGN model architecture
│   ├── train.py            # Training & validation pipeline
│   ├── evaluate.py         # Metrics & visualization
│   └── utils.py            # Helper functions
├── Untitled48.ipynb        # Notebook for experimentation
└── README.md               # Project documentation


📊 Results
Validation Accuracy: ~86%
Model generalizes well across unseen temporal data windows.
TGN outperforms static GNNs in capturing evolving patient dynamics.

🧩 Future Work
Integration of clinical text embeddings (from patient notes).
Multi-modal fusion of imaging and physiological data.
Deployment of explainability tools for clinical interpretability.
