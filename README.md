# Smart MCQ Solver Challenge

## Student Information

- **Name:** Sharan Kumar E
- **Student ID:** 24f3004935

---

## Project Overview

This project develops an AI-powered Multiple Choice Question (MCQ) answering system using transformer-based language models. Given a question and its options, the model predicts the **top three most probable answers**.

The project explores:

- Data preprocessing
- Fine-tuning transformer models (RoBERTa, DeBERTa)
- Model evaluation
- Top-3 answer prediction
- Experiment tracking using Weights & Biases (W&B)

---

## Repository Structure

```
.
├── README.md
├── requirements.txt
│
├── main-notebook/
│   └── DL-24f3004935-notebook-t22.ipynb
│
├── milestones/
│   ├── milestone-1-ai.ipynb
│   ├── milestone-2-ai.ipynb
│   ├── milestone-3-ai.ipynb
│   ├── milestone-4-ai.ipynb
│   └── milestone-5.ipynb
│
└── wandb-runs/
    ├── deberta-training.ipynb
    ├── facebook-bert.ipynb
    └── rag-lexical.ipynb
```

---

## Folder Description

### `main-notebook/`

Contains the final notebook used for training, evaluation, and submission.

### `milestones/`

Contains notebooks submitted during different project milestones.

### `wandb-runs/`

Contains experimental notebooks, including:
- Facebook BERT experiments
- DeBERTa fine-tuning
- RAG and lexical retrieval experiments

---

## Models Explored

- Facebook ROBERT-LARGE
- DeBERTa
- LEXICAL Retrieval-Augmented Generation (RAG)-TFIDF based only

---

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- scikit-learn
- Pandas
- NumPy
- Weights & Biases (W&B)

---

## Installation

Clone the repository and install the required packages.

```bash
git clone <repository-url>
cd <repository-name>

pip install -r requirements.txt
```

---

## Results

The repository includes:
- Model training notebooks
- Validation experiments
- W&B experiment logs
- Final notebook used for submission

---

## Author

**Sharan Kumar E**

Student ID: **24f3004935**
