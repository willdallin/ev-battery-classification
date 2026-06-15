# EV Battery Quality Classification

Machine learning project to predict the quality grade (Grade A, Grade B, Scrap) of Lithium-ion EV batteries based on environmental and physical manufacturing data. 

This project implements and compares four different classifiers to shift quality assurance from reactive end-of-line testing to predictive closed-loop control.

## Algorithms Evaluated
- Naive Bayes
- Decision Tree
- Support Vector Machine (SVM)
- Artificial Neural Network (ANN)

## Repository Structure
- `data/`: Contains the manufacturing dataset (`.csv`).
- `src/`: Source code for data preprocessing, model training, and evaluation.
- `images/`: Auto-generated confusion matrices and feature importance plots.
- `docs/`: Final academic report (PDF).

## Setup and Execution

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the classifiers:**
   ```bash
   python src/classifier.py
   ```

## Results
The **Decision Tree** and **Neural Network** achieved the highest accuracy (100%), successfully mapping the structural threshold of the electrolyte volume that causes severe internal resistance and battery failure. The Decision Tree is the recommended model for factory implementation due to its low computational cost and highly interpretable logic rules.