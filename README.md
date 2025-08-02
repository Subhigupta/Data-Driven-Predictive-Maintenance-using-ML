# 🧱Data-Driven-Predictive-Maintenance-using-ML
## 🧠 Use case

This project focuses on developing a predictive maintenance solution using supervised  ML algorithms. The developed solution enables users to predict using binary classifiers whether a machine will fail or not under a given set of process conditions. If a failure is predicted, using multinomial classifiers the user can classify into one of the following categories:
- Power Failure
- Tool Wear Failure
- Overstrain Failure
- Random Failure
- Heat Dissipation Failure

By anticipating potential failures, the solution supports proactive maintenance strategies, helping to minimize downtime, reduce maintenance costs, and maximize operational efficiency.

## 🚀 Getting Started

### 📦 1. Create a New Environment
Create a new environment and install all dependencies through the provided `.yaml` file.
```bash
conda env create -f environment.yaml
conda activate <your_env_name>
```
### 📦 2. Clone the repo
To start working on this project locally, clone this repository:
 ```bash
git clone https://github.com/Subhigupta/Data-Driven-Predictive-Maintenance-using-ML.git
```

## 📁 Project Structure Guide

This repository is organized into the following key directories:

### `data/`
Contains two synthetic datasets used for training and evaluation.

### `notebooks/`
This folder includes interactive Jupyter notebooks, implementing binary and multinomial classification:

**Model Training**
1. **Binary Classification**  
This notebook demonstrates the training of 5 supervised algorithms for a binary classification task. After evaluating the models, the two best-performing algorithms are selected based on their performance metrics.

2. **Multinomial Classification**  
This notebook demonstrates the training of 5 supervised algorithms for a multinomial classification task. As with binary classification, the two top-performing models are selected after evaluation.

### `templates/`
This directory contains HTML templates used for rendering the frontend pages of the web application.

### `models/`
Stores pre-trained hybrid models.

### `version1/`
This folder contains codes used at the initial phase of project development.
