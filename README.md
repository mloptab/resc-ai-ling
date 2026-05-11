# Predicting Language Model Performance from Technical Characteristics

This project presents a machine learning framework to estimate the performance of large language models (LLMs) on benchmark tasks using only their technical characteristics.

The objective is to predict benchmark scores without running costly evaluations, enabling faster and more efficient model comparison.

---

## Deployment and Inference Demo

The repository includes an inference demo that allows users to test the trained pipeline with new model characteristics.

The demo resources are stored in the `deployment/` folder and include:

- `RESC-AI-LING_Inference_Demo.ipynb`: notebook for running inference examples.
- `demo/`: additional files related to the interactive demo.

This part of the project shows how the trained model can be used beyond the exploratory analysis stage.

## Authors

- María López Tabellion
- Camilo Morales Flores
- Héctor Beltrán Pozo
- Alexa Medina Ceballos
- Sergio Samaniego Hernández
- Marcos Carrasco Panadero

**Universitat Politècnica de València (UPV)**

---

## Project Structure

```text
resc-ai-ling/
├── data/
│   ├── raw/          # Original source datasets
│   ├── interim/      # Intermediate datasets
│   └── processed/    # Final cleaned datasets
├── notebooks/        # Data preprocessing, EDA and modelling notebooks
├── deployment/       # Inference notebook and demo resources
├── README.md
├── requirements.txt
└── .gitignore

