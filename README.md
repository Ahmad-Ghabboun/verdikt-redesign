# Verdikt — Redesign

A full end-to-end data science pipeline predicting human preference
in LLM head-to-head battles using the LMSYS Chatbot Arena dataset.

## Live App


## GitHub


## How to Run Locally
1. Clone the repository
2. Install dependencies: pip install -r requirements.txt
3. Run the pipeline (optional, models are pre-trained):
   python pipeline.py
4. Launch the app: streamlit run app.py

## Dataset
LMSYS Chatbot Arena — Human Preference Predictions
https://www.kaggle.com/competitions/lmsys-chatbot-arena/data
Note: train.csv is excluded from this repo due to file size limits.
Download it from Kaggle and place it in lmsys-chatbot-arena/

## Models
All models are pre-trained and saved in the models/ folder.
No retraining is needed to run the app.

## Tech Stack
- Python 3.11
- Streamlit
- LightGBM, scikit-learn, Keras/TensorFlow
- SHAP, sentence-transformers
- Plotly
