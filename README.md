# fake-news-detection
Fake News Detector using Machine Learning and Streamlit for interactive testing and deployment.
# Fake News Detector with Machine Learning

This project uses a machine learning model to detect whether a news text is real or fake. I built it to understand the end-to-end pipeline of cleaning data, vectorizing, training a model, and testing it using a simple UI.

## Features

- Check if a news text is real or fake
- Shows confidence score for prediction
- Clean and simple interface with Streamlit

## How it works

The project uses a dataset of fake and real news articles to train a machine learning model. It uses TF-IDF for feature extraction and a Logistic Regression classifier to perform the prediction. The app is created with Streamlit, making it easy to test different news samples.

## Project Structure

- `app.py`: Streamlit app for testing news authenticity.
- `models/`: Contains trained model files.
- `requirements.txt`: Required packages.

## Running Locally

1. Clone the repository:
    ```bash
    git clone <repo-link>
    cd fake-news-detection-live-feed
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the app:
    ```bash
    streamlit run app.py
    ```

## Notes

- Accuracy may vary based on dataset and text samples.
- This project is for learning how to deploy ML models with a simple interface.

## Future Improvements

- Use transformer models for better accuracy
- Add news scraping for live testing
- Improve dataset for headline-based detection

---

### Built by Khushi Nagpure

For any queries or feedback, feel free to connect with me on LinkedIn.

