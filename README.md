# Voice Emotion Recognition using CNN, LSTM, and Attention 🗣️🎭

A Deep Learning project to classify human emotion from speech audio signals. This repository hosts a hybrid model combining **Convolutional Neural Networks (CNN)** for feature extraction, **Long Short-Term Memory (LSTM)** networks for temporal sequence modeling, and an **Attention mechanism** to focus on specific, emotionally significant parts of the audio signal.

## 📌 Project Overview

Speech Emotion Recognition (SER) is a challenging task due to the subjective nature of emotions and the complexity of audio data. This project processes raw audio to extract **MFCCs (Mel-frequency cepstral coefficients)**, treating them as visual spectrograms. These features are fed into a refined deep learning pipeline to predict 7 core emotions: *Anger, Disgust, Fear, Happiness, Neutrality, Sadness, and Surprise*.

The project workflow includes:
1.  **Data Aggregation:** Combining RAVDESS, SAVEE, and CREMA-D datasets.
2.  **Preprocessing:** Extracting MFCCs and padding sequences.
3.  **Augmentation:** Using noise injection and pitch shifting to reduce overfitting.
4.  **Modeling:** Training a hybrid CNN-LSTM-Attention architecture.

## 📂 Datasets Used

To ensure robust, speaker-independent generalization, this project combines multiple open-source datasets:

1.  **[RAVDESS](https://zenodo.org/record/1188976):** Ryerson Audio-Visual Database of Emotional Speech and Song.
2.  **[SAVEE](http://kahlan.eps.surrey.ac.uk/savee/):** Surrey Audio-Visual Expressed Emotion.
3.  **[CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D):** Crowd-Sourced Emotional Multimodal Actors Dataset.

*Note: The "Calm" emotion was filtered out during the data merging phase to ensure label consistency across all datasets.*

## 🛠️ Tech Stack

* **Language:** Python 3.x
* **Deep Learning:** TensorFlow / Keras
* **Audio Processing:** Librosa
* **Data Analysis:** NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn
* **Utilities:** Scikit-learn, Tqdm, Requests

## ⚙️ Model Architecture

The model treats audio features as images (spectrograms) and processes them sequentially:

1.  **Input Layer:** Accepts MFCC features (Shape: `40 x 174 x 1`).
2.  **CNN Block:** 3 layers of `Conv2D` + `BatchNormalization` + `MaxPooling` + `Dropout` to extract spatial features from the frequency domain.
3.  **Reshape Layer:** Flattens feature maps to prepare for the Recurrent Neural Network.
4.  **LSTM Layer:** A standard LSTM layer (128 units) captures time-dependent emotional cues.
5.  **Attention Mechanism:** A built-in Attention layer weighs specific time steps that carry the most emotional information.
6.  **Global Average Pooling:** Aggregates sequence data.
7.  **Output Layer:** Dense layer with `Softmax` activation for multi-class classification.

## 🚀 Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/voice-emotion-recognition.git](https://github.com/your-username/voice-emotion-recognition.git)
    cd voice-emotion-recognition
    ```

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas librosa tensorflow scikit-learn matplotlib tqdm requests
    ```

3.  **Prepare Data:**
    * The notebook includes a script to automatically download and extract the **RAVDESS** dataset.
    * For **SAVEE** and **CREMA-D**, ensure the datasets are downloaded and placed in folders named `savee_data` and `crema_d_data` respectively, or modify the path variables in the notebook.

4.  **Run the Notebook:**
    Open `Voice_Emotion_Recognition_using_CNN_and_LSTM.ipynb` in Jupyter Lab or Google Colab and run the cells sequentially to preprocess
