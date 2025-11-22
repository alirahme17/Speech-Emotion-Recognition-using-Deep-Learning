Voice Emotion Recognition using CNN, LSTM, and Attention 🗣️🎭
A Deep Learning project to classify human emotion from speech audio signals. This repository contains a hybrid model combining Convolutional Neural Networks (CNN) for feature extraction, Long Short-Term Memory (LSTM) networks for temporal sequence modeling, and an Attention mechanism to focus on specific parts of the audio signal.

📌 Project Overview
Speech Emotion Recognition (SER) is a challenging task due to the subjective nature of emotions and the complexity of audio data. This project utilizes MFCCs (Mel-frequency cepstral coefficients) as input features and processes them through a refined deep learning pipeline to predict emotions like Anger, Happiness, Sadness, Fear, and more.

The project evolves through several stages:

Baseline: Training on the RAVDESS dataset.

Improvement: Implementing Data Augmentation (Noise injection, Pitch shifting).

Scaling: Combining multiple datasets (RAVDESS, SAVEE, CREMA-D) for speaker-independent generalization.

Architecture: Implementing a CNN-LSTM-Attention hybrid model.

📂 Datasets Used
This project combines multiple open-source datasets to create a robust training set.

RAVDESS: Ryerson Audio-Visual Database of Emotional Speech and Song.

SAVEE: Surrey Audio-Visual Expressed Emotion.

CREMA-D: Crowd-Sourced Emotional Multimodal Actors Dataset.

Note: The "Calm" emotion was filtered out during the combination phase to ensure label consistency across datasets, resulting in 6-7 core emotion classes.

🛠️ Tech Stack
Language: Python 3.x

Deep Learning: TensorFlow / Keras

Audio Processing: Librosa

Data Manipulation: NumPy, Pandas

Visualization: Matplotlib, Seaborn

Utilities: Scikit-learn, Tqdm, Requests

⚙️ Model Architecture
The model treats audio MFCCs as images (spectrograms) and processes them sequentially:

Input Layer: Accepts MFCC features (Shape: 40 x 174 x 1).

CNN Block: 3 layers of Conv2D + BatchNormalization + MaxPooling + Dropout to extract spatial features from the spectrograms.

Reshape Layer: Flattens the feature maps to prepare for the RNN.

LSTM Layer: A Bidirectional or standard LSTM layer (128 units) to capture time-dependent emotional cues.

Attention Mechanism: A custom or built-in Attention layer to weigh specific time steps that are more emotionally significant.

Global Average Pooling: Aggregates the sequence data.

Output Layer: Dense layer with Softmax activation for multi-class classification.

🚀 Installation & Usage
Clone the repository:

Bash

git clone https://github.com/your-username/voice-emotion-recognition.git
cd voice-emotion-recognition
Install dependencies:

Bash

pip install numpy pandas librosa tensorflow scikit-learn matplotlib tqdm requests
Prepare Data:

The notebook contains a script to automatically download and extract the RAVDESS dataset.

For SAVEE and CREMA-D, ensure the datasets are downloaded and placed in folders named savee_data and crema_d_data respectively, or adjust the paths in the notebook.

Run the Notebook: Open Voice_Emotion_Recognition_using_CNN_and_LSTM.ipynb in Jupyter Lab or Google Colab and run the cells sequentially.

📊 Methodology Highlights
1. Feature Extraction
We utilize Librosa to extract MFCCs (Mel-frequency cepstral coefficients).

Sampling Rate: 22050 Hz

MFCCs: 40

Padding: Signals are padded or truncated to a fixed length of 174 time steps to ensure uniform input for the CNN.

2. Data Augmentation
To prevent overfitting and improve robustness, the training data is augmented using:

Noise Injection: Adding random Gaussian noise.

Pitch Shifting: Altering the pitch of the voice without changing the speed.

3. Training Strategy
Optimizer: Adam

Loss Function: Categorical Crossentropy

Callbacks:

ModelCheckpoint: Saves the best model based on validation accuracy.

EarlyStopping: Prevents overfitting by stopping when validation loss plateaus.

ReduceLROnPlateau: Lowers learning rate when progress stalls.

📈 Performance
The model performance varies based on the complexity of the dataset mix:

RAVDESS Only: ~60-65% Accuracy.

Combined (Speaker Independent Split): ~50-55% Accuracy.

Note: Speaker-independent testing (where the model has never heard the test speakers before) is significantly harder than random splitting but provides a more realistic metric for real-world usage.

🔮 Future Improvements
Transfer Learning: Implementing Wav2Vec 2.0 or HuBERT embeddings.

Hyperparameter Tuning: Using KerasTuner to optimize layer counts and dropout rates.

Real-time Prediction: Creating a script to record microphone input and predict emotion live.



