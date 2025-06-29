# Baby Cry Translator - Model Training and Conversion

## Overview

This repository contains the code for training a machine learning model to classify infant cries into different categories (belly pain, burping, discomfort, hungry, tired, and other). The trained model is then converted to TensorFlow Lite (TFLite) format for deployment in a mobile application.

## Model Architecture

The model uses a transfer learning approach with the following components:

1. **YAMNet Audio Feature Extraction**: 
   - Pre-trained YAMNet model from TensorFlow Hub extracts 1024-dimensional embeddings from audio input
   - Audio is resampled to 16kHz mono and padded to a fixed length

2. **Custom Classification Head**:
   - Dense layer with 256 units (ReLU activation)
   - Final dense layer with 6 units (one per class)

## Dataset

- Total samples: 1055
- Training samples: 844
- Validation samples: 105
- Test samples: 106

## Training Process

- Model trained for 150 epochs with early stopping (patience=9)
- Optimizer: Adam
- Loss: Sparse Categorical Crossentropy
- Batch size: 32
- Final validation accuracy: ~34.35%

## Model Conversion to TFLite

The trained model is converted to TFLite format with:
- Default optimizations
- Supported ops: TFLITE_BUILTINS and SELECT_TF_OPS
- Saved as `final_model.tflite`

## Usage

### Requirements
- TensorFlow 2.x
- TensorFlow Hub
- TensorFlow I/O
- NumPy

### Steps to Run

1. Prepare your dataset in the following structure:
   ```
   Aug_data/
     ├── belly_pain/
     ├── burping/
     ├── discomfort/
     ├── hungry/
     ├── Other/
     └── tired/
   ```
2. download the dataset folder Aug_data that have baby cry labeled data
3. Run the training notebook `yumnet_infant_Cry.ipynb`

3. The notebook will:
   - Preprocess the audio data
   - Train the model
   - Save the trained model
   - Convert to TFLite format

4. The final TFLite model will be saved as `final_model.tflite`

## Mobile App Integration

To integrate the TFLite model with your mobile app:

1. Add the `final_model.tflite` file to your app's assets folder

2. Use the TensorFlow Lite interpreter in your app to:
   - Load the model
   - Preprocess input audio (resample to 16kHz mono, same as training)
   - Run inference
   - Interpret the output probabilities

### Input Requirements
- Audio should be:
  - Mono channel
  - 16kHz sample rate
  - Same length as training data (padded if necessary)

### Output Interpretation
The model outputs probabilities for 6 classes:
1. belly_pain
2. burping
3. discomfort
4. hungry
5. tired
6. Other

## Performance Notes

- The model achieves moderate accuracy on the validation set
- Consider collecting more diverse data to improve performance
- The TFLite model is optimized for mobile deployment

## License
---

For questions or issues, please open an issue in this repository.
