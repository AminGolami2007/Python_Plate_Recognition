# Plate Number Recognition System

A desktop application for recognizing Persian vehicle plate numbers from image files using image processing and a machine learning model.

## Overview

This project loads a dataset of plate character images, trains a Logistic Regression classifier, saves the trained model, and then recognizes the number from a user-selected plate image. The application includes a graphical interface so the user can choose an image, train the model, and view recognition results without using the terminal.

## Features

- User-friendly graphical interface with Tkinter
- Trains the classifier on the dataset if no saved model exists
- Loads an existing trained model automatically if available
- Saves the trained model to disk for later reuse
- Lets the user select a plate image from the local machine
- Displays the processed image and final recognized plate number
- Shows live training progress in the UI

## Project Structure

```text
Plate/
├── dataset/
│   ├── 1/
│   ├── 2/
│   ├── 3/
│   └── ...
├── index.py
├── plate_model.pkl
├── README.md
└── pelak.png
```

## Requirements

Install the required Python packages:

```bash
pip install opencv-python scikit-learn
```

## Run the Project

```bash
python index.py
```

Then:

1. Click "Select image"
2. Choose the plate image you want to recognize
3. Click "Train model" if no saved model exists
4. Or click "Load model" to use a previously saved model
5. Click "Recognize" to detect the plate number

## Model Training

The training process:

- loads image samples from the dataset
- converts each image to a flattened feature vector
- splits the data into training and test sets
- trains a Logistic Regression model
- saves the model in `plate_model.pkl`

## Notes

- The dataset should be organized in numbered folders such as `dataset/1`, `dataset/2`, etc.
- The project is designed for easy experimentation and educational use.
- The model file `plate_model.pkl` is generated automatically after training.

## License

This project is provided for educational and research purposes.
