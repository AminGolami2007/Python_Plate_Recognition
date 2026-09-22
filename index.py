import os
import pickle
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ===== CONFIG =====
BASE_SIZE_X = 8
BASE_SIZE_Y = 32
BASE_SIZE = BASE_SIZE_X * BASE_SIZE_Y
DEFAULT_PLATE_IMAGE = "pelak.png"
MODEL_PATH = "plate_model.pkl"
DATASET_ROOT = "dataset"
MIN_WIDTH = 15
MAX_WIDTH = 50
LINE_THRESHOLD = 5


# ===== IMAGE LOADER =====
def load_flatten_image(image_path):
    """Load an image and convert it into a 1D feature vector."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = cv2.resize(img, (BASE_SIZE_X, BASE_SIZE_Y))
    img = img.astype(np.float32) / 255.0
    return img.flatten()


# ===== DATASET LOAD =====
def load_dataset(dataset_root=DATASET_ROOT):
    """Collect all training images and labels from the dataset folders."""
    images = []
    labels = []

    for folder_name in sorted(os.listdir(dataset_root)):
        folder_path = os.path.join(dataset_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in sorted(os.listdir(folder_path)):
            if not file_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                continue

            image_path = os.path.join(folder_path, file_name)
            images.append(load_flatten_image(image_path))
            labels.append(int(folder_name))

    if not images:
        raise ValueError(f"No image files were found in the dataset folder: {dataset_root}")

    return np.array(images), np.array(labels)


# ===== MODEL TRAIN / LOAD =====
def train_model(update_status=None):
    """Train the model, show progress, and save it for future reuse."""
    if update_status is not None:
        update_status("Training: loading dataset...")

    X, y = load_dataset()

    if update_status is not None:
        update_status("Training: splitting data into training and validation sets...")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    if update_status is not None:
        update_status("Training: fitting Logistic Regression model...")

    model = LogisticRegression(max_iter=100000)
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    if update_status is not None:
        update_status(f"Training: accuracy = {accuracy:.4f}")

    if update_status is not None:
        update_status("Training: saving model to disk...")

    with open(MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    if update_status is not None:
        update_status(f"Training complete: model saved in {MODEL_PATH}")

    return model


def load_model():
    """Load the saved model if it exists."""
    with open(MODEL_PATH, "rb") as model_file:
        return pickle.load(model_file)


def load_or_train_model(update_status=None):
    """Load a saved model if it exists; otherwise train a new one and save it."""
    if os.path.exists(MODEL_PATH):
        if update_status is not None:
            update_status("Loading saved model...")
        model = load_model()
        if update_status is not None:
            update_status("Model loaded successfully.")
        return model

    if update_status is not None:
        update_status("No saved model found. Training a new model...")
    return train_model(update_status)


# ===== PREPROCESS PLATE =====
def preprocess_plate(image_path):
    """Resize, grayscale, threshold, and project the plate image for segmentation."""
    plate_base = cv2.imread(image_path)
    if plate_base is None:
        raise FileNotFoundError(f"Plate image not found: {image_path}")

    plate_base = cv2.resize(plate_base, (410, 90))
    plate_gray = cv2.cvtColor(plate_base, cv2.COLOR_BGR2GRAY)
    plate_gray = cv2.GaussianBlur(plate_gray, (3, 3), 0)
    _, plate_binary = cv2.threshold(plate_gray, 64, 255, cv2.THRESH_BINARY)
    plate_binary = cv2.GaussianBlur(plate_binary, (3, 3), 0)

    projection = 90 - np.sum(plate_binary, axis=0) / 255.0
    return plate_base, plate_binary, projection


# ===== SEGMENTATION =====
def recognize_plate(model, image_path):
    """Segment the plate into character regions and predict the final text."""
    plate_base, plate_binary, projection = preprocess_plate(image_path)

    final_number = ""
    prev_value = 0
    start_index = 0

    for index, plot_value in enumerate(projection):
        if plot_value >= LINE_THRESHOLD and prev_value < LINE_THRESHOLD:
            start_index = index

        if plot_value < LINE_THRESHOLD and prev_value >= LINE_THRESHOLD:
            end_index = index
            width = end_index - start_index

            if MIN_WIDTH < width < MAX_WIDTH:
                cv2.rectangle(plate_base, (start_index, 0), (end_index, 89), (255, 255, 0), 1)

                char_image = plate_binary[:, start_index:end_index]
                char_image = cv2.resize(char_image, (BASE_SIZE_X, BASE_SIZE_Y))
                char_image = char_image.astype(np.float32).flatten() / 255.0

                prediction = model.predict([char_image])[0]
                final_number += str(prediction)

        prev_value = plot_value

    return final_number, plate_base, plate_binary


# ===== SHOW RESULT =====
def show_result(plate_base, plate_binary, final_number):
    """Display the processed plate and the final recognized value in a graphical window."""
    result_image = np.ones((50, plate_base.shape[1], 3), np.uint8) * 255
    plate_base = np.vstack([plate_base, result_image])

    cv2.putText(
        plate_base,
        "Plate: " + final_number,
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 0, 255),
        1,
    )

    cv2.imshow("Plate", plate_base)
    cv2.imshow("Binary", plate_binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===== GRAPHICAL USER INTERFACE =====
class PlateRecognizerApp:
    """Graphical interface for selecting a plate image, training the model, and recognizing the number."""

    def __init__(self, root):
        self.root = root
        self.root.title("Plate Number Recognizer")
        self.root.geometry("620x320")
        self.root.resizable(False, False)

        self.model = None
        self.image_path = DEFAULT_PLATE_IMAGE
        self.training_thread = None

        tk.Label(root, text="Plate image path:", font=("Arial", 10, "bold")).pack(pady=(18, 6))
        self.image_label = tk.StringVar(value=self.image_path)
        tk.Entry(root, textvariable=self.image_label, width=60, state="readonly").pack()

        button_frame = tk.Frame(root)
        button_frame.pack(pady=12)

        tk.Button(button_frame, text="Select image", width=14, command=self.select_image).grid(row=0, column=0, padx=8)
        tk.Button(button_frame, text="Train model", width=14, command=self.start_training).grid(row=0, column=1, padx=8)
        tk.Button(button_frame, text="Load model", width=14, command=self.load_model).grid(row=0, column=2, padx=8)
        tk.Button(button_frame, text="Recognize", width=14, command=self.recognize_selected_image).grid(row=0, column=3, padx=8)

        self.status_var = tk.StringVar(value="Status: waiting for action")
        tk.Label(root, textvariable=self.status_var, font=("Arial", 10), fg="darkblue", wraplength=560, justify="center").pack(pady=(8, 0))

        # Try to load the saved model automatically at startup.
        self.root.after(200, self.auto_load_or_train)

    def update_status(self, message):
        """Update the UI status text safely from any thread."""
        self.root.after(0, lambda: self.status_var.set(message))

    def select_image(self):
        """Open a file dialog and let the user select a plate image."""
        selected_file = filedialog.askopenfilename(
            title="Select plate image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")],
        )
        if selected_file:
            self.image_path = selected_file
            self.image_label.set(selected_file)
            self.update_status("Status: image selected")

    def load_model(self):
        """Load the saved model from disk."""
        try:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

            self.update_status("Loading saved model...")
            with open(MODEL_PATH, "rb") as model_file:
                self.model = pickle.load(model_file)
            self.update_status("Status: model loaded successfully")
        except Exception as error:
            self.update_status(f"Status: {error}")
            messagebox.showerror("Load error", str(error))

    def start_training(self):
        """Run training in a separate thread so the UI can display progress in real time."""
        if self.training_thread is not None and self.training_thread.is_alive():
            self.update_status("Status: training is already in progress")
            return

        self.training_thread = threading.Thread(target=self.train_in_background, daemon=True)
        self.training_thread.start()

    def train_in_background(self):
        """Background training process that updates the status step by step."""
        try:
            self.update_status("Training: loading dataset...")
            X, y = load_dataset()

            self.update_status("Training: splitting data into train/test sets...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            self.update_status("Training: fitting Logistic Regression model...")
            model = LogisticRegression(max_iter=100000)
            model.fit(X_train, y_train)

            accuracy = model.score(X_test, y_test)
            self.update_status(f"Training: accuracy = {accuracy:.4f}")

            self.update_status("Training: saving model to disk...")
            with open(MODEL_PATH, "wb") as model_file:
                pickle.dump(model, model_file)

            self.model = model
            self.update_status(f"Training complete: model saved in {MODEL_PATH} | accuracy = {accuracy:.4f}")
            messagebox.showinfo("Training complete", f"Model trained successfully. Accuracy: {accuracy:.4f}")
        except Exception as error:
            self.update_status(f"Status: training failed - {error}")
            messagebox.showerror("Training error", str(error))

    def auto_load_or_train(self):
        """Automatically loads a saved model if it exists; otherwise starts the training workflow."""
        if os.path.exists(MODEL_PATH):
            self.load_model()
        else:
            self.update_status("Status: no saved model found. Press 'Train model' to start training.")

    def recognize_selected_image(self):
        """Recognize the selected plate image using the current model."""
        if self.model is None:
            self.update_status("Status: no model loaded yet")
            messagebox.showwarning("No model", "Please train or load a model before recognition.")
            return

        if not self.image_path:
            self.update_status("Status: no image selected")
            messagebox.showwarning("No image", "Please select an image first.")
            return

        try:
            self.update_status("Recognizing plate image...")
            final_number, plate_base, plate_binary = recognize_plate(self.model, self.image_path)
            self.update_status(f"Status: recognized plate = {final_number}")
            show_result(plate_base, plate_binary, final_number)
        except FileNotFoundError:
            self.update_status("Status: the selected image file could not be found")
            messagebox.showerror("File error", "The selected image file could not be found.")
        except Exception as error:
            self.update_status(f"Status: recognition failed - {error}")
            messagebox.showerror("Recognition error", str(error))


# ===== MAIN =====
def main():
    """Launch the graphical user interface."""
    root = tk.Tk()
    PlateRecognizerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
