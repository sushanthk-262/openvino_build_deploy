import cv2
import numpy as np
import requests
from openvino import Core
import logging
from pathlib import Path
import time
import collections
import os
import sys
import random
import pygame
import traceback
from win10toast import ToastNotifier

# Initialize logging
logging.basicConfig(level=logging.DEBUG)

def download_model(model_name, precision):
    """Download the required OpenVINO model if not already present."""
    base_model_dir = Path("model")
    model_dir = base_model_dir / "intel" / model_name / precision
    model_path = model_dir / f"{model_name}.xml"
    bin_path = model_dir / f"{model_name}.bin"

    if not model_path.exists() or not bin_path.exists():
        model_dir.mkdir(parents=True, exist_ok=True)
        xml_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/{model_name}.xml"
        bin_url = f"https://storage.openvinotoolkit.org/repositories/open_model_zoo/2022.1/models_bin/3/{model_name}/{precision}/{model_name}.bin"

        logging.info(f"Downloading model from {xml_url}")
        with requests.get(xml_url, stream=True) as r:
            r.raise_for_status()
            with open(model_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        logging.info(f"Downloading model from {bin_url}")
        with requests.get(bin_url, stream=True) as r:
            r.raise_for_status()
            with open(bin_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    logging.info(f"Using model at {model_path}")
    return model_path

def load_model(model_path, device):
    """Load the OpenVINO model."""
    core = Core()
    model = core.read_model(model=model_path)
    compiled_model = core.compile_model(model=model, device_name=device)

    input_layer = compiled_model.input(0)
    output_layers = [compiled_model.output(i) for i in range(len(compiled_model.outputs))]  # Handle multiple outputs

    logging.info(f"Loaded model: {model_path}")
    logging.info(f"Input layer shape: {input_layer.shape}")
    for i, output_layer in enumerate(output_layers):
        logging.info(f"Output layer {i} shape: {output_layer.shape}")

    return compiled_model, input_layer, output_layers

def detect_emotions(frame, model, input_layer, output_layer):
    """Detect emotions using the emotion recognition model."""
    input_img = cv2.resize(frame, (input_layer.shape[2], input_layer.shape[3]))
    input_img = input_img.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)  # Shape: (1, 3, H, W)

    results = model([input_img])[output_layer[0]]  # Access the first (and only) output layer
    emotion = np.argmax(results)  # Get the emotion with the highest confidence

    return emotion

def play_music(emotion, music_dir):
    """Play music based on the detected emotion."""
    emotion_dirs = ["happy", "sad", "angry", "neutral","surprise"]
    emotion_dir = os.path.join(music_dir, emotion_dirs[emotion])
    music_files = [os.path.join(emotion_dir, f) for f in os.listdir(emotion_dir) if f.endswith('.mp3')]

    if music_files:
        pygame.mixer.init()
        pygame.mixer.music.load(random.choice(music_files))
        pygame.mixer.music.play()

def run_demo(source, emotion_model_name, model_precision, device, music_dir):
    """Run the emotion recognition demo."""
    cap = None
    emotion_buffer = collections.deque(maxlen=10)  # Buffer to store emotions for 10 seconds
    toaster = ToastNotifier()  # Initialize the toaster for notifications
    try:
        # Load model
        emotion_model_path = download_model(emotion_model_name, model_precision)
        emotion_model, emotion_input, emotion_output = load_model(emotion_model_path, device)

        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)

        if not cap.isOpened():
            logging.error("Failed to open video source.")
            return

        # Set window properties
        cv2.namedWindow("Emotion Recognition", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Emotion Recognition", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)

        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream.")
                break

            # Detect emotion
            emotion = detect_emotions(frame, emotion_model, emotion_input, emotion_output)
            emotion_buffer.append(emotion)

            # Calculate dominant emotion every 10 seconds
            if time.time() - start_time > 10:
                dominant_emotion = max(set(emotion_buffer), key=emotion_buffer.count)
                play_music(dominant_emotion, music_dir)
                start_time = time.time()  # Reset the timer

            # Display result
            cv2.imshow("Emotion Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    except Exception as e:
        logging.error(f"Exception occurred: {e}")
        logging.error(traceback.format_exc())
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', default="0", type=str, help="Video file path or webcam ID")
    parser.add_argument('--device', default="AUTO", type=str, help="Device to start inference on")
    parser.add_argument('--emotion_model_name', type=str, default="emotions-recognition-retail-0003", help="Emotion recognition model to be used")
    parser.add_argument('--model_precision', type=str, default="FP32", choices=["FP16-INT8", "FP16", "FP32"], help="Model precision")
    parser.add_argument('--music_dir', type=str, required=True, help="Directory containing music files organized by emotion")

    args = parser.parse_args()
    run_demo(args.stream, args.emotion_model_name, args.model_precision, args.device, args.music_dir)