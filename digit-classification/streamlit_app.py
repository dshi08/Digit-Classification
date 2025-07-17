import streamlit as st
import torch
from PIL import Image
import numpy as np
from pytorch_model import PyTorchMLP
from MLP import MLP
from Value import Value
from vanilla_model import *
import pickle
import os

import os

# Use this path logic that works both locally and in deployment
def get_model_path(filename):
    """Search for models in these locations in order:"""
    search_paths = [
        os.path.join(os.path.dirname(__file__), "models", filename),  # Local dev
        os.path.join("/mount/src/digit-classification", "models", filename),  # Streamlit
        os.path.join("models", filename)  # Last resort
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            return path
    
    raise FileNotFoundError(f"Could not find {filename} in any search location")

# Usage:
TORCH_MODEL_PATH = get_model_path("pytorch_model.pth")
MICROGRAD_MODEL_PATH = get_model_path("micrograd_model.pkl")

# loading PyTorch model
pytorch_model = PyTorchMLP(784, [64, 32, 10])
pytorch_model.load_state_dict(torch.load(TORCH_MODEL_PATH, map_location=torch.device('cpu')))
pytorch_model.eval()

# loading vanilla model
vanilla_model = MLP(196, [64, 32, 10])
with open(MICROGRAD_MODEL_PATH, 'rb') as f:
    saved_params = pickle.load(f)
for p, val in zip(vanilla_model.parameters(), saved_params):
    p.data = val

def preprocess_pytorch(image):
    image = image.convert('L')
    image = image.resize((28,28))
    img_array = np.array(image) / 255.0
    flat_tensor = torch.tensor(img_array, dtype=torch.float32).view(1, -1)
    return flat_tensor

def preprocess_vanilla(image):
    image = image.convert('L')
    image = image.resize((14,14))
    img_array = np.array(image) / 255.0
    flat_list = img_array.flatten().tolist()
    return flat_list

st.title('MNIST Digit Classifier Demonstration')

model_choice = st.radio("Choose Model", ["PyTorch (97.48% acc)", "Vanilla (71.0% acc)"])
uploaded_file = st.file_uploader('upload a digit image', type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Images', use_column_width=True)

    if model_choice == 'PyTorch (97.48% acc)':
        input_tensor = preprocess_pytorch(image)
        with torch.no_grad():
            out = pytorch_model(input_tensor)
            pred = out.argmax(dim=1).item()
        st.write(f'Predicted digit (PyTorch model): **{pred}**')
    else:
        input_vals = preprocess_vanilla(image)
        x_vals = [Value(v) for v in input_vals]
        y_pred = vanilla_model(x_vals)
        pred = max(range(len(y_pred)), key=lambda i: y_pred[i].data)
        st.write(f'Predicted digit (Vanilla model): **{pred}**')