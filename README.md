# COVID-19 X-ray Analysis

A Gradio web application for analyzing COVID-19 chest X-ray images using a Spiking Neural Network (SNN) based deep learning model.

## Features

- Upload and analyze chest X-ray images
- Multi-class classification (Normal, COVID-19, Lung Opacity, Viral Pneumonia)
- Visualization with Grad-CAM heatmaps highlighting important regions
- Additional image enhancement filters

## Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the application:

```bash
python app.py
```

3. Open the provided URL in your browser to use the application

## Technologies Used

- PyTorch and snntorch for the SNN model
- Gradio for the web interface
- EfficientNet-B3 as the backbone architecture
- Grad-CAM for model explainability
