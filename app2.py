import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import gradio as gr
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import snntorch as snn
from timm.models.efficientnet import efficientnet_b3
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for Matplotlib

# Define the model architecture
class COVIDSNNFinal(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = efficientnet_b3(pretrained=True)
        self.backbone.classifier = nn.Identity()

        self.temporal = snn.Leaky(beta=0.95, init_hidden=False)
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x, time_steps=10):
        x = self.backbone(x)
        mem = self.temporal.init_leaky()
        outputs = []
        for _ in range(time_steps):
            spk, mem = self.temporal(x, mem)
            outputs.append(self.fc(spk))
        return torch.stack(outputs).mean(0)

# GradCAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        # Register forward hook
        fwd_hook = target_layer.register_forward_hook(
            self._forward_hook
        )
        # Register backward hook
        bwd_hook = target_layer.register_full_backward_hook(
            self._backward_hook
        )
        self.hook_handles.extend([fwd_hook, bwd_hook])

    def _forward_hook(self, module, input, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def get_cam(self, input_tensor):
        self.model.zero_grad()
        output = self.model(input_tensor)
        class_idx = output.argmax().item()

        # Backpropagate for specific class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute CAM
        weights = torch.mean(self.gradients, dim=[2, 3])
        cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)  # Add epsilon to avoid division by zero
        return cam.squeeze().cpu().numpy()

# Function to preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Convert to PIL if it's a numpy array
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    
    # Ensure image is RGB
    image = image.convert('RGB')
    
    # Apply transformation
    img_tensor = transform(image).unsqueeze(0)
    return img_tensor, image

# Function to generate Grad-CAM visualization
def generate_gradcam(model, input_tensor, original_image):
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)
    input_tensor.requires_grad_(True)
    
    # Find the last convolutional layer
    target_layer = None
    for name, module in model.backbone.named_modules():
        if isinstance(module, nn.Conv2d):
            target_layer = module
    
    if not target_layer:
        raise ValueError("No convolutional layer found in backbone")
    
    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)
    
    # Generate heatmap
    heatmap = gradcam.get_cam(input_tensor)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Convert original image to numpy array and resize
    original_np = np.array(original_image.resize((224, 224)))
    
    # Superimpose heatmap on image
    superimposed_img = cv2.addWeighted(original_np, 0.6, heatmap_colored, 0.4, 0)
    
    return original_np, heatmap_colored, superimposed_img

# Main prediction function for Gradio
def predict(image):
    # Class names
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    
    # Check if image is provided
    if image is None:
        return None, None, None, "Please upload an image.", 0, 0, 0, 0
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = COVIDSNNFinal(num_classes=4).to(device)
    
    # Load model weights - path needs to be adjusted based on where you store the model
    try:
        model.load_state_dict(torch.load('covid_snn_model.pth', map_location=device))
    except:
        return None, None, None, "Model file not found. Please make sure 'covid_snn_model.pth' is in the current directory.", 0, 0, 0, 0
    
    model.eval()
    
    # Preprocess image
    try:
        input_tensor, original_image = preprocess_image(image)
        input_tensor = input_tensor.to(device)
    except Exception as e:
        return None, None, None, f"Error processing image: {str(e)}", 0, 0, 0, 0
    
    # Get prediction
    with torch.no_grad():
        output = model(input_tensor)
    
    pred_prob = torch.nn.functional.softmax(output, dim=1)[0] * 100
    
    # Ensure confidence levels never show 100%
    rounded_pred = torch.round(pred_prob * 100) / 100  # Round to two decimals
    mask = rounded_pred >= 100.0
    pred_prob = torch.where(mask, torch.tensor(99.8, device=pred_prob.device), pred_prob)
    
    pred_class = torch.argmax(output).item()
    
    # Generate Grad-CAM visualization
    try:
        original_np, heatmap, superimposed_img = generate_gradcam(model, input_tensor, original_image)
    except Exception as e:
        return None, None, None, f"Error generating Grad-CAM: {str(e)}", 0, 0, 0, 0
    
    # Create diagnosis message
    diagnosis = f"Final Diagnosis: {class_names[pred_class]}"
    
    # Return results
    confidences = pred_prob.cpu().numpy().tolist()
    
    return original_np, heatmap, superimposed_img, diagnosis, confidences[0], confidences[1], confidences[2], confidences[3]

# Create Gradio interface
with gr.Blocks(title="COVID-19 X-ray Diagnosis with SNN and Grad-CAM") as app:
    gr.Markdown("# COVID-19 X-ray Diagnosis with Spiking Neural Network")
    gr.Markdown("Upload a chest X-ray image to get a diagnosis and visualization of the model's focus areas")
    
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input X-ray Image", type="pil")
            submit_btn = gr.Button("Analyze X-ray", variant="primary")
        
        with gr.Column():
            diagnosis_text = gr.Textbox(label="Diagnosis Result")
            
            with gr.Row():
                with gr.Column():
                    covid_conf = gr.Number(label="COVID Confidence (%)")
                with gr.Column():
                    opacity_conf = gr.Number(label="Lung Opacity Confidence (%)")
            
            with gr.Row():
                with gr.Column():
                    normal_conf = gr.Number(label="Normal Confidence (%)")
                with gr.Column():
                    pneumonia_conf = gr.Number(label="Viral Pneumonia Confidence (%)")
    
    with gr.Row():
        original_output = gr.Image(label="Original X-ray")
        heatmap_output = gr.Image(label="Activation Map (Grad-CAM)")
        superimposed_output = gr.Image(label="X-ray with Highlighted Areas")
    
    submit_btn.click(
        predict,
        inputs=[input_image],
        outputs=[original_output, heatmap_output, superimposed_output, diagnosis_text, 
                covid_conf, opacity_conf, normal_conf, pneumonia_conf]
    )
    
    gr.Markdown("""
    ## About this Application
    
    This application uses a Spiking Neural Network (SNN) with an EfficientNet-B3 backbone to analyze chest X-rays 
    and classify them into four categories:
    
    - COVID-19
    - Lung Opacity (non-COVID lung abnormality)
    - Normal
    - Viral Pneumonia
    
    The Grad-CAM visualization shows which areas of the X-ray the model is focusing on to make its diagnosis.
    """)

# Launch the app
if __name__ == "__main__":
    app.launch()