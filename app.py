import os
import numpy as np
import cv2
import matplotlib.cm as cm
import gradio as gr
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# ——— Helper to find the real last Conv2D layer, even if nested ———
def get_last_conv_layer(model: keras.Model):
    # Check each layer in reverse; if it's a Conv2D, return it.
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D):
            return layer
        # If it's itself a Model or Sequential, dive in
        if isinstance(layer, keras.Model):
            found = get_last_conv_layer(layer)
            if found is not None:
                return found
    return None

# ——— Fallback model for demos only ———
def create_fallback_model():
    inputs = keras.Input(shape=(70, 70, 3))
    x = keras.layers.Conv2D(128, 3, activation='relu')(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(16, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(4)(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    model._is_fallback = True
    return model

# ——— Load your real model (or fallback) ———
def load_model(path='covid_model.h5'):
    if not os.path.exists(path):
        print(f"Model file not found at {path}, using fallback.")
        return create_fallback_model()

    try:
        # Load without compile so we bypass reduction='auto' errors
        model = keras.models.load_model(path, compile=False)
        model.compile(
            optimizer='adam',
            loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy']
        )
        model._is_fallback = False
        print("✔️ Model loaded and compiled successfully")
        return model

    except Exception as e:
        print("❌ Error loading model:", e)
        print("Falling back to demo model.")
        return create_fallback_model()

# ——— Preprocess into a batch of size 1 ———
def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    # ensure 3 channels
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)
    image = cv2.resize(image, (70, 70))
    image = image.astype('float32') / 255.0
    return np.expand_dims(image, axis=0)

# ——— Predict with dynamic softmax check ———
def predict_class(model, image):
    batch = preprocess_image(image)
    preds = model.predict(batch)
    # if outputs already sum to 1, assume they’re probs
    if np.allclose(preds.sum(axis=1), 1.0, atol=1e-3):
        probs = preds
    else:
        probs = tf.nn.softmax(preds, axis=1).numpy()
    idx = int(np.argmax(probs[0]))
    conf = float(probs[0, idx])
    names = {0: "Normal", 1: "Covid Positive", 2: "Lung Opacity", 3: "Viral Pneumonia"}
    return names.get(idx, str(idx)), conf, idx

# ——— Build Grad-CAM by wiring the real conv layer’s output ———
def make_gradcam_heatmap(img_array, model, conv_layer, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs],
            [conv_layer.output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            loss = preds[:, pred_index]
        grads = tape.gradient(loss, conv_out)
        pooled = tf.reduce_mean(grads, axis=(0,1,2))
        conv_out = conv_out[0]
        heatmap = conv_out @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        print("Grad-CAM error:", e)
        # placeholder radial mask
        h, w = img_array.shape[1], img_array.shape[2]
        Y, X = np.ogrid[:h, :w]
        dist = np.sqrt((X - w/2)**2 + (Y - h/2)**2)
        mask = 1 - (dist / dist.max())
        return np.clip(mask, 0, 1)

# ——— Overlay it on the original ———
def create_heatmap_overlay(image, heatmap, alpha=0.4):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # jet map
    heat = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")(np.arange(256))[:, :3]
    colored = (jet[heat] * 255).astype(np.uint8)
    colored = cv2.resize(colored, (image.shape[1], image.shape[0]))
    return cv2.addWeighted(colored, alpha, image, 1 - alpha, 0)

# ——— Two simple filters ———
def ben_graham_enhancement(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    try:
        hsv = cv2.cvtColor(cv2.resize(bgr, (512,512)), cv2.COLOR_BGR2HSV)
        enhanced = cv2.addWeighted(hsv, 4, cv2.GaussianBlur(hsv,(0,0),51.2), -4, 128)
        return cv2.cvtColor(enhanced, cv2.COLOR_HSV2RGB)
    except:
        return cv2.convertScaleAbs(image, alpha=1.5, beta=0)

def extract_b_channel(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    if image.ndim == 2:
        image = np.stack((image,)*3, axis=-1)
    b = image[:,:,0]
    return np.stack((b,)*3, axis=-1)

# ——— The main Gradio callback ———
def analyze_xray(image):
    global model, last_conv_layer
    if model is None:
        return ("Model not available", image, image, image, image, "N/A", 0.0)

    # Predict
    cls, conf, idx = predict_class(model, image)

    # Grad-CAM
    batch = preprocess_image(image)
    heat = make_gradcam_heatmap(batch, model, last_conv_layer, pred_index=idx)
    overlay = create_heatmap_overlay(image, heat)

    # Filters
    ben = ben_graham_enhancement(image)
    bch = extract_b_channel(image)

    arr = np.array(image)
    sheet = (
        f"Shape: {arr.shape}\n"
        f"Min/Max: {arr.min()}/{arr.max()}\n"
        f"Mean/Std: {arr.mean():.2f}/{arr.std():.2f}\n\n"
        f"Class: {cls}\nConfidence: {conf:.4f}"
    )
    if getattr(model, "_is_fallback", False):
        sheet += "\n\n⚠️ Demo mode (fallback weights)"

    return sheet, overlay, ben, bch, arr, cls, conf

# ——— Load model once at startup ———
model = load_model()
last_conv_layer = get_last_conv_layer(model)
if last_conv_layer is None:
    raise ValueError("No Conv2D layer found in your model!")
print("Using conv layer:", last_conv_layer.name)

# ——— Build Gradio UI ——
with gr.Blocks(title="COVID-19 X-ray Analysis") as demo:
    gr.Markdown("# COVID-19 X-ray Classifier")
    gr.Markdown("Upload a chest X-ray to see the prediction and Grad-CAM")

    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(label="X-ray", type="numpy")
            btn = gr.Button("Analyze")
            out_cls = gr.Textbox(label="Class")
            out_conf = gr.Number(label="Confidence")
        with gr.Column(scale=2):
            out_sheet = gr.Textbox(label="Details", lines=8)

    gr.Markdown("## Visualizations")
    with gr.Row():
        orig = gr.Image(label="Original")
        cam = gr.Image(label="Grad-CAM")
    with gr.Row():
        ben = gr.Image(label="Ben Graham")
        bch = gr.Image(label="B-Channel")

    btn.click(
        analyze_xray,
        inputs=[inp],
        outputs=[out_sheet, cam, ben, bch, orig, out_cls, out_conf]
    )

if __name__ == "__main__":
    demo.launch()
