# VisionEarth – Deep Learning for Earth Observation

VisionEarth is a satellite image classification project that uses deep learning to analyze and categorize land use patterns from Sentinel-2 RGB satellite imagery. The model is trained on the EuroSAT dataset using TensorFlow and MobileNetV2 as the feature extractor.

An interactive Gradio interface is included for live predictions.

---

## 📂 Dataset

- **Name**: EuroSAT (RGB version)
- **Source**: [TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/eurosat)
- **Classes**: Residential, Industrial, River, Forest, Highway, etc.
- **Image Format**: Sentinel-2 RGB, 64x64 pixels

---

## 🧠 Model Overview

- **Architecture**: MobileNetV2 (pretrained on ImageNet)
- **Technique**: Transfer Learning
- **Input Size**: 128x128x3
- **Output**: Softmax layer for multi-class classification

```python
model = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(128, 128, 3))
```

A custom classifier head with GlobalAveragePooling and Dense layers is added for classification.

---

## 🛠️ Preprocessing

- Images resized to 128x128
- Normalized (pixel values scaled to [0, 1])
- One-hot encoded labels
- Data loading and batching using `tf.data` API

---

## 🚀 Training

- **Loss Function**: Categorical Crossentropy
- **Optimizer**: Adam
- **Batch Size**: 32
- **Early Stopping**: Enabled (patience = 3)

Training and validation datasets are split as 80%/20%.

---

## 🎯 Evaluation

Evaluation includes accuracy measurement on validation data using TensorFlow's model.evaluate and prediction visualization.

Confusion matrix and classification report can be added optionally.

---

## 🧪 Demo (Gradio Interface)

An interactive Gradio UI allows uploading satellite images and receiving predicted land use categories in real-time.

```python
def predict_image(img):
    img = tf.image.resize(img, [128, 128])
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.expand_dims(img, axis=0)
    preds = model.predict(img)[0]
    class_idx = tf.argmax(preds).numpy()
    confidence = tf.reduce_max(preds).numpy()
    return f"Prediction: {class_names[class_idx]} (Confidence: {confidence:.2f})"
```

---

## 📦 Dependencies

- TensorFlow
- TensorFlow Datasets
- Gradio
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install using:

```bash
pip install tensorflow tensorflow-datasets gradio numpy matplotlib seaborn scikit-learn
```





