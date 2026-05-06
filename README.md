# 🧠 Brain Tumor Detection using Deep Learning (VGG16 + Flask)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Flask](https://img.shields.io/badge/Flask-Web%20App-green?style=for-the-badge&logo=flask)
![Deep Learning](https://img.shields.io/badge/Model-VGG16-red?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

---

## 📌 Overview

A **web-based Brain Tumor Detection system** powered by a fine-tuned **VGG16 convolutional neural network**. Users can upload MRI scan images and get instant AI-powered classification into one of four categories — with confidence scores displayed in real time.

> ⚠️ **Disclaimer:** This tool is intended for academic and research purposes only. It is not a substitute for professional medical diagnosis.

---

## 🎯 Features

- 🔍 **Multi-class tumor classification** — Glioma, Meningioma, Pituitary, No Tumor
- ⚡ **Real-time prediction** via a Flask web interface
- 🧠 **Transfer Learning** using pre-trained VGG16 (ImageNet weights)
- 📊 **Confidence score** displayed for each prediction
- 🖼️ **MRI image preview** after upload
- 💻 Clean, responsive UI

---

## 🗂️ Project Structure

```
brain-tumor-detection/
│
├── static/
│   ├── btd.css              # Stylesheet for UI
│   └── uploads/             # Stores uploaded MRI images (auto-created)
│
├── templates/
│   └── index.html           # Frontend HTML template (Jinja2)
│
├── app.py                   # Flask application + model inference logic
├── model.h5                 # Pre-trained VGG16 model weights
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## 🧠 Model Architecture

| Layer | Details |
|-------|---------|
| Base Model | VGG16 (pre-trained on ImageNet) |
| Input Shape | 128 × 128 × 3 |
| Fine-tuning | Last 3 conv layers unfrozen |
| Flatten | — |
| Dropout | 0.3 |
| Dense | 128 units, ReLU |
| Dropout | 0.2 |
| Output | 4 units, Softmax |

**Classes:** `glioma` · `meningioma` · `pituitary` · `no_tumor`

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-detection.git
cd brain-tumor-detection
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python app.py
```

### 4. Open in Browser

```
http://127.0.0.1:5000
```

Upload an MRI scan image and click **Analyze** to get a prediction!

---

## 📦 Requirements

```
flask
tensorflow
opencv-python
numpy
```

---

## 🖼️ Sample MRI Scans

The project was tested on MRI brain scans including cases of:
- Glioma tumors
- Meningioma tumors
- Pituitary tumors
- Normal (no tumor) brain scans

---

## 🔬 How It Works

1. User uploads an MRI image via the web interface
2. Image is resized to **128×128** pixels and normalized
3. The image is passed through the **fine-tuned VGG16 model**
4. Model outputs class probabilities via **Softmax**
5. The highest probability class is shown along with a **confidence score**

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python | Core language |
| TensorFlow / Keras | Model building & inference |
| VGG16 | Pre-trained CNN (Transfer Learning) |
| Flask | Web framework |
| HTML / CSS | Frontend UI |
| NumPy | Array processing |
| OpenCV | Image handling |

---

## 📈 Future Improvements

- [ ] Add Grad-CAM heatmap visualization for explainability
- [ ] Deploy on cloud (Heroku / AWS / Render)
- [ ] Add patient history and report generation (PDF)
- [ ] Improve UI with more medical-grade design
- [ ] Increase model accuracy with larger dataset

---

## 👨‍💻 Author

**[Your Name]**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- LinkedIn: [Your LinkedIn](https://linkedin.com/in/YOUR_PROFILE)
- Email: your.email@example.com

---

## 📄 License

This project is licensed under the **MIT License** — feel free to use it for learning and research.

---

## ⭐ Show Your Support

If you found this project helpful, please consider giving it a **⭐ star** on GitHub — it helps others discover it!
