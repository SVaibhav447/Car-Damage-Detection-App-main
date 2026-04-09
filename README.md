# 🚗 Automated Vehicle Damage Classification

A deep learning-based system that detects and classifies vehicle damage from images using a fine-tuned ResNet50 model with transfer learning. Built to handle real-world variability and enable automated damage assessment workflows.

---

## 📌 Features

* Damage classification using ResNet50 (transfer learning)
* Supports multiple damage categories
* Robust to varying lighting and real-world conditions
* Data augmentation for improved generalization
* End-to-end pipeline from input image to prediction
* Scalable design for integration into larger systems

---

## 🧠 Model Details

* **Architecture:** ResNet50
* **Approach:** Transfer Learning
* **Framework:** PyTorch
* **Techniques Used:**

  * Data Augmentation
  * Hyperparameter Tuning
  * Fine-tuning pretrained weights

---

## 🗂️ Project Structure

```
├── data/                # Dataset (not included)
├── models/              # Saved model weights
├── src/                 # Training and inference scripts
├── app/                 # Application / interface code
├── utils/               # Helper functions
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/vehicle-damage-classification.git
cd vehicle-damage-classification
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Model

```bash
python src/train.py
```

### 2. Run Inference

```bash
python src/predict.py --image path/to/image.jpg
```

---

## 📊 Results

* Achieved strong classification performance on validation dataset
* Improved robustness through augmentation and tuning
* Suitable for real-world deployment scenarios

---

## 🔮 Future Improvements

* Deploy as a web application (React + Node.js)
* Integrate real-time image capture
* Expand dataset for better generalization
* Add severity estimation for damage

---

## 🛠️ Tech Stack

* Python
* PyTorch
* OpenCV
* NumPy

---

## 📄 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Vaibhav Sharma**
B.Tech CSE | SRMIST

---

## ⭐ Acknowledgements

* PyTorch team for deep learning framework
* Open-source datasets and research community
