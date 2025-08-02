# 🧠 Handwritten Digit Recognizer

 **MNIST handwritten digit recognizer** built **from scratch** using only **NumPy** for the neural network implementation with a live interactive UI powered by **Pygame**.

---

## ✨ Features

- 🧮 **Fully Custom Neural Network** using **only NumPy**
- 🖼️ **Interactive UI** to draw digits on a 28x28 grid using **Pygame**
- 🔄 **Live predictions** rendered directly on the interface
- 📉 **Training visualization** via `tqdm` and `matplotlib`
- 📦 Model is saved as a **lightweight JSON file**
- 🧪 Uses the official **MNIST dataset** (via Keras) for training and evaluation
- ✅ Accuracy **97%+**

---

## 🛠 Technologies & Skills Used

| Category         | Tools / Concepts                               |
|------------------|-------------------------------------------------|
| Core Logic       | `NumPy` neural network implementation (manual backprop, SGD) |
| Data Loading     | `Keras.datasets` for MNIST                     |
| User Interface   | `Pygame`                                       |
| Visualization    | `matplotlib` (for prediction previews) |
| Training Utility | `tqdm` for progress bars                       |
| Model Persistence| `JSON` for saving weights and biases           |

---

## 🧠 How It Works

- **Model**: A simple 3-layer neural network trained with **categorical cross-entropy loss** and **stochastic gradient descent**.
- **Architecture**:
  - Input: 784 (flattened 28x28 image)
  - Hidden Layer 1: 128 neurons (ReLU)
  - Hidden Layer 2: 64 neurons (ReLU)
  - Output: 10 neurons (Softmax)
- **Training**: Done in batches with shuffled data and real-time loss display using `tqdm`.
- **Prediction**: Draw a digit, and the model processes it in real time after pre-processing and centering the input.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/handwritten-digit-recognizer
cd handwritten-digit-recognizer
pip install -r requirements.txt
```

> ⚠️ `tensorflow` is required **only to access the MNIST dataset** via `keras.datasets`.

---

## ▶️ Usage

```bash
python main.py
```

- Press `P` or click **Predict** to get the model's prediction.
- Press `Q` or click **Clear** to reset the canvas.
- Digit predictions appear on the right along with a visual preview of what the model saw.

---

## 📈 Training From Scratch (Optional)

You can retrain the model and save it:

```python
from model import train_model
train_model("model.json")
```

---

## 📁 File Structure

```
├── main.py               # UI and event loop
├── model.py              # Neural network logic
├── activation.py         # ReLU, Softmax and derivatives
├── loss.py               # Loss functions
├── model.json            # Saved model weights
├── README.md
├── requirements.txt
```

---

## 🧠 Why This Project?

This project was built to deeply understand:
- Neural network fundamentals (forward pass, backpropagation)
- How digit classification works under the hood
- Manual training loop without black-box libraries
- Real-time UI integration with a trained model

---

