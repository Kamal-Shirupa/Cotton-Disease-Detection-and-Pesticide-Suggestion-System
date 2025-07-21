# 🌱 AI Crop Disease Detection & Recommendation

This project leverages deep learning to detect crop diseases from images and generate targeted pesticide recommendations, empowering precision agriculture through automated image analysis and actionable treatment advice.

---

## 📦 Project Structure

```
ai-crop-disease-detector/
├── data/             # Datasets (train/test images, external CSVs)
├── example/          # Representative healthy and diseased leaf images
├── notebook/         # Jupyter/Colab notebook(s) for pipeline demonstration
├── pesticide_data/   # Pesticide recommendation datasets (CSVs)
├── requirements/     # Python dependency file(s)
├── results/          # Model metrics, plots, output examples
├── samples/          # Example input images and prediction outputs
├── src/              # Source code: preprocessing, training, inference, utils
├── .gitignore
├── LICENSE
├── README.md
```

---

## 🚀 Features

- Fast and accurate plant disease classification from images
- CSV-driven pesticide recommendation engine
- Modular, well-documented Python codebase
- End-to-end demo notebook with Google Colab support
- Sample inputs/outputs and example images for easy testing

---

## ⚙️ Installation

Install all project dependencies:

```
pip install -r requirements/requirements.txt
```

---

## 🚩 Quick Start

- **Try the Demo Notebook:**  
  Launch and run the main pipeline in your browser:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1J_LTsWD7rRbvfEGfPBAaRUbxGjeea0CO)

- **Test the Model:**  
  Use images from `example/` or `samples/` to try out predictions and recommendations via the notebook or scripts in `src/`.

---

## 📊 Results

- See the `results/` folder for training curves and test metrics.
- Review `samples/sample_output.json` for example predictions and recommendations.

---

## 💾 Data

- All required datasets—with descriptions—are in `data/` and `pesticide_data/`.
- See each folder’s README for structure and usage guidelines.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👤 Author

Made by [Kamal-Shirupa](https://github.com/Kamal-Shirupa).  
For questions or contributions, feel free to raise an issue or pull request.
