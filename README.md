# Human Image Segmentation using PyTorch

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

A complete **end-to-end deep learning project** for **semantic segmentation of humans** from images using **U-Net** architecture with a pre-trained **ResNet34** backbone.

Built with PyTorch and `segmentation-models-pytorch` for fast and accurate human mask prediction.

---

## ✨ Features

- Custom PyTorch Dataset with Albumentations augmentation
- U-Net architecture with ImageNet-pretrained ResNet34 encoder
- Combined **BCE + Dice Loss** (industry standard for segmentation)
- Learning rate scheduling with `ReduceLROnPlateau`
- Best model checkpointing
- Clean training & validation loops
- Ready-to-use inference pipeline
- Fully reproducible on Google Colab

---

## 🛠️ Tech Stack

- **Framework**: PyTorch
- **Model**: `segmentation_models_pytorch` (U-Net + ResNet34)
- **Augmentation**: Albumentations
- **Visualization**: Matplotlib
- **Others**: OpenCV, Pandas, scikit-learn

---

## 📊 Dataset

- **Human Segmentation Dataset** by [VikramShenoy97](https://github.com/VikramShenoy97/Human-Segmentation-Dataset)
- ~600 high-quality images with corresponding binary masks
- Used `train.csv` for easy data handling
- Images resized to **320×320**

---

## 🚀 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/human-segmentation-pytorch.git
cd human-segmentation-pytorch
2. Install Dependencies
Bashpip install -r requirements.txt
requirements.txt (create this file):
txttorch
torchvision
torchaudio
segmentation-models-pytorch
albumentations
opencv-contrib-python
pandas
scikit-learn
tqdm
matplotlib
3. Run on Google Colab (Recommended)
Open the notebook Deep_Learning_with_PyTorch_ImageSegmentation.ipynb in Google Colab and run all cells.

📁 Project Structure
texthuman-segmentation-pytorch/
├── Deep_Learning_with_PyTorch_ImageSegmentation.ipynb   # Main notebook
├── best_model.pth                                       # Best trained model (after training)
├── requirements.txt
├── README.md
└── Human-Segmentation-Dataset-master/                   # Cloned dataset

🎯 Results
Training Summary (Example):

Epochs: 25
Best Validation Loss: ~0.18
Model: U-Net (ResNet34 backbone)

The model successfully learns to separate humans from complex backgrounds.
Example Output:
(Add your inference results here after running Task 9)

👨‍💻 Author
Raj Mishra

GitHub: @yourusername
LinkedIn: Raj Mishra(update link)
Twitter: @yourhandle(optional)


🙏 Acknowledgments

Original Dataset: VikramShenoy97/Human-Segmentation-Dataset
Model Library: segmentation-models-pytorch
Guided Project by Parth Dhameliya (inspiration)


📜 License
This project is licensed under the MIT License — feel free to use it for learning and projects.

⭐ If you found this project helpful, please give it a star on GitHub!