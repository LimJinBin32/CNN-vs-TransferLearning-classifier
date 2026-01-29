# 🧠 CNN vs Transfer Learning: Vehicle Image Classifier
This project compares 2 deep learning approaches - a custom Convolutional Neural Network (CNN) and Transfer learning - to classify images into three categories: **airplane**, **automobile**, and **truck**

---

## 🎯 Objective

To design, build, and evaluate deep learning models capable of performing accurate multi-class image classification using a real-world dataset. This includes:
- Developing two deep learning models (CNN and Transfer Learning)
- Applying regularization and tuning strategies
- Evaluating using appropriate performance metrics
- Comparing both models in terms of performance, efficiency, and learning suitability

---

## 📁 Dataset

- **Name**: `dataset_transport`
- **Classes**: 3 (airplane, automobile, truck)
- **Training Set**: 6,000 images  
- **Test Set**: 1,500 images  
- **Image Resolution**: `32x32x3` (low resolution)  
- **Source**: Provided by lecturer via Brightspace

> ⚠️ The low-resolution nature of the dataset (32x32 pixels) posed a challenge for feature extraction.

![Sample dataset images](https://github.com/LimJinBin32/CNN-vs-TransferLearning-classifier/blob/36fe7af0976e0c44c1917150c40fc43e2dae5263/Image_Batch.png?raw=true)

---

## 🗂️ Project Structure
```
T2_221128Z_EGT214_PROJECT.ipynb
├── Data Preprocessing
├── CNN Model
│ └── Architecture, Training, Regularization
├── Summary of CNN Model
├── Transfer Learning
│ └── MobileNetV2, DenseNet121, InceptionV3 Trials
│ └── Final InceptionV3 Model
│ └── Testing TL Model
├── Summary of Transfer Learning
├── Final Comparison of CNN vs TL
```

---

## ⚙️ Tools & Libraries

- Python 3
- TensorFlow / Keras
- Matplotlib, Seaborn (for visualization)
- Scikit-learn (for evaluation metrics)

---

## 📊 Results Summary

| Model              | Accuracy | AUC Score | Notable Observations                          |
|-------------------|----------|-----------|-----------------------------------------------|
| Custom CNN        | ~90%     | 0.92      | Good performance, regularized, fast training  |
| MobileNetV2       | Low      | Unstable  | Low accuracy, poor convergence                |
| DenseNet121       | ~90%     | 0.91      | Stable, plateaued early                       |
| InceptionV3       | Highest  | 0.94+     | Best performance, improved with fine-tuning   |

✅ **Final model selected**: Fine-tuned **InceptionV3**

---

## 🧠 Key Learnings

- Designed and tuned a CNN from scratch  
- Gained hands-on experience with Transfer Learning workflows  
- Applied techniques like:
  - Batch normalization
  - Dropout & L2 regularization
  - Data augmentation
  - Fine-tuning pretrained models
- Evaluated models using multiple metrics (Accuracy, AUC, F1)
- Compared architectures to understand trade-offs

---

## 📝 File

- `T2_221128Z_EGT214_PROJECT.ipynb` – Full notebook with model building, results, and summaries

---

## 👤 Author

**Lim Jin Bin**  
AI & Data Engineering – Nanyang Polytechnic  
Module: EGT214 – Applied Deep Learning  
Admin No: 221128Z

---

## 📄 License

This project is for educational and academic use only.
