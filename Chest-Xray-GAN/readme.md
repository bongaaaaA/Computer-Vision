# Chest X-ray Image Augmentation using GAN 🧠🔥

## 🎯 Project Goal

The main goal of this project is to **enhance medical image datasets** by generating **realistic and diverse brain images** using **Generative Adversarial Networks (GANs)**.

Medical datasets often suffer from:
- Limited number of samples
- Low diversity
- Class imbalance (Normal vs Tumor)

To address these challenges, we use GANs to perform **data mutation and augmentation**, focusing on generating variations in:
- **Texture**
- **Thickness**
- **Intensity distribution**
- **Structural patterns**

These variations help simulate real-world medical differences between patients.

---

## 🧠 Core Idea

Instead of simply duplicating images or applying basic augmentations (rotation, flip, etc.), this project uses **GAN-based generation** to create **new synthetic images** that:

- Preserve medical realism
- Introduce natural mutations in tissue appearance
- Improve model generalization

The generated images act as **new training samples**, not just modified copies.

---

## 🧬 Mutation Concept in This Project

GAN-generated images introduce realistic **mutations**, such as:

- Changes in **tumor thickness**
- Variations in **tissue density**
- Different **contrast and texture patterns**
- Slight structural differences in brain regions

These mutations help the classifier learn **robust features** instead of overfitting to a small dataset.

---

## 🏗️ Project Pipeline

1. **Load real medical images**
2. **Train GAN models** on each class
   - Normal images
   - Tumor images
3. **Generate synthetic images**
4. **Combine real + generated data**
5. **Train a classification model**
6. **Evaluate performance improvement**

---

## 🧠 GAN Architecture

The GAN consists of:
- **Generator**: Learns to create realistic chest images with controlled variation
- **Discriminator**: Distinguishes real images from generated ones

Through adversarial training, the Generator learns to:
- Mimic real image distribution
- Introduce meaningful diversity in texture and thickness

---

## 🖼️ Generated Image Samples

### 🔹 Normal chast Images (Generated)
Synthetic images representing healthy brain scans with natural variations.

> Example: 24 generated Normal images in a single grid.

--- **image generated sqample 1**
<img <img width="1536" height="1024" alt="generated_xray1" src="https://github.com/user-attachments/assets/6f1ab4f8-e8f1-45f9-a7dd-4ff7b1153c27" />

--- **image generated sqample 2**
<img width="1536" height="1024" alt="generated_xray2 png" src="https://github.com/user-attachments/assets/cd94c98e-a7c2-41f2-bf5a-c734729605c5" />


## 📈 Why GAN-based Augmentation?

Traditional augmentation:
- Rotation
- Flip
- Zoom

GAN-based augmentation:
- Generates **new samples**
- Simulates unseen patient cases
- Improves robustness and generalization

This approach is especially effective in **medical imaging**, where collecting data is expensive and limited.

---

## 🧪 Use Cases

- chest x-ray classification
- Medical image research
- Data augmentation for deep learning
- Training robust CNN models on small datasets

---

## ⚠️ Disclaimer

This project is intended for **research and educational purposes only**.
The generated images are **not real medical data** and should not be used for clinical diagnosis.

---

## 🚀 Technologies Used

- Python
- PyTorch
- GANs (Generator & Discriminator)
- OpenCV
- NumPy
- Matplotlib

---

## 👨‍💻 Author

Developed by **Abdelrahman (bongaaaaA)**  
GitHub: https://github.com/bongaaaaA

---

## ⭐ Final Note

This project demonstrates how **GAN-based mutation and augmentation** can significantly improve deep learning performance in medical imaging tasks by enriching datasets with realistic synthetic samples.
