# ML-based Channel Estimation for MIMO-OFDM Systems

This repository contains MATLAB code for the paper:

> **Machine Learning-Based 5G-and-Beyond Channel Estimation for MIMO-OFDM Communication Systems**  
> *Ha An Le, Trinh Van Chien, Tien Hoa Nguyen, Hyunseung Choo, and Van Duc Nguyen*  
> Published in *Sensors*, 2021  
> [Read the paper](https://www.mdpi.com/1424-8220/21/14/4861)

---

## 📝 Overview

This project implements machine learning-based channel estimation techniques (DNN, CNN, LSTM) for MIMO-OFDM systems in the context of 5G and beyond wireless communication.

---

## 📁 File Descriptions

- `trainingdata_gen.m`  
  Generates training and testing datasets simulating MIMO-OFDM channel conditions.

- `DNN_training.m`, `CNN_training.m`, `LSTM_training.m`  
  Train deep learning models (DNN, CNN, LSTM) for channel estimation.

- `OFDM_ChannelEstimation_Inference.m`  
  Runs the trained models to perform channel estimation on test data.

---

## 🚀 Getting Started

### Requirements

- MATLAB R2021a or later
- Deep Learning Toolbox
- Signal Processing Toolbox

### How to Use

1. **Generate Dataset**
   ```matlab
   trainingdata_gen
   ```

2. **Train a Model**
   Choose one model to train:
   ```matlab
   DNN_training    % For Deep Neural Network
   CNN_training    % For Convolutional Neural Network
   LSTM_training   % For Long Short-Term Memory Network
   ```

3. **Run Inference**
   ```matlab
   OFDM_ChannelEstimation_Inference
   ```

---

## 📊 Results

The models are evaluated based on normalized mean square error (NMSE) performance. Example comparison with LS/MMSE baselines is provided in the paper.

---

## 📖 Citation

If you use this code in your research, please cite the following paper:

```bibtex
@Article{s21144861,
  AUTHOR = {Le, Ha An and Van Chien, Trinh and Nguyen, Tien Hoa and Choo, Hyunseung and Nguyen, Van Duc},
  TITLE = {Machine Learning-Based 5G-and-Beyond Channel Estimation for MIMO-OFDM Communication Systems},
  JOURNAL = {Sensors},
  VOLUME = {21},
  YEAR = {2021},
  NUMBER = {14},
  ARTICLE-NUMBER = {4861},
  URL = {https://www.mdpi.com/1424-8220/21/14/4861},
  PubMedID = {34300599},
  ISSN = {1424-8220},
  DOI = {10.3390/s21144861}
}
```

---

## 📬 Contact

For questions or collaborations, feel free to contact **Ha An Le** at `lehaan@snu.ac.kr`.

---

## 📄 License

This project is provided for academic and research purposes. Please check with the authors before using it for commercial applications.
