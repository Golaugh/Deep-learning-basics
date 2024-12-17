# Deep Learning Experience

This repository contains various deep learning implementations using TensorFlow, focusing on different neural network architectures and applications.

## Project Structure

### Basic Neural Networks
- Basic TensorFlow operations and version check (`0.0_test.py`)
- COVID-19 data visualization (`0.1_data_plot.py`)
- Iris dataset classification with different approaches:
  - Gradient-based implementation (`1.1_handle_iris_gradient.py`)
  - Simple neural network (`1.2_simply_handle_iris.py`) 
  - Custom model implementation (`1.3_custom_handle_iris.py`)

### Advanced Neural Networks
- Dot dataset classification with regularization (`2.1_handle_dot_regularization.py`)
- Fashion MNIST implementations:
  - Basic implementation (`3.1_myself_handle_fashion.py`)
  - Data augmentation version (`3.2_data_enhancement_handle_fashion.py`)

### MNIST Dataset
- Dataset preprocessing (`4.1_programme_handle_dataset_mnist.py`)
- Model saving and loading (`4.2_save_load_weight_display_mnist.py`)
- Image prediction (`4.3_load_change_predict_mnist.py`)

### Convolutional Neural Networks
- CIFAR-10 implementations with various architectures:
  - Dataset exploration (`5.1_cifar_10_dataset.py`)
  - Basic CNN (`5.2_convolutional_features_exeraction.py`)
  - LeNet-5 (`5.3_LeNet5_handle_cifar10.py`)
  - AlexNet (`5.4_AlexNet_handle_cifar10.py`)
  - VGGNet (`5.5_VGGNet_handle_cifar10.py`)
  - Inception (`5.6_Inception10_handle_cifar10.py`)
  - ResNet (`5.7_Resnet_handle_cifar10.py`)

### Recurrent Neural Networks
- Character sequence prediction:
  - One-hot encoding (1-to-1) (`6.1_Onehot_series_1pre1.py`)
  - One-hot encoding (4-to-1) (`6.2_Onehot_series_4pre1.py`)
  - Embedding layer (1-to-1) (`6.3_Embedding_series_1pre1.py`)
  - Embedding layer (4-to-1) (`6.4_Embedding_series_4pre1.py`)

### Stock Price Prediction
- Data collection (`7.0_collect_data.py`)
- Different RNN architectures:
  - Simple RNN (`7.1_Rnn_stock.py`)
  - LSTM (`7.2_LSTM_stock.py`)
  - GRU (`7.3_GRU_stock.py`)
- Final model implementation (`8.0_final_model_predict.py`)

## Requirements
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PIL (Python Imaging Library)

## Usage
Each Python file can be run independently. Make sure to have the required datasets in the appropriate directories before running the scripts.

## Model Weights
Model weights are saved in checkpoint directories and can be loaded for inference. Weight information is also logged in `weights.txt`.
