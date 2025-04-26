# GANHide: A GAN-Based Steganography System

![Project Banner](https://via.placeholder.com/800x200.png?text=GANHide+Steganography+System)

**GANHide** is a machine learning project that leverages Generative Adversarial Networks (GANs) to perform steganography—hiding messages within images in a way that is imperceptible to human observers while ensuring reliable message recovery. This project uses a conditional GAN (cGAN) to embed 8-bit binary messages into 8x8 grayscale images from the UCI Optical Recognition of Handwritten Digits dataset, with a separate CNN-based Extractor to recover the messages.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Mathematical Background](#mathematical-background)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Steganography is the practice of concealing messages within non-secret data to avoid detection. Traditional steganographic methods, such as least significant bit (LSB) embedding, can be vulnerable to statistical detection. **GANHide** introduces an innovative approach by using a cGAN to embed messages into images, ensuring both imperceptibility (via the Discriminator) and recoverability (via the Extractor). The system is trained on the UCI Optical Recognition of Handwritten Digits dataset, achieving high message recovery accuracy while maintaining visual similarity between original and stego-images.

## Features
- **Message Embedding**: Embeds 8-bit binary messages into 8x8 grayscale images using a Generator.
- **Imperceptibility**: Uses a Discriminator to ensure stego-images are indistinguishable from originals.
- **Message Recovery**: Employs a CNN-based Extractor to recover hidden messages with high accuracy.
- **Evaluation Metrics**: Includes bit-wise accuracy for message recovery and visual/histogram comparisons for imperceptibility.
- **Custom Training**: Utilizes `tf.GradientTape` for flexible training of the Generator.

## Installation

### Prerequisites
- Python 3.11+
- TensorFlow 2.x
- NumPy
- Pandas
- Matplotlib
- Requests

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ganhide.git
   cd ganhide
   ```
2. Install the required packages:
   ```bash
   pip install tensorflow numpy pandas matplotlib requests
   ```
3. Ensure you have an internet connection to download the UCI dataset during execution.

## Usage

1. **Prepare the Code**: The main script is `ganhide.py`. Ensure it’s in your working directory.
2. **Run the Script**:
   ```bash
   python ganhide.py
   ```
   - The script downloads the UCI dataset, trains the GAN for 20 epochs, and evaluates the system.
   - It generates plots (`training_metrics.png`, `image_comparison.png`, `histogram_comparison.png`) and prints test results.
3. **Expected Output**:
   - Training logs showing Discriminator Loss/Accuracy, Generator Loss, Extractor Loss/Accuracy, and Validation Extractor Accuracy.
   - Visualizations comparing original and stego-images, histograms, and training metrics.
   - Test results showing original vs. extracted messages for 5 samples.

## Mathematical Background

### System Components
- **Generator (G)**: Embeds message \( m \) into image \( x \), producing stego-image \( \hat{x} = G(x, m) \).
- **Discriminator (D)**: Outputs probability \( D(x) \in [0,1] \), classifying images as real (1) or fake (0).
- **Extractor (E)**: Recovers message \( \hat{m} = E(\hat{x}) \) from stego-image.

### Loss Functions
- **Discriminator Loss**:
  \[
  \mathcal{L}_D = -\mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] - \mathbb{E}_{x \sim p_{\text{data}}, m \sim p_m} [\log (1 - D(G(x, m)))]
  \]
- **Generator Loss**:
  \[
  \mathcal{L}_G = -\mathbb{E}_{x \sim p_{\text{data}}, m \sim p_m} [\log D(G(x, m))] + \lambda \cdot \mathcal{L}_{\text{extraction}}
  \]
  where \( \mathcal{L}_{\text{extraction}} = \mathbb{E}_{x \sim p_{\text{data}}, m \sim p_m} [\text{BCE}(m, E(G(x, m)))] \), and \( \lambda = 1.0 \).
- **Extractor Loss**:
  \[
  \mathcal{L}_E = \mathbb{E}_{x \sim p_{\text{data}}, m \sim p_m} [\text{BCE}(m, E(G(x, m)))]
  \]

### Bit-Wise Accuracy
- Measures Extractor's accuracy in recovering message bits:
  \[
  \text{Accuracy} = \frac{1}{N \cdot 8} \sum_{i=1}^{N} \sum_{j=1}^{8} \mathbb{I}(m_{i,j} = \hat{m}_{i,j})
  \]

## Results

### Training Metrics (20 Epochs)
- **Discriminator Accuracy (D Acc)**: Stabilized at ~0.5001, indicating stego-images are indistinguishable from originals.
- **Extractor Accuracy (E Acc)**: Reached 97.07%, showing reliable message recovery.
- **Validation Extractor Accuracy (Val E Acc)**: Achieved 97.99%, confirming generalization to unseen data.

### Visualizations
- **Training Metrics Plot**: Shows trends in D Loss/Acc, G Loss, E Loss/Acc, and Val E Acc.
- **Original vs. Stego-Images**: Visual comparison demonstrates imperceptibility.
- **Histogram Comparison**: Highlights subtle statistical differences, supporting stealth.

### Test Results
The system perfectly recovered messages in test samples, aligning with the high E Acc and Val E Acc.

## Contributing

Contributions are welcome! To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/your-feature`).
3. Commit your changes (`git commit -m "Add your feature"`).
4. Push to the branch (`git push origin feature/your-feature`).
5. Open a Pull Request.

Please ensure your code follows PEP 8 style guidelines and includes relevant documentation.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
