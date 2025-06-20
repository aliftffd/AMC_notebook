# Automatic Modulation Classification (AMC) using Deep Learning

## 🚀 Project Overview

This repository provides a comprehensive framework for exploring, visualizing, and building Automatic Modulation Classification (AMC) models using deep learning. Leveraging the RadioML2018.01A dataset, you can:

* Perform exploratory data analysis and feature extraction
* Visualize I/Q constellation diagrams
* Develop, train, and evaluate Convolutional Neural Network (CNN)–based architectures
* Experiment with enhanced and residual CNN variants for improved accuracy

Let's push the boundaries of edge‑optimized AMC together! 🎉

## 📁 Repository Structure

```
.
├── Experimental_notebook/            # Cutting‑edge experiments and prototypes
├── improvement_notebook/             # Notebooks for model refinements
├── __pycache__/                      # Python cache files
├── notebook_for_windows_with_dataset_on_windows.ipynb  # Windows dataset setup
├── CNN_class.py                      # Base CNN model class
├── DDRCNN_class.py                   # Deep Dense Residual CNN class
├── Improved_CNN2.ipynb               # Enhanced CNN architecture experiments
├── constellation.ipynb               # Constellation diagram plotting
├── dataset_explorer.ipynb            # Data loading & EDA for RadioML2018.01A
├── energy_exploration.ipynb          # Signal energy feature analysis
├── exploredataset.ipynb              # Initial dataset exploration
├── multiple_two_diffrentsource.py    # Handle multi‑source signal generation
├── test_model_function.py            # Unit tests for model components
├── .gitignore                        # Files and folders to ignore
└── README.md                         # (This file)
```

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/<your_username>/<your_repo>.git
   cd <your_repo>
   ```
2. **Create and activate a virtual environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```
3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

> Tip: After installing, run `pip freeze > requirements.txt` to capture your environment.

## ⚙️ Usage

1. **Data Exploration**: Open `dataset_explorer.ipynb` to inspect and preprocess the RadioML2018.01A dataset.
2. **Feature Analysis**: Use `energy_exploration.ipynb` to extract and analyze signal energy features.
3. **Visualization**: Launch `constellation.ipynb` to plot and study constellation diagrams.
4. **Model Development**:

   * **Base CNN**: Configure and run `CNN_class.py`.
   * **Improved CNN**: Experiment interactively with `Improved_CNN2.ipynb`.
   * **Residual CNN**: Build deeper architectures in `DDRCNN_class.py`.
5. **Testing**: Execute `test_model_function.py` to ensure model components work as expected.
6. **Advanced Experiments**: Explore notebooks under `Experimental_notebook/` and `improvement_notebook/` for the latest prototypes.

## 📊 Results & Benchmarks

* **Baseline CNN**: \~60.9% accuracy on RadioML2018.01A (see `energy_exploration.ipynb`).
* **Improved CNN2**: Ongoing work, targeting >70% accuracy.
* **DDRCNN**: Experimental deep residual CNN achieving promising initial results.

## 🤝 Contributing

Your ideas and contributions are highly encouraged! Whether it's bug fixes, new model architectures, or additional data visualizations, feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-cool-feature`)
3. Commit your changes (`git commit -m 'Add my cool feature'`)
4. Push to the branch (`git push origin feature/my-cool-feature`)
5. Open a Pull Request

Let’s advance AMC research side by side! 🚀


## 📚 References

* O’Shea, T. J., & West, N. (2016). Radio machine learning dataset generation with GNU Radio. GNU Radio Conference.
* Huynh-The, T., Pham, Q.-V., Nguyen, T.-V., Nguyen, T. T., Ruby, R., Zeng, M., & Kim, D.-S. (2021). Automatic modulation classification: A deep architecture survey. IEEE Access, 9, 142950–142973. [https://doi.org/10.1109/ACCESS.2021.3120419](https://doi.org/10.1109/ACCESS.2021.3120419)
* Kong, W., Jiao, X., Xu, Y., & Yang, Q. (2025). An effective masked Transformer model for automatic modulation recognition. IEEE Transactions on Cognitive Communications and Networking.
* Jang, J., Pyo, J., Yoon, Y., & Choi, J. (2024). Meta-Transformer: A meta-learning framework for scalable automatic modulation classification. IEEE Access, 12, 9267–9279.
* Ma, W., Cai, Z., & Wang, C. (2024). A Transformer and convolution-based learning framework for automatic modulation classification. IEEE Communications Letters, 28(6), 1392–1396.
* Huynh-The, T., Hua, C. H., Pham, Q.-V., & Kim, D.-S. (2020). MCNet: An efficient CNN architecture for robust automatic modulation classification. IEEE Communications Letters, 24(4), 811–814.
* P, D., Das, D., & Bora, P. K. (2020). Dense layer dropout based CNN architecture for automatic modulation classification. 2020 IEEE Conference on Signal Processing, Computing and Control (ISPCC).
* Abdulkarem, A. M., Abedi, F., Ghanimi, H. M. A., et al. (2022). Robust automatic modulation classification using convolutional deep neural network based on scalogram information. Computers, 11(11), 162.
* Pu, X., Luo, C., Yin, Y., Liu, Z., & Luo, Y. (2024). Chromosomal mutation-inspired radio augmentation for enhanced automatic modulation classification. IEEE Internet of Things Journal, 11(24), 41124–41136.
* Rajendran, S., Meert, W., Giustiniano, D., Lenders, V., & Pollin, S. (2018). Deep learning models for wireless signal classification with distributed low-cost spectrum sensors. IEEE Transactions on Cognitive Communications and Networking, 4(3), 433–445.
* Chandhok, S., Joshi, H., Darak, S. J., & Subramanyam, A. V. (2020). LSTM guided modulation classification and experimental validation for sub-Nyquist rate wideband spectrum sensing. 2020 IEEE International Symposium on Dynamic Spectrum Access Networks (DySPAN).
* Kang, Y., Cheng, C., Chen, K., & Lv, X. (2024). Mixed signal modulation classification based on deep convolutional neural networks. 2024 3rd International Conference on Electronics and Information Technology (EIT).

---

Happy experimenting! 😊
