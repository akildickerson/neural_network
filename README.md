# Neural Network

A neural network implementation for heart disease prediction using the heart dataset.

## Project Structure

```
neural_network/
├── data/
│   ├── raw/
│   │   └── heart.csv
│   └── processed/
│       └── normalized_heart.csv
├── src/
│   ├── data/
│   │   └── preprocess.py
│   ├── models/
│   │   └── neural_network.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/akildickerson/neural_network.git
cd neural_network
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python src/main.py
```

## Features

- Custom neural network implementation from scratch
- Tanh normalization for numerical features
- Hyperparameter tuning (learning rate optimization)
- Train/validation/test split
