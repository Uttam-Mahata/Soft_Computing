

# Housing Price Analysis - Overfitting and Underfitting Study

This project explores overfitting and underfitting issues in machine learning models using housing price prediction as a case study. The analysis is implemented using Python and popular data science libraries.

## Project Overview

The project analyzes housing price data to demonstrate:
- Data preprocessing and feature engineering
- Model training and evaluation
- Analysis of overfitting and underfitting scenarios
- Comparison of different regression models

## Dataset

The analysis uses housing price datasets from:
- Housing Price Dataset (Kaggle)
- Melbourne Housing Snapshot Dataset (Kaggle)

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Key Features

- Data preprocessing and cleaning
- Feature selection and engineering
- Model implementation:
  - Support Vector Machine (SVM)
  - Decision Tree
  - Random Forest
  - Logistic Regression
- Performance metrics:
  - Mean Squared Error
  - Mean Absolute Error
  - Confusion Matrix
  - Accuracy Score

## File Structure

```
Assignment1/
│
└── housing.ipynb      # Main Jupyter notebook containing the analysis
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Uttam-Mahata/Soft_Computing.git
```

2. Install required dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook Assignment1/housing.ipynb
```

## Key Findings

The notebook demonstrates:
- Impact of feature selection on model performance
- Comparison of different regression models
- Analysis of overfitting and underfitting scenarios
- Model optimization techniques




# Iris Flower Classification

This project implements machine learning models to classify iris flowers into three species (Iris-setosa, Iris-versicolor, and Iris-virginica) based on their sepal and petal measurements. 

## Overview

The project uses the famous Iris dataset to demonstrate:
- Data exploration and visualization 
- Model training and evaluation
- Comparison of different classification algorithms

## Dataset Features

The dataset contains 150 samples with 4 features each:
- Sepal length (cm)
- Sepal width (cm) 
- Petal length (cm)
- Petal width (cm)

Target classes:
- Iris Setosa
- Iris Versicolour 
- Iris Virginica

## Models Implemented

- Logistic Regression (97.8% accuracy)
- Decision Tree Classifier (93.3% accuracy) 
- K-Nearest Neighbors (95.6% accuracy)
- Support Vector Machine (97.8% accuracy)

## Requirements

- Python 3.x
- pandas 
- numpy
- matplotlib
- seaborn
- scikit-learn

## Project Structure

```
Assignment1/
│
├── iris.ipynb      # Main Jupyter notebook with analysis
└── IRIS.csv       # Dataset file
```

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Uttam-Mahata/Soft_Computing.git
```

2. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Run the Jupyter notebook:
```bash 
jupyter notebook Assignment1/iris.ipynb
```

## Key Findings

- SVM and Logistic Regression achieved the highest accuracy (97.8%)
- Iris-setosa is linearly separable from the other species
- Petal measurements provide better separation between species compared to sepal measurements
- All models performed well with accuracies above 90%

## Visualizations

The notebook includes:
- Scatter plots of feature relationships
- Pair plots showing feature distributions
- Correlation heatmaps
- Confusion matrices for model evaluation


# Mobile Price Classification Analysis

This project demonstrates the analysis of mobile phone price classification using various machine learning models, with a focus on understanding and addressing overfitting and underfitting issues.

## Project Overview

The project analyzes mobile phone features to predict price ranges using multiple classification models. It includes comprehensive data analysis, model training, and evaluation of model performance with respect to overfitting and underfitting.

## Dataset Features

The dataset includes 21 features:
- Battery power
- Bluetooth (blue)
- Clock speed
- Dual SIM
- Front Camera (fc)
- 4G support
- Internal memory
- Mobile depth (m_dep)
- Mobile weight
- Number of cores
- Primary camera
- Pixel resolution (height & width)
- RAM
- Screen dimensions (sc_h, sc_w)
- Talk time
- 3G support
- Touch screen
- WiFi
- Price range (target variable)

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

## Project Structure

```
Assignment1/
│
├── mobile_price.ipynb        # Main Jupyter notebook with analysis
├── mobile_price_train.csv    # Training dataset
└── mobile_price_test.csv     # Test dataset
```

## Implementation Details

### Data Preprocessing
- Data loading and inspection
- Feature analysis
- Missing value detection
- Duplicate check
- Statistical summary

### Models Implemented
- Support Vector Machine (SVC)
- Decision Tree Classifier
- Random Forest Classifier
- Logistic Regression

### Analysis Focus
- Overfitting detection
- Underfitting analysis
- Model performance comparison
- Feature importance evaluation

## Getting Started

1. Clone the repository:
```bash
git clone https://github.com/Uttam-Mahata/Soft_Computing.git
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

3. Open and run the Jupyter notebook:
```bash
jupyter notebook Assignment1/mobile_price.ipynb
```






# Fuzzy Logic Implementation for Speed Limit Analysis

This project demonstrates the implementation of fuzzy logic concepts using Python, specifically focusing on analyzing speed limits through fuzzy set operations.

## Project Overview

The project implements fuzzy set operations to analyze speed limits on a highway using two fuzzy sets:
- Set A: "Low speed limit"
- Set B: "High speed limit"

## Features

- Implementation of membership functions for low and high speed limits
- Visualization of fuzzy sets using matplotlib
- Implementation of basic fuzzy set operations:
  - Union (A ∪ B)
  - Intersection (A ∩ B)
  - Complement (A', B')
  - Difference (A - B)

## Mathematical Definitions

### Fuzzy Set A (Low speed limit)
μA(x) = 
- 1, if x < 30 (low)
- (x−30)/(50−30), if 30 < x < 50 (moderately low)
- (x−50)/(70−50), if 50 < x < 70 (increasing)
- 0, if x > 70 (not low)

### Fuzzy Set B (High speed limit)
μB(x) = 
- 0, if x < 60 (Not high)
- (x−60)/(80−60), if 60 < x < 80 (moderately high)
- (x−80)/(100−80), if 80 < x < 100 (increasing)
- 1, if x > 100 (high)

## Requirements

- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Uttam-Mahata/Soft_Computing.git
```

2. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage

1. Navigate to the Assignment2 directory:
```bash
cd Assignment2
```

2. Run the Jupyter notebook:
```bash
jupyter notebook Fuzzy_logic.ipynb
```

## Project Structure

```
Assignment2/
│
└── Fuzzy_logic.ipynb    # Main Jupyter notebook with implementations
```

## Implementation Details

The project includes:
1. Definition of membership functions for both fuzzy sets
2. Visualization of individual fuzzy sets
3. Implementation of fuzzy set operations
4. Graphical representation of results

## Visualization Examples

The notebook includes visualizations for:
- Membership function for low speed limit (Set A)
- Membership function for high speed limit (Set B)
- Union of sets A and B
- Intersection of sets A and B











# Fuzzy Numbers Implementation

## Description
This project demonstrates the implementation of fuzzy logic concepts, including fuzzy numbers and a Mamdani Fuzzy Inference System for temperature-based fan speed control. The project includes implementations of different membership functions and a practical application in control systems.

## Features
1. Implementation of three types of fuzzy membership functions:
   - Trapezoidal
   - Triangular
   - Gaussian

2. Fuzzy Control System for Fan Speed Control:
   - Temperature-based input
   - Fan speed control output
   - Mamdani inference system implementation
   - Centroid defuzzification

## Mathematical Implementations

### 1. Membership Functions

#### Triangular Membership Function
```python
def triangular(x, a, b, c):
    return np.maximum(0, np.minimum((x-a)/(b-a), (c-x)/(c-b)))
```

#### Trapezoidal Membership Function
```python
def trapezoidal(x, a, b, c, d):
    return np.maximum(0, np.minimum(np.minimum((x-a)/(b-a), 1), (d-x)/(d-c)))
```

#### Gaussian Membership Function
```python
def gaussian(x, c, sigma):
    return np.exp(-0.5 * ((x - c) / sigma) ** 2)
```

## Fan Speed Control System

### Components
1. **Input Variable**: Temperature (200-1000°C)
2. **Output Variable**: Fan Speed (500-3000 RPM)
3. **Linguistic Variables**:
   - Temperature: "Risky", "Average", "Excellent"
   - Fan Speed: "Slow", "Moderate", "High"

### Fuzzy Rules
1. IF Temperature is "Risky", THEN Fan-Speed is "Low"
2. IF Temperature is "Average", THEN Fan-Speed is "Moderate"
3. IF Temperature is "Excellent", THEN Fan-Speed is "High"

## Requirements
- Python 3.x
- NumPy
- Matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Uttam-Mahata/Soft_Computing.git
```

2. Install required packages:
```bash
pip install numpy matplotlib
```

## Usage

1. Navigate to the Assignment3 directory:
```bash
cd Assignment3
```

2. Run the Jupyter notebook:
```bash
jupyter notebook Fuzzy_Numbers.ipynb
```

## Project Structure

```
Assignment3/
│
└── Fuzzy_Numbers.ipynb    # Main Jupyter notebook with implementations
```

## Features
- Interactive input for temperature values
- Visual representation of membership functions
- Real-time calculation of fan speed
- Graphical output of fuzzy inference process

## Implementation Details

### 1. Temperature Membership Functions
- Risky: Triangular (200-400°C)
- Average: Trapezoidal (380-750°C)
- Excellent: Triangular (600-1000°C)

### 2. Fan Speed Membership Functions
- Slow: Trapezoidal (500-1600 RPM)
- Moderate: Triangular (1200-2200 RPM)
- High: Trapezoidal (2000-3000 RPM)






# Breast Cancer Classification using Neural Network

A neural network implementation for classifying breast cancer cases as either recurrent or non-recurrent events using the UCI Breast Cancer dataset.

## Overview

This project implements a feedforward neural network to predict breast cancer recurrence. The model analyzes various medical features to determine the likelihood of cancer recurrence in patients.

## Dataset

The dataset used is the Breast Cancer dataset from the UCI Machine Learning Repository, which includes the following features:

- age
- menopause
- tumor-size
- inv-nodes
- node-caps
- deg-malig
- breast
- breast-quad
- irradiat
- Class (Target variable: recurrence-events/no-recurrence-events)

## Implementation Details

### Data Preprocessing
- Handled missing values using mode imputation
- Performed one-hot encoding for categorical variables
- Applied standardization to numeric features
- Split data into 70% training and 30% testing sets

### Neural Network Architecture
- Input Layer: Based on number of features after one-hot encoding
- Hidden Layer: 10 neurons with sigmoid activation
- Output Layer: 1 neuron with sigmoid activation

### Training Parameters
- Learning Rate: 0.001
- Epochs: 1000
- Loss Function: Mean Squared Error
- Activation Function: Sigmoid

## Results

The model achieved the following performance metrics on the test set:

### Binary Classification Metrics
- Accuracy: ~65%
- Precision: 
  * No-recurrence: 0.70
  * Recurrence: 0.38
- Recall:
  * No-recurrence: 0.86
  * Recurrence: 0.19
- F1-Score:
  * No-recurrence: 0.77
  * Recurrence: 0.25

## Key Features

1. Custom implementation of neural network components:
   - Forward propagation
   - Backward propagation
   - Gradient descent optimization

2. Visualization:
   - Error vs Epochs plot
   - Confusion Matrix visualization

3. K-Fold Cross Validation for robust model evaluation

## Requirements

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

## Usage

1. Install required packages:
```bash
pip install numpy pandas scikit-learn matplotlib
```

2. Run the notebook:
```bash
jupyter notebook breast_cancer_classification.ipynb
```

## Future Improvements

1. Experiment with different network architectures
2. Implement regularization techniques
3. Try different optimization algorithms
4. Feature selection/engineering
5. Hyperparameter tuning



# Neural Network Implementation for 7-Segment Display Recognition
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)

## Overview

A neural network implementation for recognizing patterns in 7-segment displays. The project focuses on binary classification to distinguish between valid and invalid 7-segment digit patterns.

## Problem Description

The task is to implement a neural network that can:
1. Learn the patterns of valid 7-segment display configurations for digits 0-9
2. Distinguish between valid and invalid patterns using binary classification
3. Achieve high accuracy and reliability in pattern recognition

## Network Architecture

### Layer Configuration
- **Input Layer**: 7 neurons (one for each segment a-g)
- **Hidden Layer 1**: 10 neurons
- **Hidden Layer 2**: 10 neurons  
- **Output Layer**: 1 neuron (binary classification)

### Activation Functions
- Sigmoid activation used throughout the network
- Sigmoid derivative for backpropagation

### Loss Function
- Mean Squared Error (MSE)

## Dataset

### Data Generation
- **Valid Patterns**: Predefined patterns for digits 0-9
- **Invalid Patterns**: 100 randomly generated invalid configurations
- **Balance**: Oversampling of valid patterns (10x duplication)
- **Format**: CSV file with 8 columns (7 segments + label)

## Implementation Details

### Core Components

1. **Neural Network Class**
   - Forward propagation
   - Backward propagation
   - Training with configurable epochs
   - Gradient descent optimization

2. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1 Score
   - Specificity
   - Confusion Matrix

3. **Cross-Validation**
   - 5-fold stratified cross-validation
   - Average performance metrics
   - Confusion matrix visualization

### Training Parameters
- Learning Rate: 0.01
- Epochs: 900
- Train-Test Split: 80-20

## Results

The model achieves:
- Accuracy: 99%
- Precision: 98.1%
- Recall: 100%
- F1 Score: 99%
- Specificity: 98%

## Requirements

```python
numpy
pandas
matplotlib
scikit-learn
seaborn
```

## Usage

1. Install dependencies:
```bash
pip install numpy pandas matplotlib scikit-learn seaborn
```

2. Run the Jupyter notebook:
```bash
jupyter notebook 7-segment-digit.ipynb
```

3. Dataset generation:
```python
# The code will automatically generate and save the dataset as 'balanced_seven_segment_digit_dataset.csv'
```

## File Structure

```
.
├── 7-segment-digit.ipynb      # Main implementation notebook
├── balanced_seven_segment_digit_dataset.csv  # Generated dataset
└── README.md                  # Project documentation
```

## Training Visualization

The implementation includes:
- Loss vs Iterations plot
- Confusion Matrix heatmap
- Cross-validation performance metrics

## Technical Notes

1. **Data Preprocessing**
   - Binary encoding of segments
   - Stratified sampling for cross-validation
   - Balanced dataset creation

2. **Model Training**
   - Gradient descent optimization
   - MSE loss minimization
   - Learning rate scheduling

3. **Performance Optimization**
   - K-fold cross-validation
   - Oversampling for class balance
   - Metrics averaging across folds




# Traveling Salesman Problem (TSP) Solver using Genetic Algorithm

This project implements a Genetic Algorithm solution for the Traveling Salesman Problem (TSP) using real-world Indian city coordinates.

## Overview

The Traveling Salesman Problem is a classic optimization challenge where the goal is to find the shortest possible route that visits each city exactly once and returns to the starting point. This implementation uses:

- Real coordinates of 53 major Indian cities
- Genetic Algorithm optimization
- Geodesic distance calculations
- Matplotlib visualizations

## Features

- City coordinate retrieval using GeoPy
- Distance matrix calculation using geodesic distances
- Genetic Algorithm implementation with:
  - Roulette wheel selection
  - Order crossover (OX)
  - Swap mutation
  - Elitism preservation
- Route visualization and progress tracking
- CSV exports of coordinates and distance matrices

## Requirements

```python
geopy==2.3.0
pandas==1.5.3
numpy==1.24.3
matplotlib==3.7.1
```

## Installation

```bash
git clone https://github.com/Uttam-Mahata/Soft_Computing.git
cd Assignment6
pip install -r requirements.txt
```

## Usage

1. **Generate City Coordinates**
```python
python tsp_ga.ipynb
```

2. **Run the Genetic Algorithm**
```python
# Configure parameters in the notebook:
POPULATION_SIZE = 100
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.1
ELITE_SIZE = 10
```

## Algorithm Details

### 1. Population Initialization
- Random permutation of cities
- Population size: 100 routes

### 2. Fitness Function
```python
F(Ck) = 1 / (Total Distance of Route Ck)
```
where Total Distance is calculated using geodesic distances between consecutive cities.

### 3. Selection
- Roulette wheel selection
- Selection probability proportional to fitness
- Elite preservation of best routes

### 4. Crossover
- Order Crossover (OX)
- Preserves order and position of some cities
- Probability: 0.6

### 5. Mutation
- Swap mutation
- Random city pair exchange
- Probability: 0.1

## Data Format

### Input Data
```csv
City,Latitude,Longitude
Delhi,28.6273928,77.1716954
Mumbai,19.0815772,72.8866275
...
```

### Distance Matrix
```csv
City,Delhi,Mumbai,...
Delhi,0,1147.3,...
Mumbai,1147.3,0,...
...
```

## Results

- Convergence typically occurs within 500-1000 generations
- Average improvement in route length: 30-40%
- Final routes maintain feasibility constraints
- Computation time scales with O(n²) for n cities

## Visualization

The implementation includes:
- Route visualization on India map
- Generation-wise fitness plots
- Distance matrix heatmaps

## Project Structure

```
Assignment6/
│
├── tsp_ga.ipynb          # Main implementation notebook
├── data/
│   ├── indian_cities_coordinates.csv
│   └── distance_matrix.csv
├── requirements.txt
└── README.md
```

## Mathematical Formulation

### Objective Function
Minimize the total route distance:
```
min Σ(i=1 to n) Σ(j=1 to n) dij * xij
```
where:
- dij = distance between cities i and j
- xij = 1 if route goes from i to j, 0 otherwise

### Constraints
1. Each city must be visited exactly once
2. Route must form a complete cycle
3. No sub-tours allowed



## Acknowledgments
- GeoPy for coordinate retrieval
- Indian cities dataset
- Genetic Algorithm implementation references



## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request






## License

This project is part of the Soft Computing repository. See the repository's license for more details.



## Author

- Uttam Mahata ([@Uttam-Mahata](https://github.com/Uttam-Mahata))
