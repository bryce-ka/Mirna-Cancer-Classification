# Mirna-Cancer-Classification

**Research Paper and Code for Classification using KNN and Random Forest**

This repository contains the code and research paper for performing classification using K-Nearest Neighbors (KNN) and Random Forest algorithms. The research paper explores the performance of these algorithms on a given dataset and analyzes their accuracy and confusion matrices. The code is implemented in Python using various libraries such as pandas, scikit-learn, and matplotlib.

## Dataset

The classification task is performed on the "combined_dataset.csv" file, which contains the data for the classification problem. The dataset is loaded into a pandas DataFrame using the `read_csv` function. The input features are stored in the variable `X`, and the corresponding labels are stored in the variable `y`.

## Data Preprocessing

Before training the classifiers, the input data is preprocessed using scaling techniques. The `MinMaxScaler` is applied to normalize the input data, ensuring that all features have the same scale. The training and testing data are split using the `train_test_split` function from scikit-learn, with a test size of 30% and a random state of 42.

## Dummy Classifier

To establish a baseline for comparison, a dummy classifier is trained using the `DummyClassifier` class from scikit-learn. The most frequent strategy is used for the dummy classifier, and its accuracy on the test set is calculated using the `accuracy_score` function.

## Baseline KNN Model

A baseline KNN model is trained using the `KNeighborsClassifier` class from scikit-learn. The model is trained on the normalized training data and evaluated on the test set. The accuracy of the baseline KNN model is calculated using the `score` method.

## KNN and Random Forest Classifiers

The KNN and Random Forest classifiers are trained and evaluated using a series of experiments. The number of experiments, k values for KNN, n_estimators values for Random Forest, and test sizes are defined as parameters. For each experiment, the dataset is split into training and testing sets using a different random state.

The KNN classifier is tuned using `GridSearchCV` to find the best values for the number of neighbors (`n_neighbors`) and the weight function (`weights`). The Random Forest classifier is also tuned using `GridSearchCV` to find the best values for the number of estimators (`n_estimators`) and the maximum number of features (`max_features`).

For each experiment, the classifiers are trained on the training data and evaluated on the test data. The accuracy of each classifier is calculated using the `accuracy_score` function. The best accuracy for both KNN and Random Forest classifiers across all experiments is recorded, along with the corresponding predictions and true labels.

## Results

The results of the experiments are displayed, showing the accuracy of the KNN and Random Forest classifiers for each experiment. The confusion matrices for the best KNN and Random Forest classifiers are plotted using the `ConfusionMatrixDisplay` class from scikit-learn and saved as images.

The accuracy scores and diagonal ratios for the best KNN and Random Forest classifiers are printed at the end.

## Conclusion

The research paper provides a detailed analysis of the classification task using KNN and Random Forest algorithms. The code in this repository serves as a demonstration of the implementation and evaluation of these classifiers on a specific dataset.

For more details, please refer to the research paper included in this repository.

## Usage

To run the code, follow these steps:

1. Clone the repository:

```
git clone https://github.com/your-username/repo-name.git
```

2. Install the required dependencies:

```
pip install pandas matplotlib scikit-learn
```

3. Navigate to the repository directory:

```
cd repo-name
```

4. Run
