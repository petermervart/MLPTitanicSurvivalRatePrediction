# Predicting Titanic Survival Chance with MLP

**Authors:** Michal Lüley, Peter Mervart

This project aims to predict the survival chances of passengers on the Titanic using a Multilayer Perceptron (MLP) model implemented in TensorFlow. The dataset used for this project is the Titanic dataset, which contains information about the passengers and whether they survived.

Detailed explanation with visualizations in jupyter notebook: [Notebook](TitanicSurvivalRate.ipynb)

## Libraries Used
- `pandas`
- `seaborn`
- `matplotlib`
- `numpy`
- `sklearn`
- `tensorflow`

## Dataset
The dataset is read from a CSV file named `titanic_dataset.csv`. It includes the following columns:
- Survived
- Pclass
- Name
- Sex
- Age
- Siblings/Spouses Aboard
- Parents/Children Aboard
- Fare

### Data Preprocessing
1. **Renaming Columns:** The dataset columns are renamed for easier access.
2. **Extracting Titles:** Titles are extracted from names and standardized (e.g., "Mlle" to "Miss").
3. **Encoding Categorical Data:** The `sex` and `title` columns are encoded using `LabelEncoder`.
4. **Exploratory Data Analysis (EDA):** Various histograms are plotted to visualize the distributions of different features.

### Data Augmentation
A function is implemented to augment the dataset by scaling or translating the `fare` and `age` columns.

### Data Transformation
Different scalers (e.g., `MinMaxScaler`, `StandardScaler`) and a `PowerTransformer` are applied to preprocess the features.

### Feature Selection
A correlation matrix is created to select relevant features for the model.

### Train-Test Split
The dataset is split into training and testing sets using an 90-10% ratio.

### Model Creation
An MLP model is created using TensorFlow with:
- Input layer
- Dense layers with `leaky_relu` activation
- Dropout layers for regularization
- Output layer with sigmoid activation

### Callbacks
Early stopping and model checkpointing are implemented to save the best model weights.

### Model Training and Evaluation
The model is trained using K-Fold cross-validation. Training and validation accuracies are stored for later analysis.

## Results
The average accuracies achieved during the model training and evaluation are as follows:
- Average accuracy (train): 0.859
- Average accuracy (valid): 0.775
- Average accuracy (test): 0.816

The training and validation accuracies are monitored, and the model performance is evaluated based on the best checkpoint weights saved during training.

## Conclusion
This project provides insights into predicting Titanic survival chances using a neural network approach and highlights the importance of data preprocessing and model evaluation techniques.
