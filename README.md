                                                                                      -----Autism Detection Using Machine Learning-----

This project demonstrates how machine learning can be used to detect autism risk in children. Autism Spectrum Disorder (ASD) is a developmental disorder that affects communication, social interaction, and behavior. Early detection of autism can help children receive the necessary intervention and support, leading to better outcomes. 

Using a dataset with various behavioral and developmental features, the project applies a **Random Forest Classifier** to predict whether a child is at risk of autism. The model is trained on data that includes indicators like speech delay, social interaction patterns, and parental concerns. The goal is to identify children at risk of autism early for appropriate interventions.

## Dataset

The dataset used in this project contains various features related to child development. The following columns are included in the dataset:

| Column Name             | Description                                                      |
|-------------------------|------------------------------------------------------------------|
| **Child_ID**             | Unique identifier for each child                                 |
| **Age**                  | Age of the child (years)                                         |
| **Speech_Delay**         | Whether the child has speech delay (1 = Yes, 0 = No)             |
| **Social_Interaction**   | Whether the child shows signs of social interaction issues (1 = Yes, 0 = No) |
| **Imaginative_Play**     | Whether the child engages in imaginative play (1 = Yes, 0 = No)  |
| **Repetitive_Behaviors** | Whether the child exhibits repetitive behaviors (1 = Yes, 0 = No) |
| **Sensory_Sensitivity**  | Whether the child shows signs of sensory sensitivity (1 = Yes, 0 = No) |
| **Eye_Contact**          | Whether the child has difficulty making eye contact (1 = Yes, 0 = No) |
| **Parental_Concern**     | Whether parents are concerned about their child's development (1 = Yes, 0 = No) |

The target variable, **Autism_Risk**, is a binary classification where:
- `1` indicates the child is at risk for autism.
- `0` indicates the child is not at risk.

## Features

- **Age**: The child's age in years.
- **Speech Delay**: Whether there is a delay in speech development.
- **Social Interaction**: Indicators of social challenges or isolation.
- **Imaginative Play**: Whether the child engages in pretend or creative play.
- **Repetitive Behaviors**: Indicates the presence of repetitive movements or speech.
- **Sensory Sensitivity**: Sensitivity to sounds, lights, or textures.
- **Eye Contact**: Difficulty making eye contact with others.
- **Parental Concern**: Whether the parents have concerns about the child's development.

## Output Metrics

After training the model, the following evaluation metrics are computed:

- **Confusion Matrix**: Displays the counts of True Positives, False Positives, True Negatives, and False Negatives, providing insight into the model's performance.
- **Accuracy**: Proportion of correctly classified samples.
- **Precision**: Proportion of predicted positives that are truly positive.
- **Recall**: Proportion of actual positives that are correctly identified.
- **F1 Score**: Harmonic mean of precision and recall, providing a single metric for model performance.

## Requirements

To run this project, you'll need the following Python libraries:
- `pandas`: For data handling and preprocessing.
- `scikit-learn`: For machine learning and evaluation.
- `matplotlib`: For data visualization.

To install these dependencies, run the following command:

```bash
pip install pandas scikit-learn matplotlib
```

## Steps to Run

### Clone the Repository

1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/Autism-Detection.git
   ```

2. Navigate into the project directory:
   ```bash
   cd Autism-Detection
   ```

### Dataset Setup

Ensure you have your dataset file (CSV) ready. The dataset should have the following columns:

- `Age`
- `Speech_Delay`
- `Social_Interaction`
- `Imaginative_Play`
- `Repetitive_Behaviors`
- `Sensory_Sensitivity`
- `Eye_Contact`
- `Parental_Concern`

### Run the Code

3. Run the Python script to train the model and evaluate performance:
   ```bash
   python autism_detection.py
   ```

### Output

The code will generate:
- **Confusion Matrix Plot**: Visualizes the results of the classification model.
- **Precision-Recall Curve**: A graphical representation of the precision-recall trade-off.

Console output will display metrics like accuracy, precision, recall, and F1 score.

### Code Walkthrough

The project includes the following major steps:

1. **Loading Data**: The dataset is read into a pandas DataFrame.
2. **Preprocessing**: Missing values are handled (optional) and the target variable (`Autism_Risk`) is generated based on certain behavioral columns.
3. **Model Training**: A **Random Forest Classifier** is used to train the model with the data. Class weights are balanced to mitigate any class imbalance.
4. **Model Evaluation**: The trained model is evaluated using metrics such as confusion matrix, accuracy, precision, recall, and F1 score.
5. **Visualization**: The confusion matrix and precision-recall curve are plotted using `matplotlib`.

