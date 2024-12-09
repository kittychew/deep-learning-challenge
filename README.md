# Report on the Neural Network Model
## Overview of the Analysis

Neural networks are a type of machine learning model inspired by the human brain, designed to recognize patterns and make predictions. They consist of layers of interconnected nodes, or "neurons," where each neuron processes data and passes it to the next layer. The goal of this project was to use a neural network to predict whether a charity organization would be successful based on various features like their application type, classification, and financial information.

In this project, we started with a simple neural network with two hidden layers, using **ReLU activation**. To improve the model, we added a third hidden layer, increased the number of neurons, and tested different **dropout rates** (0.2 and 0.4) to prevent overfitting. We also experimented with the **tanh activation function**, though it didn’t yield better results. We used **early stopping** to stop training early when the model’s performance stopped improving, and chose **Stochastic Gradient Descent (SGD)** as the optimizer.

Despite these optimizations, our model achieved an accuracy of **72.96%** and a loss of **0.55**, which was below the target accuracy. Through this process, we learned how tuning different aspects of the neural network, such as the number of layers, neurons, and training techniques, can impact the model’s performance.


## Data Preprocessing

### Target Variable(s)
The target variable for the model is `IS_SUCCESSFUL`, which indicates whether the organization was successful in securing funding (1 for success, 0 for failure).

### Feature Variables
The features used to train the model include variables like `APPLICATION_TYPE`, `CLASSIFICATION`, `INCOME_AMT`, `ASK_AMT`, and others. These features were selected based on their relevance to the classification task.

### Variables Removed
The columns `EIN` and `NAME` were removed from the dataset as they are unique identifiers and do not provide useful information for classification.


## Compiling, Training, and Evaluating the Model

### Model Architecture
- **Input Layer**: The input layer size is based on the number of features in the dataset (after preprocessing).
- **Hidden Layers**:
  - First hidden layer: 128 neurons with ReLU activation.
  - Second hidden layer: 64 neurons with ReLU activation.
  - Third hidden layer: 32 neurons with ReLU activation.
- **Output Layer**: The output layer consists of 1 neuron with a sigmoid activation function for binary classification.

### Model Performance
- **Accuracy**: After training the model, the accuracy achieved on the test set was approximately **72.96%**.
- **Loss**: The model’s loss was **0.55** on the test set.

The following two plots show the **Training vs Validation Loss** (left) and **Training vs Validation Accuracy** (right) for the deep learning model across epochs.

![Training vs Validation Loss and Accuracy](Images/accuracy_plot.png)

- **Training vs Validation Loss**:
  - The training loss steadily decreases, indicating that the model is learning and improving. 
  - However, the validation loss does not decrease at the same rate and flattens out, suggesting that the model may not be generalizing well on unseen data.

- **Training vs Validation Accuracy**:
  - The training accuracy continues to increase, reflecting improved performance on the training dataset.
  - The validation accuracy shows a similar upward trend but is lower than the training accuracy, which may indicate **overfitting**. This means the model performs well on the training data but struggles with new, unseen data.

#### **Takeaway**:
From these plots, it's evident that while the model performs well on the training data, there are signs of overfitting, as shown by the validation accuracy lagging behind the training accuracy. Techniques like early stopping, regularization, or using a different model could help reduce overfitting and improve generalization to unseen data.

### Attempts to Improve the Model
To optimize the model further, I experimented with several changes:
- **Changed the number of epochs** from 50 to 200, expecting it would allow the model to learn more effectively.
- **Tried different dropout rates**: I tested dropout rates of **0.2** and **0.4** in the hidden layers, aiming to prevent overfitting. The model's performance did not significantly change with either rate.
- **Used the SGD optimizer** with a learning rate of 0.01 and momentum of 0.9, hoping to improve the model’s convergence.
- **Added early stopping** with a patience value of 10 to stop training early if the model stopped improving.
- **Tried changing the activation function** from `relu` to `tanh` in the hidden layers, hoping it would enhance performance.
- **Added a third hidden layer** and **increased the number of neurons** in the existing layers, hoping it would improve model complexity and performance.

Despite these improvements, **none of the changes led to a significant improvement** in the model’s performance. The accuracy remained at **72.96%** and the loss remained at **0.55**, suggesting that the model might be nearing its performance limit with the given configuration.

### **Recommendations:**

The deep learning model developed for the classification task achieved an accuracy of approximately 73%. Despite various optimizations—such as adjusting the number of hidden layers, increasing the number of neurons, tweaking the dropout rate, and changing the activation function and optimizer—the model did not surpass the target accuracy of 75%.

To improve the performance, I suggest experimenting with the following approaches:

- **Logistic Regression:** While our neural network is a more complex model, logistic regression can be a simpler alternative that may perform well, especially when the data has a more linear relationship. It’s quick to implement and could provide a baseline for comparison.

- **Random Forest:** This ensemble model combines multiple decision trees to make more robust predictions. Random Forests are less prone to overfitting and can handle both linear and non-linear data patterns effectively. It could potentially outperform the neural network with less tuning required.

By trying these models and fine-tuning the parameters, we could potentially achieve the target accuracy and provide a better solution for this classification problem.
