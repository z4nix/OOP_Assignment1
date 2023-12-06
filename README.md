# OOP - 2023/24 - Assignment 1

This is the base repository for assignment 1.
Please follow the instructions given in the [PDF](https://brightspace.rug.nl/content/enforced/243046-WBAI045-05.2023-2024.1/2023_24_OOP.pdf) for the content of the exercise.

## How to carry out your assignment

1. Clone this template into a private repository copying both the main and the submission branches.
2. Please add your partner and `oop-otoz` to the collaborators.
3. Create your code in the `main` branch.
4. Once you are done with the assignment (or earlier), create a pull request from the `main` branch to your `submission` branch and add `oop-otoz` to the reviewers.

The assignment is divided into 4 blocks.
Block 1, 2, and 3 all define different classes.

Put the three classes in three separate files in the `src` folder, with the names specified in the PDF.
**Leave the __init__.py file untouched**.

Put the **main.py** script **outside** of the `src` folder, in the root of this repo.

Below this line, you can write your report to motivate your design choices.

## Submission

The code should be submitted on GitHub by opening a Pull Request from the branch you were working on to the `submission` branch.

There are automated checks that verify that your submission is correct:

1. Deadline - checks that the last commit in a PR was made before the deadline
2. Reproducibility - downloads libraries included in `requirements.txt` and runs `python3 src/main.py`. If your code does not throw any errors, it will be marked as reproducible.
3. Style - runs `flake8` on your code to ensure adherence to style guides.

---
# Code Report - Design & Choices
 
## MultipleLinearRegressor

## Introduction
The `MultipleLinearRegressor` class is designed for performing multiple linear regression, using the Linear Regressor code covered in class as a blueprint. This part of the report describes the design choices regarding the public and private members of this class, together with its general functionality. 

## Attributes

### 1. `self._intercept` and `self._slope`
- **Private or Public?:** Private
- **Motivation:** 
  - These attributes represent the internal state of the regression model, more specifically, the intercept and slope (or slopes) of the regression line. Making them private ensures that they cannot be directly modified from outside the class, maintaining the model integral. 
  - We decided that changes to these attributes should only be made through designated methods (`train` and `set_params`), which include all  of tthe necessary validations and computations

### 2. Other Attributes
- All other attributes were set to public

## Methods

### 1. `train`
- **Choice:** Public
- **Motivation:** 
  - This method allows users to train the regression model on the data being inputted, therefore, they need to be able to access it

### 2. `predict`
- **Choice:** Public
- **Motivation:** 
  - **Model Utilization:** After training, `predict` is used to make predictions on new data, a fundamental aspect of any regression model.

### 3. `get_params` and `set_params`
- **Choice:** Public
- **Motivation:** 
  - These methods provide an interface  for users to access / modify the model's parameters. Making them public allows for dynamic use while (as mentioned earlier) maintaining the model's integrity

### 4. Exception Handling in `train`
   - Exception handling for singular matrix inversion is critical to make sure that the matrix inputted is indeed invertible. Necessary type-checks were also inserted.

### 5. Exception Handling in Other Methods
  - **Type Checking in `train`, `predict`, and `set_params`:** makes sure inputs are of expected types, preventing wrong inputs and unnecessary computation to make up for the error. 
  - **Validation in `set_params`:** Exception handling as an object. Ensures params are of type dict. 
    
## Additional Notes
- **Type Checking:** The methods include type checks for inputs, improving the robustness of the class
- **Encapsulation:** Private attributes take advantage of encapsulation, ensuring that the internal state of the class is protected from potential interference or misuse

## Conclusion
The  `MultipleLinearRegressor` class has been created keeping in mind the principles of encapsulation and robustness. The private attributes protect the internal state of the model, while the public methods provide the functionality to train and use the regression model.

# RegressionPlotter

-   The `plot` method adjusts based on the dataset's features. For 1D features, `plotND` is utilized for the feature-target relationship visualization. For 2D features, there is an option to select either a 3D plot (`plot3D`) or individual 2D plots (`plotND`).
- The use of `plot3D` and `plotND`, without a separate `plot2D` method, simplifies the class while providing visualization options for any feature dimension.
- the variable `feature_number` was privatized because it is an internal state variable used for logic within the class rather than something that should be directly modified by users

# ModelSaver
-  The class supports multiple serialization formats (JSON and Pickle) to work with different use cases: JSON for human readable and interoperable data, and Pickle for more efficient Python-object handling. The preferred format can be set at initialization.
  
- We made the ModelSaver class independent of any specific model, as underlined in the assignment file. It uses generic get_parameters and set_parameters methods, which means that any model class containing such methods will be able to use the ModelSaver class.
  
- We added exceptions to the file, in order to ensure the following: permission errors, incorrect file paths, etc. These errors are all related to the user.
  






