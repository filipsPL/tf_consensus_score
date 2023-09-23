# tf_consensus_score

This module contains a collection of functions for calculating consensus scores from multiple TensorFlow Lite classificaitons.

```text
┌─────────────┐
│             │
│  Image      │       ┌─────────────────────────────────┐      ┌──────────────────────┐
│             ├──────►│                                 │      │                      │
└─────────────┘       │                                 │      │                      │
                      │                                 │      │ Output:              │
┌─────────────┐       │      tf_consensus_score.py      ├─────►│                      │
│tfl model 1  ├──────►│                                 │      │ cat 0.9255041480     │
├─────────────┤       │                                 │      │                      │
│tfl model 2  │       │                                 │      │                      │
├─────────────┤       └─────────────────────────────────┘      └──────────────────────┘
│tfl model n  │
└─────────────┘
```


Usage:

```python
import cv2
from tf_consensus_score import *

# Initialize a dictionary for local model files
model_files = {}

# Add model files to the dictionary
model_files = [
    "animals1.tflite",
    "animals2.tflite"
]

# Define the path where models are located
models_path = "models/"

# Read the image the cv2 way
image = cv2.imread("cat.jpg", cv2.IMREAD_COLOR)

# Calc consensus score and return the highest scored class
category_name, score = calc_consensus(image, models_path, model_files)
print(category_name, score)

# cat 0.9255041480
```

## Installation

Just download `tf_consensus_score.py` and put in the program directory.

## Functions

### `calc_probabilities_for_image(image, local_model_file)`

Calculate probabilities for a given image using a specific model file.

#### Arguments

- `image`: The input image to be classified.
- `local_model_file`: The path to the local TensorFlow Lite model file.

#### Returns

A list of classification categories and their corresponding probabilities.

### `calculate_consensus_scores(ProbabilitiesForImage)`

Calculate consensus scores based on a list of probabilities for multiple models.

#### Arguments

- `ProbabilitiesForImage`: A list of probabilities for each model.

#### Returns

A list of consensus scores for each category based on the input probabilities.

### `return_best_consensus_category(consensus)`

Find the category with the highest consensus score.

#### Arguments

- `consensus`: A list of consensus scores for each category.

#### Returns

A tuple containing the best category name and its score.

### `calc_probabilities_for_all_models(image, models_path, local_model_files)`

Calculate probabilities for all models in a list.

#### Arguments

- `image`: The input image to be classified.
- `models_path`: The path where model files are located.
- `local_model_files`: A list of model file names for a specific category.

#### Returns

A list of probabilities for each model in the input list.

### `calc_consensus(image, models_path, local_model_files)`

Calculate the consensus category and score for a given image.

#### Arguments

- `image`: The input image to be classified.
- `models_path`: The path where model files are located.
- `local_model_files`: A list of model file names for a specific category.

#### Returns

A tuple containing the consensus category name and its score based on the input probabilities.
