import cv2
from tflite_support.task import core
from tflite_support.task import processor
from tflite_support.task import vision


def calc_probabilities_for_image(image, local_model_file):
    """
    Calculate probabilities for a given image using a specific model file.

    Args:
        image: The input image.
        local_model_file: The path to the local model file.

    Returns:
        List of classification categories and their probabilities.
    """
    base_options = core.BaseOptions(file_name=local_model_file, use_coral=False, num_threads=2)
    classification_options = processor.ClassificationOptions(max_results=-1, score_threshold=0)
    options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
    classifier = vision.ImageClassifier.create_from_options(options)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor_image = vision.TensorImage.create_from_array(rgb_image)
    probabilities = classifier.classify(tensor_image)
    return probabilities.classifications[0].categories


def calculate_consensus_scores(ProbabilitiesForImage):
    """
    Calculate consensus scores based on a list of probabilities for multiple models.

    Args:
        ProbabilitiesForImage: List of probabilities for each model.

    Returns:
        List of consensus scores for each category.
    """
    category_data = {}
    for structure in ProbabilitiesForImage:
        for category in structure:
            category_name = category.category_name
            score = category.score
            if category_name in category_data:
                category_data[category_name]['total_score'] += score
                category_data[category_name]['count'] += 1
            else:
                category_data[category_name] = {'total_score': score, 'count': 1}
    average_data = {
        category_name: category_info['total_score'] / category_info['count']
        for category_name, category_info in category_data.items()
    }
    consensus = [[category_name, average_score] for category_name, average_score in average_data.items()]
    return consensus


def return_best_consensus_category(consensus):
    """
    Find the category with the highest consensus score.

    Args:
        consensus: List of consensus scores for each category.

    Returns:
        A tuple containing the best category name and its score.
    """
    category_with_highest_score = max(consensus, key=lambda x: x[1])
    category_name = category_with_highest_score[0]
    score = category_with_highest_score[1]
    return (category_name, score)


def calc_probabilities_for_all_models(image, models_path, local_model_files):
    """
    Calculate probabilities for all models in a list.

    Args:
        image: The input image.
        models_path: The path where model files are located.
        local_model_files: List of model file names for a specific category.

    Returns:
        List of probabilities for each model.
    """
    ProbabilitiesForImage = []
    for local_model_file in local_model_files:
        ProbabilitiesForImage.append(calc_probabilities_for_image(image, models_path + local_model_file))
    return ProbabilitiesForImage


def calc_consensus(image, models_path, local_model_files):
    """
    Calculate the consensus category and score for a given image.

    Args:
        image: The input image.
        models_path: The path where model files are located.
        local_model_files: List of model file names for a specific category.

    Returns:
        A tuple containing the consensus category name and its score.
    """
    ProbabilitiesForImage = calc_probabilities_for_all_models(image, models_path, local_model_files)
    consensus = calculate_consensus_scores(ProbabilitiesForImage)
    category_name, score = return_best_consensus_category(consensus)
    return category_name, score
