import math

def thanksgiving_dish_predictor(preference_scores: list[float]) -> list[float]:
    """
    Predict the probability of choosing each Thanksgiving dish using softmax.

    Args:
        preference_scores: List of preference scores for each dish
            (e.g., [turkey_score, stuffing_score, cranberry_score, pie_score])

    Returns:
        List of probabilities for each dish
    """
    if not preference_scores:
        return []

   
    exp_scores = [math.exp(score) for score in preference_scores]

    # Normalize the scores to obtain probabilities
    total = sum(exp_scores)
    probabilities = [score / total for score in exp_scores]

    return probabilities