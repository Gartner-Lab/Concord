
import numpy as np


def calculate_class_weights(class_labels, heterogeneity_scores=None):
    unique_classes, class_counts = np.unique(class_labels, return_counts=True)
    if heterogeneity_scores:
        total_heterogeneity = sum(heterogeneity_scores.values())
        weights = {cls: heterogeneity_scores.get(cls, 1) / total_heterogeneity for cls in unique_classes}
    else:
        total_samples = sum(class_counts)
        weights = {cls: count / total_samples for cls, count in zip(unique_classes, class_counts)}
    return weights