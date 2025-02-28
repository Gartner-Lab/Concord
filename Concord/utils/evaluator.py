
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .. import logger


def log_classification(epoch, phase, preds, labels, unique_classes, logger):
    from sklearn.metrics import classification_report
    
    unique_classes_str = [str(cls) for cls in unique_classes]
    report = classification_report(y_true=labels, y_pred=preds, labels=unique_classes, output_dict=True)
    accuracy = report['accuracy']
    precision = {label: metrics['precision'] for label, metrics in report.items() if label in unique_classes_str}
    recall = {label: metrics['recall'] for label, metrics in report.items() if label in unique_classes_str}
    f1 = {label: metrics['f1-score'] for label, metrics in report.items() if label in unique_classes_str}

    # Create formatted strings for logging
    precision_str = ", ".join([f"{label}: {value:.2f}" for label, value in precision.items()])
    recall_str = ", ".join([f"{label}: {value:.2f}" for label, value in recall.items()])
    f1_str = ", ".join([f"{label}: {value:.2f}" for label, value in f1.items()])

    # Log to console
    logger.info(
        f'Epoch: {epoch:3d} | {phase.capitalize()} accuracy: {accuracy:5.2f} | precision: {precision_str} | recall: {recall_str} | f1: {f1_str}')

    return accuracy, precision, recall, f1



def evaluate_classification(class_true, class_pred, target_names=None, figsize=(5,3), dpi=300, save_path = None):
    from sklearn.metrics import classification_report, confusion_matrix
    # Calculate metrics
    report = classification_report(class_true, class_pred, target_names=target_names, output_dict=True)
    accuracy = report['accuracy']
    precision = {label: metrics['precision'] for label, metrics in report.items() if label in target_names}
    recall = {label: metrics['recall'] for label, metrics in report.items() if label in target_names}
    f1 = {label: metrics['f1-score'] for label, metrics in report.items() if label in target_names}


    # Generate confusion matrix
    conf_matrix = confusion_matrix(class_true, class_pred)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    if save_path:
        plt.savefig(save_path, dpi=dpi)

    plt.show()
    return accuracy, precision, recall, f1, conf_matrix