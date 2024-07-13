
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from ..model.predict_functions import predict_with_model
from .plotting import Plotter


def log_classification(epoch, phase, preds, labels, logger, target_names):
    # Calculate metrics
    report = classification_report(labels, preds, target_names=target_names, output_dict=True)
    accuracy = report['accuracy']
    precision = {label: metrics['precision'] for label, metrics in report.items() if label in target_names}
    recall = {label: metrics['recall'] for label, metrics in report.items() if label in target_names}
    f1 = {label: metrics['f1-score'] for label, metrics in report.items() if label in target_names}

    # Logging to wandb
    wandb.log({f"{phase}/accuracy": accuracy})
    wandb.log({f"{phase}/precision_{label}": value for label, value in precision.items()})
    wandb.log({f"{phase}/recall_{label}": value for label, value in recall.items()})
    wandb.log({f"{phase}/f1_{label}": value for label, value in f1.items()})

    # Create formatted strings for logging
    precision_str = ", ".join([f"{label}: {value:.2f}" for label, value in precision.items()])
    recall_str = ", ".join([f"{label}: {value:.2f}" for label, value in recall.items()])
    f1_str = ", ".join([f"{label}: {value:.2f}" for label, value in f1.items()])

    # Log to console
    logger.info(
        f'Epoch: {epoch:3d} | {phase.capitalize()} accuracy: {accuracy:5.2f} | precision: {precision_str} | recall: {recall_str} | f1: {f1_str}')

    return accuracy, precision, recall, f1




def evaluate_classifier(model, dataloader, data_structure, device, target_names, plot_path = None):
    model.eval()
    embeddings, class_preds, class_true = predict_with_model(model, dataloader, device, data_structure, sort_by_indices=True)

    # Calculate metrics
    report = classification_report(class_true, class_preds, target_names=target_names, output_dict=True)
    accuracy = report['accuracy']
    precision = {label: metrics['precision'] for label, metrics in report.items() if label in target_names}
    recall = {label: metrics['recall'] for label, metrics in report.items() if label in target_names}
    f1 = {label: metrics['f1-score'] for label, metrics in report.items() if label in target_names}

    # Log metrics to wandb
    wandb.log({"combined/accuracy": accuracy})
    for label, value in precision.items():
        wandb.log({f"combined/precision_{label}": value})
    for label, value in recall.items():
        wandb.log({f"combined/recall_{label}": value})
    for label, value in f1.items():
        wandb.log({f"combined/f1_{label}": value})

    # Generate confusion matrix
    conf_matrix = confusion_matrix(class_true, class_preds)

    # Plot confusion matrix as heatmap
    plt.figure(figsize=(5, 3))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plotter = Plotter()
    plotter.wandb_log_plot(plt, "combined/confusion_matrix", plot_path)
    plt.show()
    plt.close()
    return accuracy, precision, recall, f1, conf_matrix, class_preds, embeddings
