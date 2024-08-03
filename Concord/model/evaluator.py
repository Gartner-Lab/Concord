


from typing import Dict, Optional
import numpy as np
from .. import logger
import scib
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from archive.predict_functions import predict_with_model
from ..utils.other_util import wandb_log_plot

# Function made by scGPT: https://github.com/bowang-lab/scGPT/blob/main/scgpt/utils/util.py
# Wrapper for all scib metrics, we leave out some metrics like hvg_score, cell_cyvle,
# trajectory_conservation, because we only evaluate the latent embeddings here and
# these metrics are evaluating the reconstructed gene expressions or pseudotimes.
def eval_scib_metrics(
    adata,
    batch_key: str = "str_batch",
    label_key: str = "cell_type",
    notes: Optional[str] = None,
) -> Dict:

    results = scib.metrics.metrics(
        adata,
        adata_int=adata,
        batch_key=batch_key,
        label_key=label_key,
        embed="encoded",
        isolated_labels_asw_=False,
        silhouette_=True,
        hvg_score_=False,
        graph_conn_=True,
        pcr_=True,
        isolated_labels_f1_=False,
        trajectory_=False,
        nmi_=True,  # use the clustering, bias to the best matching
        ari_=True,  # use the clustering, bias to the best matching
        cell_cycle_=False,
        kBET_=False,  # kBET return nan sometimes, need to examine
        ilisi_=False,
        clisi_=False,
    )

    if notes is not None:
        logger.info(f"{notes}")

    logger.info(f"{results}")

    result_dict = results[0].to_dict()
    logger.info(
        "Biological Conservation Metrics: \n"
        f"ASW (cell-type): {result_dict['ASW_label']:.4f}, graph cLISI: {result_dict['cLISI']:.4f}, "
        f"isolated label silhouette: {result_dict['isolated_label_silhouette']:.4f}, \n"
        "Batch Effect Removal Metrics: \n"
        f"PCR_batch: {result_dict['PCR_batch']:.4f}, ASW (batch): {result_dict['ASW_label/batch']:.4f}, "
        f"graph connectivity: {result_dict['graph_conn']:.4f}, graph iLISI: {result_dict['iLISI']:.4f}"
    )

    result_dict["avg_bio"] = np.mean(
        [
            result_dict["NMI_cluster/label"],
            result_dict["ARI_cluster/label"],
            result_dict["ASW_label"],
        ]
    )

    # remove nan value in result_dict
    result_dict = {k: v for k, v in result_dict.items() if not np.isnan(v)}

    return result_dict






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

    wandb_log_plot(plt, "combined/confusion_matrix", plot_path)
    plt.show()
    plt.close()
    return accuracy, precision, recall, f1, conf_matrix, class_preds, embeddings
