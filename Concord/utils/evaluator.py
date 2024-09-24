
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from .. import logger

def evaluate_scib(
    adata_pre,
    adata_post, 
    embedding_obsm_keys, 
    batch_key="batch", 
    label_key="cell_type",
    isolated_labels_asw_=True,
    silhouette_=True,
    hvg_score_=False,
    graph_conn_=True,
    pcr_=True,
    isolated_labels_f1_=True,
    trajectory_=False,
    nmi_=True,
    ari_=True,
    cell_cycle_=False,
    kBET_=False,
    ilisi_=False,
    clisi_=False,
    organism="human"
):
    try:
        import scib
    except ImportError:
        logger.error("scib is not installed. Please install scib to use this function.")
        return None
    results = []
    for embed in embedding_obsm_keys:
        eval_result = scib.metrics.metrics(
            adata=adata_pre,
            adata_int=adata_post,
            batch_key=batch_key,
            label_key=label_key,
            embed=embed,
            isolated_labels_asw_=isolated_labels_asw_,
            silhouette_=silhouette_,
            hvg_score_=hvg_score_,
            graph_conn_=graph_conn_,
            pcr_=pcr_,
            isolated_labels_f1_=isolated_labels_f1_,
            trajectory_=trajectory_,
            nmi_=nmi_,
            ari_=ari_,
            cell_cycle_=cell_cycle_,
            kBET_=kBET_,
            ilisi_=ilisi_,
            clisi_=clisi_,
            organism=organism
        )
        
        # Rename the column to the embedding method
        eval_result.columns = [embed]
        logger.info(f"Evaluated {embed}.")
        # Append the DataFrame to the results list
        results.append(eval_result)
    
    # Concatenate all DataFrames in the results list
    concatenated_results = pd.concat(results, axis=1)
    
    return concatenated_results


def log_classification(epoch, phase, preds, labels, logger):
    from sklearn.metrics import classification_report
    # Calculate metrics
    unique_classes = np.sort(np.unique(labels))
    report = classification_report(labels, preds, target_names=unique_classes, output_dict=True)
    accuracy = report['accuracy']
    precision = {label: metrics['precision'] for label, metrics in report.items() if label in unique_classes}
    recall = {label: metrics['recall'] for label, metrics in report.items() if label in unique_classes}
    f1 = {label: metrics['f1-score'] for label, metrics in report.items() if label in unique_classes}


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