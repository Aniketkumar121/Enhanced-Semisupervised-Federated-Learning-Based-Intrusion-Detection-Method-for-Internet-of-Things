import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_class_distribution(clients, scenario="Scenario 1"):
    client_ids = []
    class_ids = []
    sample_counts = []

    for client_idx, client in enumerate(clients):
        for class_id, count in enumerate(client.each_class_cnt):
            if count > 0:
                client_ids.append(client_idx)
                class_ids.append(class_id)
                sample_counts.append(count)

    plt.figure(figsize=(10, 6))
    plt.scatter(client_ids, class_ids, s=np.array(sample_counts)*3, alpha=0.6, cmap='viridis')
    plt.xlabel("Client ID")
    plt.ylabel("Class ID")
    plt.title(f"Sample Distribution - {scenario}")
    plt.grid(True)
    plt.show()

def plot_metrics(acc_list, f1_list, precision_list):
    rounds = list(range(1, len(acc_list)+1))
    plt.plot(rounds, acc_list, label='Accuracy')
    plt.plot(rounds, f1_list, label='F1 Score')
    plt.plot(rounds, precision_list, label='Precision')
    plt.xlabel("Rounds")
    plt.ylabel("Metric")
    plt.title("Model Performance")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_conf_matrix(y_true, y_pred, scenario="Scenario 1"):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(xticks_rotation=45)
    plt.title(f"Confusion Matrix - {scenario}")
    plt.show()

def compute_comm_overhead(grads_history):
    overhead = 0
    for batch_grads in grads_history:
        for grad in batch_grads:
            if grad is not None:
                overhead += grad.numel()
    return overhead

def plot_theta_accuracy(thetas, theta_accuracies):
    plt.plot(thetas, theta_accuracies)
    plt.xlabel('Confidence Threshold θc')
    plt.ylabel('Final Accuracy')
    plt.title('Impact of θc on Accuracy')
    plt.grid(True)
    plt.show()

def plot_label_strategy_comparison(label_strategies, accuracies, comm_costs):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(label_strategies, accuracies, 'g-', label='Accuracy')
    ax2.plot(label_strategies, comm_costs, 'b-', label='Comm Overhead')
    ax1.set_xlabel('Label Strategy')
    ax1.set_ylabel('Accuracy', color='g')
    ax2.set_ylabel('Comm. Overhead', color='b')
    plt.title("Label Strategy vs Accuracy & Comm Cost")
    plt.grid(True)
    plt.show()

def plot_comm_overhead_vs_accuracy(comm_overheads, top_accuracies, labels):
    plt.figure(figsize=(8,6))
    for i in range(len(comm_overheads)):
        plt.scatter(comm_overheads[i], top_accuracies[i], label=labels[i])
    plt.xlabel("Communication Overhead")
    plt.ylabel("Top Accuracy")
    plt.title("Comm Overhead vs Top Accuracy")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_multiple_accuracy_curves(acc_dict, scenario="Scenario 1"):
    """
    acc_dict: dictionary with key = label, value = list of accuracies per round
    """
    plt.figure(figsize=(10, 6))
    for label, acc_curve in acc_dict.items():
        plt.plot(range(1, len(acc_curve)+1), acc_curve, label=label)
    plt.xlabel("Rounds")
    plt.ylabel("Test Accuracy")
    plt.title(f"Test Accuracy Curves - {scenario}")
    plt.legend()
    plt.grid(True)
    plt.show()
