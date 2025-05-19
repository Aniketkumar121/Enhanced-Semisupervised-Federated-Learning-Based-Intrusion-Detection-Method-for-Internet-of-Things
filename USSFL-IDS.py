import sys
sys.path.append("utils/")

import torch
import torch.optim as optim
from torch import nn
import numpy as np
import copy
from sklearn.metrics import f1_score, precision_score  # Added for metrics
from utils.model_utils import GetNbaIotModel
from utils.creator import CreateDataset, SSFL_IDS_Client, SSFL_IDS_Server
from utils.process_data_utils import GetFeatureFromOpenDataset
from utils.train_utils import TrainWithDataset, PredictWithDisUnknown, \
    TrainWithFeatureLabel, Predict, Metrics, DisUnknown, OneHot2Label, HardLabelVoteHard, HardLabel, GetDeviceClientCnt
from utils.helper import compute_arith_mean, compute_geo_mean, compute_grad_variance
from utils.helper import l2_between_dicts, apply_fishr_loss
from utils.visualization import plot_metrics, plot_conf_matrix, plot_class_distribution  # Added for visualization

def SSFL_IDS(conf, dev, clients, server, test_dataset, open_dataset):
    comm_cnt = conf["comm_cnt"]
    open_idx_set_cnt = conf["open_idx_set_cnt"]
    batchsize = conf["batchsize"]
    train_rounds = conf["train_rounds"]
    dis_rounds = conf["discri_rounds"]
    dist_rounds = conf["dist_rounds"]
    theta = conf["theta"]
    labels = conf["labels"]
    first_train_rounds = conf["first_train_rounds"]
    class_cat = conf["classify_model_out_len"] if conf["classify_model_out_len"] > 1 else 2
    dis_train_cnt = 10000
    start_idx = 0
    end_idx = start_idx + open_idx_set_cnt
    open_len = len(open_dataset)

    # FedILC parameters
    aggregation_method = "arith"
    penalty_weight_factor = 0.1
    penalty_anneal_iters = 50

    all_client_grads_history = []
    grads_variance_history = []
    fishr_gradients = []
    global_model = server.model
    global_optimizer = server.dist_opt

    flags = {
        "batch_size": batchsize,
        "learning_rate": 0.001,
        "epochs": train_rounds,
        "momentum": 0.9,
        "weight_decay": 1e-4,
        "device": str(dev),
        "model_type": "CNN",
        "loss_function": nn.CrossEntropyLoss(),
        "optimizer": optim.Adam,
        "data_augmentation": True,
        "shuffle": True,
        "dropout_rate": 0.5,
        "validation_split": 0.2,
        "seed": conf["seed"] if "seed" in conf else 42,
        "aggregation": aggregation_method
    }

    acc_list = []  # Added for accuracy plot
    f1_list = []   # Added for F1 plot
    precision_list = []  # Added for precision plot

    for e in range(comm_cnt):
        sure_unknown_none = set()
        all_client_hard_label = []
        open_feature, open_label = GetFeatureFromOpenDataset(open_dataset, start_idx, end_idx)
        global_logits = torch.zeros(min(open_idx_set_cnt, open_len), len(labels))
        client_cnt = len(clients)
        participate = 0

        print("Round {} Stage I".format(e+1))
        all_client_grads_history = []
        grads_variance_history = []

        for c_idx in range(client_cnt):
            print("Client {} Training...".format(c_idx+1))
            cur_client = clients[c_idx]
            cur_train_rounds = train_rounds if e != 0 else first_train_rounds

            if len(cur_client.classify_dataset) == 0:
                continue

            client_grad_history = []
            for train_r in range(cur_train_rounds):
                avg_loss, batch_grads = TrainWithDataset(
                    dev, cur_client.classify_dataset, batchsize, 
                    cur_client.classify_model, cur_client.classify_opt, 
                    cur_client.hard_label_loss_func, flags, aggregation_method
                )
                client_grad_history.extend(batch_grads)

            if client_grad_history:
                all_client_grads_history.append(client_grad_history)
                client_grad_variances = {}
                for param_idx, param in enumerate(cur_client.classify_model.parameters()):
                    if param.grad is not None:
                        client_grad_variances[f'layer_{param_idx}'] = param.grad.detach().pow(2).mean()
                grads_variance_history.append(client_grad_variances)

            if sum(i > 0 for i in cur_client.each_class_cnt) != 1:
                participate += 1
                dis_train_feature, _ = GetFeatureFromOpenDataset(open_dataset, 0, dis_train_cnt)
                succ = DisUnknown(dev, cur_client, dis_rounds, batchsize, dis_train_feature, theta)
                if not succ:
                    sure_unknown_none.add(c_idx)

                cur_client_open_feature = open_feature.detach().clone()
                if c_idx not in sure_unknown_none:
                    local_logit = PredictWithDisUnknown(
                        dev, cur_client_open_feature,
                        cur_client.classify_model, cur_client.classify_model_out_len,
                        cur_client.discri_model, cur_client.discri_model_out_len,
                        len(labels)
                    )
                    hard_label = HardLabel(local_logit.detach().clone())
                    all_client_hard_label.append(hard_label)

        if all_client_grads_history:
            flat_grads = [batch_grads for client_grads in all_client_grads_history for batch_grads in client_grads if any(g is not None for g in batch_grads)]
            if flat_grads:
                if "arith" in aggregation_method:
                    for client in clients:
                        if sum(i > 0 for i in client.each_class_cnt) > 1:
                            compute_arith_mean(list(client.classify_model.parameters()), flat_grads)
                    compute_arith_mean(list(server.model.parameters()), flat_grads)
                elif "geo" in aggregation_method:
                    for client in clients:
                        if sum(i > 0 for i in client.each_class_cnt) > 1:
                            compute_geo_mean(list(client.classify_model.parameters()), flat_grads)
                    compute_geo_mean(list(server.model.parameters()), flat_grads)

        if "fishr" in aggregation_method and grads_variance_history:
            fishr_loss = apply_fishr_loss(global_model, grads_variance_history, penalty_weight_factor, e, penalty_anneal_iters)
            if fishr_loss > 0:
                global_optimizer.zero_grad()
                fishr_loss.backward()
                global_optimizer.step()

        if all_client_hard_label:
            global_logits = HardLabelVoteHard(all_client_hard_label, class_cat)

        print("Round {} Stage II".format(e+1))
        for c_idx in range(len(clients)):
            cur_client = clients[c_idx]
            print("Client {} Distillation Training...".format(c_idx+1))
            for r in range(dist_rounds):
                cur_global_logits = global_logits.detach().clone()
                cur_client_open_feature = open_feature.detach().clone()
                if cur_client.classify_model_out_len != 1:
                    TrainWithFeatureLabel(dev, cur_client_open_feature, cur_global_logits, batchsize,
                                          cur_client.classify_model, cur_client.classify_opt,
                                          cur_client.hard_label_loss_func, flags)
                else:
                    cur_global_logits = OneHot2Label(cur_global_logits)
                    TrainWithFeatureLabel(dev, cur_client_open_feature, cur_global_logits, batchsize,
                                          cur_client.classify_model, cur_client.classify_opt,
                                          cur_client.hard_label_loss_func, flags)
        print()

        print("Server Training...")
        for dist_i in range(dist_rounds):
            cur_global_logits = global_logits.detach().clone()
            server_open_feature = open_feature.detach().clone()
            if server.model_out_len != 1:
                TrainWithFeatureLabel(dev, server_open_feature, cur_global_logits, batchsize,
                                     server.model, server.dist_opt, server.hard_label_loss_func, flags)
            else:
                cur_global_logits = OneHot2Label(cur_global_logits)
                TrainWithFeatureLabel(dev, server_open_feature, cur_global_logits, batchsize,
                                     server.model, server.dist_opt, server.hard_label_loss_func, flags)

        test_feature, test_label = test_dataset[:]
        pred_label = Predict(dev, test_feature, server.model, server.model_out_len)
        correct_num, test_acc = Metrics(test_label, pred_label)
        print("Round {} Test Acc = {} ".format(e+1, test_acc))
        print()

        # Compute metrics and collect for final plots - Added section
        f1 = f1_score(test_label.cpu(), pred_label.cpu(), average='macro')
        precision = precision_score(test_label.cpu(), pred_label.cpu(), average='macro')
        acc_list.append(test_acc)
        f1_list.append(f1)
        precision_list.append(precision)

        # Plot only after final round - Added
        if e == comm_cnt - 1:
            plot_metrics(acc_list, f1_list, precision_list)
            plot_conf_matrix(test_label.cpu(), pred_label.cpu(), scenario="Scenario 1")
            plot_class_distribution(clients, scenario="Scenario 1")


def SSFL_IDS_NBaIoT():
    # Configurations
    configs = {
        "comm_cnt": 100,
        "device_client_cnt": 11,
        "private_percent": 0.9,
        "batchsize": 100,
        "iid": False,
        "need_dist": True,
        "open_percent": 0.1,
        "label_lr": 0.0001,
        "dist_lr": 0.0001,
        "discri_lr": 0.0001,
        "train_rounds": 3,
        "discri_rounds": 3,
        "dist_rounds": 10,
        "first_train_rounds": 3,
        "open_idx_set_cnt": 10000,
        "discri_cnt": 10000,
        "dist_T": 0.1,
        "need_SA": False,
        "test_batch_size": 256,
        "label_start_idx": 115,
        "test_round": 1,
        "data_average": True,
        "labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "clien_need_dist_opt": False,
        "discri_model_out_len": 1,
        "classify_model_out_len": 11,
        "sample_cnt": 1000,
        "random": True,
        "vote": True,
        "seed": 7,
        "load_data_from_pickle": False,
        "soft_label": False,
        "num_after_float": 4,
        "theta": -1,
        "split": "dile",
        "alpha_of_dile": 0.1,
    }

    if configs["seed"] is not None: 
        np.random.seed(configs["seed"])
    
    # Setup the devices and models
    device_names = [
        "Danmini_Doorbell/", "Ecobee_Thermostat/", "Philips_B120N10_Baby_Monitor/",
        "Provision_PT_737E_Security_Camera/", "Provision_PT_838_Security_Camera/", "SimpleHome_XCS7_1002_WHT_Security_Camera/",
        "SimpleHome_XCS7_1003_WHT_Security_Camera/","Ennio_Doorbell/", "Samsung_SNH_1011_N_Webcam/",
    ]

    device_cnt = len(device_names)
    clients = []
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    client_idx = 0
    test_dataset, private_dataset, open_dataset = CreateDataset(configs, "NBaIoT")

    for d_idx in range(device_cnt):
        cur_device_client_cnt = GetDeviceClientCnt(device_names[d_idx], configs["device_client_cnt"], configs["classify_model_out_len"])
        cur_device_private_datasets = private_dataset[d_idx]
        for i in range(cur_device_client_cnt):
            classify_model_out_len = configs["classify_model_out_len"]
            classify_model = GetNbaIotModel(classify_model_out_len)
            discri_model_out_len = configs["discri_model_out_len"]
            discri_model = GetNbaIotModel(discri_model_out_len)
            client = SSFL_IDS_Client(client_idx, cur_device_private_datasets[i], classify_model, classify_model_out_len,
                                            configs["label_lr"], discri_model, discri_model_out_len, configs["discri_lr"])
            clients.append(client)
            client_idx += 1

    server_model = GetNbaIotModel(configs["classify_model_out_len"])
    server = SSFL_IDS_Server(server_model, configs["classify_model_out_len"], clients, configs["dist_lr"])
    SSFL_IDS(configs, dev, clients, server, test_dataset, open_dataset)

if __name__ == "__main__":
    SSFL_IDS_NBaIoT()