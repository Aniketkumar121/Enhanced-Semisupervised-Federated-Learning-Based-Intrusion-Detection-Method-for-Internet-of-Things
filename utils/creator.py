import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from utils.process_data_utils import GetDataset, SplitPrivateOpen, DilSplitPrivate, SplitPrivate, GetAllFeatureLabel, ShuffleDataset
from utils.train_utils import GetDeviceClassCat, GetDeviceClientCnt, reshape_sample
from utils.helper import l2_between_dicts  # Fixed import path
from utils.helper import SSFL_IDS_CELoss  # Fixed import path


class SSFL_IDS_Client():
    def __init__(self, idx, 
                classify_dataset:torch.utils.data.Dataset, classify_model:nn.Module, classify_model_out_len, 
                classify_lr:float,
                discri_model, discri_model_out_len, discri_lr):
        # Initialize the classification model and its settings
        self.classify_model = classify_model
        self.classify_dataset = classify_dataset
        self.class_cat = classify_model_out_len if classify_model_out_len > 1 else 2
        self.each_class_cnt = [0] * self.class_cat
        for _, label in self.classify_dataset:
            self.each_class_cnt[label.item()] += 1
        self.classify_lr = classify_lr
        self.c_idx = idx
        self.classify_opt = optim.Adam(self.classify_model.parameters(), lr=self.classify_lr)
        
        # Initialize discriminator model
        self.discri_model = discri_model
        self.discri_lr = discri_lr
        self.discri_opt = optim.Adam(self.discri_model.parameters(), lr=self.discri_lr)
        self.discri_model_out_len = discri_model_out_len
        
        # Loss functions based on output length of the model
        if discri_model_out_len == 1:
            self.discri_loss_func = nn.BCEWithLogitsLoss()
        else:
            self.discri_loss_func = nn.CrossEntropyLoss()
        
        self.classify_model_out_len = classify_model_out_len
        if classify_model_out_len == 1:
            self.hard_label_loss_func = nn.BCEWithLogitsLoss()
            self.feature, self.label = self.classify_dataset[:]
            self.label = self.label.double()
            self.classify_dataset = torch.utils.data.TensorDataset(self.feature, self.label)
        else:
            self.hard_label_loss_func = nn.CrossEntropyLoss()
        self.soft_label_loss_func = SSFL_IDS_CELoss()

        # Initialize a list to hold gradient history for Fishr
        self.gradients_history = []

    def collect_gradients(self):
        """Collect gradients after training step for Fishr loss."""
        model_grads = [p.grad.clone() for p in self.classify_model.parameters()]
        self.gradients_history.append(model_grads)


class SSFL_IDS_Server():
    def __init__(self, model, model_out_len, clients, dist_lr):
        self.model = model
        self.clients = clients
        self.client_cnt = len(clients)
        self.model_out_len = model_out_len
        self.dist_lr = dist_lr
        self.dist_opt = optim.Adam(self.model.parameters(), lr=self.dist_lr)
        self.soft_label_loss_func = SSFL_IDS_CELoss()
        
        if model_out_len != 1:
            self.hard_label_loss_func = nn.CrossEntropyLoss()
        else:
            self.hard_label_loss_func = nn.BCEWithLogitsLoss()

    def apply_fishr_loss(self, grads_variance_history, penalty_weight_factor, round_idx, penalty_anneal_iters):
        dict_grad_statistics_averaged = {}
        first_dict_grad_statistics = grads_variance_history[0]

        for name in first_dict_grad_statistics:
            grads_list = [dict_grad_statistics[name] for dict_grad_statistics in grads_variance_history]
            dict_grad_statistics_averaged[name] = torch.stack(grads_list, dim=0).mean(dim=0)

        fishr_loss = 0
        for dict_grad_statistics in grads_variance_history:
            fishr_loss += l2_between_dicts(dict_grad_statistics, dict_grad_statistics_averaged)

        penalty_weight = penalty_weight_factor if round_idx >= penalty_anneal_iters else 1.0
        fishr_loss *= penalty_weight
        return fishr_loss


def CreateDataset(configs, dataset_name="NBaIoT"):
    if dataset_name == "NBaIoT":
        return create_NBaIoT(configs)


def create_NBaIoT(configs):
    prefix = "data/nba_iot_1000/"

    device_names = [
        "Danmini_Doorbell/", "Ecobee_Thermostat/", "Philips_B120N10_Baby_Monitor/",
        "Provision_PT_737E_Security_Camera/", "Provision_PT_838_Security_Camera/", "SimpleHome_XCS7_1002_WHT_Security_Camera/",
        "SimpleHome_XCS7_1003_WHT_Security_Camera/", "Ennio_Doorbell/", "Samsung_SNH_1011_N_Webcam/",
    ]
    attack_names = [
        "benign", "g_combo", "g_junk", "g_scan", "g_tcp", "g_udp",
        "m_ack", "m_scan", "m_syn", "m_udp", "m_udpplain"
    ]
    if configs["classify_model_out_len"] == 1:
        attack_names = ["benign", "attack"]

    all_device_train_feature = None
    all_device_train_label = None
    all_device_open_feature = None
    all_device_open_label = None
    all_device_private_feature = []
    all_device_private_label = []
    all_device_test_feature = None
    all_device_test_label = None
    device_cnt = len(device_names)

    if configs["load_data_from_pickle"] == False:
        for d_idx in range(device_cnt):
            cur_device_class_cat = GetDeviceClassCat(device_names[d_idx], configs["classify_model_out_len"])
            train_filenames = []
            test_filenames = []
            for i in range(len(attack_names)):
                if i < cur_device_class_cat:
                    train_filename = prefix + device_names[d_idx] + attack_names[i] + "_train.csv"
                    test_filename = prefix + device_names[d_idx] + attack_names[i] + "_test.csv"
                    train_filenames.append(train_filename)
                    test_filenames.append(test_filename)
            train_feature, train_label = GetAllFeatureLabel(train_filenames, configs["label_start_idx"])
            private_feature, private_label, open_feature, open_label = SplitPrivateOpen(train_feature, train_label, configs["private_percent"], configs["open_percent"], cur_device_class_cat, False)
            all_device_private_feature.append(private_feature)
            all_device_private_label.append(private_label)
            if all_device_open_feature is None:
                all_device_open_feature = open_feature
                all_device_open_label = open_label
            else:
                all_device_open_feature = np.concatenate((all_device_open_feature, open_feature), axis=0)
                all_device_open_label = np.concatenate((all_device_open_label, open_label), axis=0)

            if all_device_train_feature is None:
                all_device_train_feature = train_feature
                all_device_train_label = train_label

            test_feature, test_label = GetAllFeatureLabel(test_filenames, configs["label_start_idx"])
            if all_device_test_feature is None:
                all_device_test_feature = test_feature
                all_device_test_label = test_label
            else:
                all_device_test_feature = np.concatenate((all_device_test_feature, test_feature), axis=0)
                all_device_test_label = np.concatenate((all_device_test_label, test_label), axis=0)

    scaler = MinMaxScaler()
    scaler.fit(all_device_open_feature)
    all_device_open_feature = scaler.transform(all_device_open_feature)
    all_device_open_feature = reshape_sample(all_device_open_feature)
    open_dataset = GetDataset(all_device_open_feature, all_device_open_label)
    open_dataset = ShuffleDataset(open_dataset)
    all_device_test_feature = scaler.transform(all_device_test_feature)
    all_device_test_feature = reshape_sample(all_device_test_feature)
    test_dataset = GetDataset(all_device_test_feature, all_device_test_label)
    
    private_datasets = []
    for d_idx in range(device_cnt):
        cur_device_class_cat = GetDeviceClassCat(device_names[d_idx], configs["classify_model_out_len"])
        cur_device_client_cnt = GetDeviceClientCnt(device_names[d_idx], configs["device_client_cnt"], configs["classify_model_out_len"])
        cur_device_private_feature = all_device_private_feature[d_idx]
        cur_device_private_label = all_device_private_label[d_idx]
        cur_device_private_feature = scaler.transform(cur_device_private_feature)
        cur_device_private_feature = reshape_sample(cur_device_private_feature)

        if configs["iid"] == True:
            cur_device_private_datasets = SplitPrivate(cur_device_private_feature, cur_device_private_label,
                                                       cur_device_client_cnt, cur_device_class_cat, configs["iid"],
                                                       configs["data_average"])
            private_datasets.append(cur_device_private_datasets)
        elif configs["split"] == "dile":
            cur_device_private_datasets = DilSplitPrivate(cur_device_private_feature, cur_device_private_label,
                                                          cur_device_client_cnt, cur_device_class_cat,
                                                          configs["alpha_of_dile"], configs["seed"])
            private_datasets.append(cur_device_private_datasets)
        elif configs["split"] == "equally":
            cur_device_private_datasets = SplitPrivate(cur_device_private_feature, cur_device_private_label,
                                                       cur_device_client_cnt, cur_device_class_cat, configs["iid"],
                                                       configs["data_average"])
            private_datasets.append(cur_device_private_datasets)

    return test_dataset, private_datasets, open_dataset