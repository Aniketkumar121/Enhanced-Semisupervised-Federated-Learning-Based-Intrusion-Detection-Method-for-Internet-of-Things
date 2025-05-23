import torch
import numpy as np
from utils.process_data_utils import ShuffleDataset
from utils.helper import compute_arith_mean, compute_geo_mean, compute_grad_variance  # Importing aggregation functions
from utils.helper import l2_between_dicts  # Import L2 loss function for gradient comparison
from utils.helper import SSFL_IDS_CELoss  # Make sure SSFL_IDS_CELoss is in helper.py


def TrainWithFeatureLabel(dev, feature, label, batchsize, model, opt, loss_func, flags=None, algorithm="arith"):
    dataset = torch.utils.data.TensorDataset(feature, label)
    avg_loss, grads = TrainWithDataset(dev, dataset, batchsize, model, opt, loss_func, flags, algorithm)
    return avg_loss, grads

def TrainWithDataset(dev, dataset, batchsize, model, opt, loss_func, flags=None, algorithm="arith"):
    """Train model with dataset and collect gradients for aggregation"""
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    model = model.to(dev)
    model.train()
    
    total_loss = 0
    batch_count = 0
    model_grads = []
    
    for batch_idx, (batch_feature, batch_label) in enumerate(data_loader):
        opt.zero_grad()
        batch_feature = batch_feature.to(dev)
        batch_label = batch_label.to(dev)
        
        preds = model(batch_feature)
        loss = loss_func(preds, batch_label)
        loss.backward()
        
        # Store gradients for this batch before optimization
        current_grads = [p.grad.clone() if p.grad is not None else None for p in model.parameters()]
        model_grads.append(current_grads)
        
        opt.step()
        total_loss += loss.item() * batch_label.size(0)
        batch_count += 1
    
    avg_loss = total_loss / len(dataset) if len(dataset) > 0 else 0
    
    return avg_loss, model_grads

def EvalWithFeatureLabel(dev, feature, label, batchsize, model, loss_func):
    dataset = torch.utils.data.TensorDataset(feature, label)
    avg_loss = EvalWithDataset(dev, dataset, batchsize, model, loss_func)
    return avg_loss

def EvalWithDataset(dev, dataset, batchsize, model, loss_func):
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batchsize, shuffle=True)
    total_loss = 0
    with torch.no_grad():
        for batch_idx, (feature, label) in enumerate(test_loader):
            feature = feature.to(dev)
            label = label.to(dev)
            out = model(feature)
            loss = loss_func(out, label)
            total_loss += (loss.item() * label.size(0))
    test_loss = total_loss / (len(dataset))
    return test_loss

def Predict(dev, feature, model, model_out_len):
    with torch.no_grad():
        model = model.to(dev)
        feature = feature.to(dev)
        logits = model(feature)
        pred_label = Logits2PredLabel(logits, model_out_len)
        return pred_label

def Logits2PredLabel(logits, model_out_len):
    "pred -> hard label"
    with torch.no_grad():
        if model_out_len == 1:
            prediction = torch.round(torch.sigmoid(logits))
        else:
            _, prediction = torch.max(logits, 1)
    return prediction

def Predict2SoftLabel(dev, feature, model, model_out_len):
    with torch.no_grad():
        model = model.to(dev)
        feature = feature.to(dev)
        logits = model(feature)
        logits = Logits2Soft(logits, model_out_len)
        return logits

def Logits2Soft(logits, model_out_len):
    sigmoid = torch.nn.Sigmoid()
    softmax = torch.nn.Softmax(1)
    with torch.no_grad():
        if model_out_len == 1:
            logits = sigmoid(logits)
            soft_max_logits = torch.zeros(len(logits), 2)
            for i in range(len(logits)):
                soft_max_logits[i] = torch.tensor([1 - logits[i].item(), logits[i].item()])
            logits = soft_max_logits
        else:
            logits = softmax(logits)
        return logits

def Metrics(true_label, pred_label):
    print("Evaluation...")
    all_prediction = pred_label.cpu()
    all_label = true_label.cpu()
    correct_num = (all_label == all_prediction).sum().item()
    test_acc = correct_num / (len(true_label))
    return correct_num, test_acc

def PredictWithDisUnknown(dev, open_feature, 
                        classify_model, classify_model_len_out_tensor,
                        discri_model, discri_model_len_out_tensor,
                        class_cat):

    discri_model = discri_model.to(dev)
    classify_model = classify_model.to(dev)
    average_tensor = [1.0 / class_cat] * class_cat
    average_tensor = torch.tensor(average_tensor)

    with torch.no_grad():
        open_feature = open_feature.to(dev)
        wait_to_dis_label = classify_model(open_feature)
        wait_to_dis_label = Logits2Soft(wait_to_dis_label, classify_model_len_out_tensor)
        dis_label = discri_model(open_feature)
        dis_label = Logits2PredLabel(dis_label, discri_model_len_out_tensor)
        for i in range(len(dis_label)):
            if dis_label[i].item() == 1:
                wait_to_dis_label[i] = average_tensor.clone()
        return wait_to_dis_label

def HardLabel(soft_label):
    sample_cnt = len(soft_label)
    class_cat = len(soft_label[0])
    boundary = 1 / class_cat
    hard_label = [0] * sample_cnt
    for i in range(sample_cnt):
        cur_soft_label = soft_label[i]
        pred_label, pred_pro = torch.argmax(cur_soft_label), torch.max(cur_soft_label)
        hard_label[i] = pred_label.item() if pred_pro > boundary else class_cat
    return hard_label


def PredictFilter(dev, open_feature, classify_model, classify_model_len_out_tensor, class_cat, theta):
    print()
    print("in predict filter")
    print("theta = {}".format(theta))
    classify_model = classify_model.to(dev)
    average_tensor = [1.0 / class_cat ] * class_cat
    average_tensor = torch.tensor(average_tensor)

    with torch.no_grad():
        open_feature = open_feature.to(dev)
        wait_to_dis_label = classify_model(open_feature)
        wait_to_dis_label = Logits2Soft(wait_to_dis_label,classify_model_len_out_tensor)
        if theta < 0:
            max_num, pred_label = torch.max(wait_to_dis_label,1)
            theta = max_num.median()
        for i in range(len(wait_to_dis_label)):
            pred_label, pred_pro = torch.argmax(wait_to_dis_label[i]), torch.max(wait_to_dis_label[i])
            if pred_pro < theta:
                wait_to_dis_label[i] = average_tensor.clone()
        return wait_to_dis_label  # Fixed typo here from wait_to_dis_labe to wait_to_dis_label
    

def HardLabelVoteHard(all_client_hard_label, class_cat):
    client_cnt = len(all_client_hard_label)
    sample_cnt = len(all_client_hard_label[0])
    pred_labels = []
    label_votes_none_cnt = 0
    for i in range(sample_cnt):
        label_votes = [0] * class_cat
        for j in range(client_cnt):
            cur_client_cur_sample_hard_label = all_client_hard_label[j][i]
            pred_label = cur_client_cur_sample_hard_label
            if pred_label != class_cat:
                label_votes[pred_label] += 1
        if (len(label_votes) == 0):
            label_votes_none_cnt += 1
        max_vote_nums = max(label_votes)
        max_vote_idx = label_votes.index(max_vote_nums)
        pred_labels.append(max_vote_idx)
    pred_labels = torch.tensor(pred_labels)
    return pred_labels

def HardLabelVoteOneHot(all_client_hard_label, class_cat):
    client_cnt = len(all_client_hard_label)
    sample_cnt = len(all_client_hard_label[0])
    pred_labels = []
    all_vote_tensor = []
    label_votes_none_cnt = 0
    for i in range(sample_cnt):
        label_votes = [0] * class_cat
        for j in range(client_cnt):
            cur_client_cur_sample_hard_label = all_client_hard_label[j][i]
            pred_label = cur_client_cur_sample_hard_label
            if pred_label != class_cat:
                label_votes[pred_label] += 1
        if (len(label_votes) == 0):
            label_votes_none_cnt += 1
        max_vote_nums = max(label_votes)
        max_vote_idx = label_votes.index(max_vote_nums)
        pred_labels.append(max_vote_idx)
    all_one_hot_tensor = []
    for i in range(len(pred_labels)):
        cur_label = [0.0] * class_cat
        cur_label[pred_labels[i]] = 1.0
        all_one_hot_tensor.append(cur_label)
    all_vote_tensor = torch.tensor(all_one_hot_tensor)
    print("len of pred = {}".format(len(pred_labels)))
    print()
    return all_vote_tensor

def OneHot2Label(one_hot_vectors):
    _,labels = torch.max(one_hot_vectors,1)
    labels = labels.double()
    return labels

def GetDeviceClientCnt(device_name, client_cnt, classify_model_out_len):
    if classify_model_out_len == 1:
        return 4
    else:
        if ((device_name == "Ennio_Doorbell/" or device_name == "Samsung_SNH_1011_N_Webcam/")):
            return int(client_cnt / 2) + 1

        else:
            return client_cnt

def GetDeviceClassCat(device_name,classify_model_out_len):
    if classify_model_out_len == 1:
        return 2
    if ((device_name == "Ennio_Doorbell/" or device_name == "Samsung_SNH_1011_N_Webcam/")):
        return 6
    else :
        return 11

def reshape_sample(feature):
    feature = np.reshape(feature, (-1, 23, 5))
    return feature

def PredUnknown(dev, feature, model, theta, model_out_len):
    sure_unknown = []
    wait_to_dis = []
    soft_max = torch.nn.Softmax(1)
    sigmoid = torch.nn.Sigmoid()
    model = model.to(dev)
    feature = feature.to(dev)
    with torch.no_grad():
        out = model(feature)
        if model_out_len == 1:
            out = sigmoid(out)
            soft_max_out = torch.zeros(len(out),2)
            for i in range(len(out)):
                soft_max_out[i] = torch.tensor([1-out[i].item(),out[i].item()])
            out = soft_max_out
        else:
            out = soft_max(out)
        max_num, pred_label = torch.max(out, 1)
        if theta < 0 :
            theta = max_num.median()
        for i in range(len(max_num)):
            if max_num[i] < theta:
                sure_unknown.append(feature[i])
            else:
                wait_to_dis.append(feature[i])
    if len(sure_unknown) == 0:
        return None 
    sure_unknown = torch.stack(sure_unknown)

    return sure_unknown

def LabelFeature(feature,label):
    labels = [label] * len(feature)
    labels = torch.tensor(labels)
    return feature, labels

def DisUnknown(dev, client, dis_rounds, batchsize, dis_train_feature, theta):
    dis_train_feature = dis_train_feature.detach().clone()
    sure_unknown_feature = PredUnknown(dev, dis_train_feature, client.classify_model, theta,
                                       client.classify_model_out_len)
    if sure_unknown_feature is None:
        return False

    unknown_label_num = -1
    known_label_num = -1
    if client.discri_model_out_len == 1:
        unknown_label_num = 1.0
        known_label_num = 0.0
    else:
        unknown_label_num = 1
        known_label_num = 0

    sure_unknown_feature,sure_unknown_label = LabelFeature(sure_unknown_feature,unknown_label_num)
    sure_known_feature, _ = client.classify_dataset[:]
    sure_known_feature = sure_known_feature.detach().clone()
    
    sure_known_feature,sure_known_label = LabelFeature(sure_known_feature, known_label_num)

    sure_unknown_feature = sure_unknown_feature.to(dev)
    sure_known_feature = sure_known_feature.to(dev)

    dis_feature = torch.cat((sure_known_feature,sure_unknown_feature),0)

    sure_known_label = sure_known_label.to(dev)
    sure_unknown_label = sure_unknown_label.to(dev)

    dis_label = torch.cat((sure_known_label,sure_unknown_label),0)
    cpu_dev = torch.device("cpu")
    dis_feature = dis_feature.to(cpu_dev)
    dis_label = dis_label.to(cpu_dev)
    dis_dataset = torch.utils.data.TensorDataset(dis_feature, dis_label)
    dis_dataset = ShuffleDataset(dis_dataset)

    # Add default flags for backward compatibility
    default_flags = {
        "batch_size": batchsize,
        "learning_rate": 0.001,
        "device": str(dev)
    }
    
    for r in range(dis_rounds):
        # Add missing arguments to TrainWithDataset
        avg_loss, _ = TrainWithDataset(dev, dis_dataset, batchsize, 
                                      client.discri_model, client.discri_opt, 
                                      client.discri_loss_func, default_flags, "arith")

    return True


def apply_fishr_loss(model, grads_variance_history, penalty_weight_factor, round_idx, penalty_anneal_iters):
    """
    Apply Fishr loss to penalize gradient variance differences.
    
    Args:
        model: The model whose gradients are being analyzed
        grads_variance_history: List of gradient variance dictionaries from clients
        penalty_weight: Weight factor for the penalty term
        round_idx: Current round/iteration number
        anneal_iters: Number of iterations for penalty annealing
    Returns:
        Fishr loss value
    """
    if not grads_variance_history:
        return 0.0
    
    dict_grad_statistics_averaged = {}
    first_dict_grad_statistics = grads_variance_history[0]

    for name in first_dict_grad_statistics:
        grads_list = [dict_grad_statistics[name] for dict_grad_statistics in grads_variance_history if name in dict_grad_statistics]
        if grads_list:
            dict_grad_statistics_averaged[name] = torch.stack(grads_list, dim=0).mean(dim=0)

    fishr_loss = 0.0
    for dict_grad_statistics in grads_variance_history:
        fishr_loss += l2_between_dicts(dict_grad_statistics, dict_grad_statistics_averaged)

    penalty_weight = penalty_weight_factor if round_idx >= penalty_anneal_iters else 1.0
    fishr_loss *= penalty_weight
    return fishr_loss