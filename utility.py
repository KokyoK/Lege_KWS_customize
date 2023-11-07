from argparse import Namespace

import numpy as np
import torch

torch.manual_seed(42)

from torch.optim import lr_scheduler
from losses import *

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
time_check = False
if not train_on_gpu:
    print('CUDA is not available.  Using CPU ...')
else:
    print('CUDA is available!  Using GPU ...')
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_check = True

torch.set_printoptions(precision=4)

e_count = 2


####### optimization constraints ##########
# latency_constraint = 0.6

########## optimization methods ##########
# opt_methods = ["heuristic","ippp","sa"]
# opt_method = opt_methods[2]


def train(model, loaders, num_epoch, args):
    """
    Trains TCResNet
    """
    tuning = False
    if tuning:
        if args.model_name == "resnet32":
            ratios = [0.5, 0.8, 1]
            model.load(args="", name="test/res32_3_e2e_ee_928_infer_87.080.pt")
        elif args.model_name == "tcresnet8":
            ratios = [0.83, 0.17]   #0.5
            # ratios = [0.5, 1] #0.7
            model.load(args="", name="test/tcres_93.017.pt")
        elif args.model_name == "mobilev2":
            ratios = [0.18, 1]
            model.load(args="", name="test/mobilev2_e2e_ee_664_infer_91.200.pt")

    ratios = [0.5, 1]



    # ratios = [0.5, 0.8,  1]  # random initialization
    # ratios = [0.18,1] # for mobile v2 0.5
    # Enable GPU training
    if train_on_gpu:
        model.cuda()
    [train_dataloader,valid_dataloader, test_dataloader] = loaders
    criterion = ConfCrLoss()    # ce
    name = ""
    # Training
    step_idx = 0
    # thresholds = [0.3, 0.3, 0.3]
    best_thresolds = [0] * e_count
    thresholds = np.array([1] * e_count)
    thresh_idxs = [0] * e_count
    # previous_valid_accuracy = [0] * e_count
    previous_infer_valid_acc = 0
    # model_names = ["EE1", "EE2", "FULL"]

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9, weight_decay=0)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=1e-3)

    for epoch in range(num_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        valid_kw_correct = [0] * e_count
        train_kw_correct = [0] * e_count
        ################################## Train ##############################################################
        model.train()
        for batch_idx, (audio_data, label_kw) in enumerate(train_dataloader):
        # for batch_idx, (audio_data, label_kw) in enumerate(valid_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                label_kw = label_kw.cuda()
            outs = model(x=audio_data)
            # [out0, out1, out2] = outs
            # out = outs[e_idx]
            # exit_outs = [0, 0, 0]
            # non_exit_outs = [0, 0, 0]
            # exit_labels = [0, 0, 0]
            # non_exit_labels = [0, 0, 0]
            exit_indices = [0] * e_count
            non_exit_indices = [0] * e_count
            # allocated_memory = torch.cuda.memory_allocated()
            # print(f"1: {allocated_memory} bytes")
            # calculate threshold and partition the batch
            batch_size = label_kw.shape[0]
            temp_thresholds = [0] * e_count
            optimizer.zero_grad()

            
            for i in range(e_count):
                if (i == 0):
                    outs_conti = [None, outs[i], outs[i + 1]]
                    ratio = [0, ratios[i]]
                elif (i == e_count - 1):
                    outs_conti = [outs[i - 1], outs[i], None]
                    ratio = [ratios[i - 1], 1]
                else:
                    outs_conti = [outs[i - 1], outs[i], outs[i + 1]]
                    ratio = [ratios[i - 1], ratios[i]]
                [temp_thresholds[i], thresh_idxs[i]], [exit_indices[i], non_exit_indices[i]] \
                    = partition_batch_idx(outs=outs, label=label_kw, thresh=torch.Tensor(thresholds), ratio=ratio, exit_index=i,
                                          type=0)

                loss = criterion(outs_conti, label_kw, exit_indices[i], non_exit_indices[i], epoch, i, args.a)

                if (i < (e_count - 1)):
                    loss.backward(retain_graph=True)
                else:
                    loss.backward()
                train_loss += loss.item() * audio_data.size(0)
            #
            # for j in range(len(temp_thresholds)):
            thresholds = np.array(temp_thresholds)
            # allocated_memory = torch.cuda.memory_allocated()
            # print(f"2: {allocated_memory} bytes")
            # allocated_memory = torch.cuda.memory_allocated()
            # print(f"3: {allocated_memory} bytes")

            optimizer.step()
            scheduler.step()

            ba = torch.zeros([e_count])
            b_hit = torch.zeros([e_count])
            for i in range(e_count):
                b_hit[i] = float(torch.sum(torch.argmax(outs[i].value, 1) == label_kw).item())
                ba[i] = b_hit[i] / float(audio_data.shape[0]) * 100
                train_kw_correct[i] += b_hit[i]

            if (batch_idx % 200 == 0):
                print(
                    "| Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | Learning Rate: {:.7f} ".format(
                        epoch, step_idx, loss,
                        optimizer.state_dict()['param_groups'][0]['lr']))
                for i in range(e_count):
                    print("#  | ACC: {:.2f}% \t| Exit Ratio {:.2f}% \t| Threshold {:.4f}".format(
                        ba[i], exit_indices[i].shape[0] / batch_size * 100, thresholds[i]))
                    # train_kw_correct += b_hit
            step_idx += 1
            # break

            # del exit_labels[i] if exit_labels[i]!= None
            # del non_exit_outs[i] if non_exit_outs[i]!= None

        ################################## Validation (1 epoch) ##############################################################
        model.eval()
        # model.mode = "eval"
        total_infer_time = 0
        hit_exit = torch.zeros([e_count])  #
        exit_count = torch.zeros([e_count])  # count of current exit samples
        hit_non_exit = torch.zeros([e_count])
        valid_acc_exit = torch.zeros([e_count])
        valid_acc_non_exit = torch.zeros([e_count])
        valid_exit_ratio = torch.zeros([e_count])


        for batch_idx, (audio_data, label_kw) in enumerate(valid_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                label_kw = label_kw.cuda()
            with torch.no_grad():
                outs = model(x=audio_data)
                exit_outs, exit_labels, exit_indices = infer_batch(outs, label_kw, thresholds)
            loss = 0
            b_exit_hit = []
            b_non_exit_hit = []
            
            for i in range(e_count):
                if (exit_outs[i].dim() == 2):
                    # print(exit_outs[i].shape)
                    b_exit_hit.append(float(torch.sum(torch.argmax(exit_outs[i], 1) == exit_labels[i]).item()))
                    hit_exit[i] += b_exit_hit[i]
                    exit_count[i] += exit_outs[i].shape[0]
                    loss += criterion.ce_loss(exit_outs[i], exit_labels[i])
                else:  # batch has output
                    b_exit_hit.append(torch.Tensor(0))

            # if (i == 0):
            #     outs_conti = [None, outs[i], outs[i + 1]]
            #     ratio = [0, ratios[i]]
            # elif (i == e_count - 1):
            #     outs_conti = [outs[i - 1], outs[i], None]
            #     ratio = [ratios[i - 1], 1]
            # else:
            #     outs_conti = [outs[i - 1], outs[i], outs[i + 1]]
            #     ratio = [ratios[i - 1], ratios[i]]

            valid_loss += loss.item() * audio_data.size(0)
            ba = torch.zeros([e_count])
            b_hit = torch.zeros([e_count])
            for i in range(e_count):
                b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
                ba[i] = b_hit[i] / float(audio_data.shape[0]) * 100
                valid_kw_correct[i] += b_hit[i]

            if (batch_idx % 100 == 0):
                print("| Epoch {} | Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.2f} {:.2f}|  ".format(
                    epoch, step_idx, loss, ba[0], ba[1]))

            # valid_kw_correct += b_hit
            step_idx += 1
            # break
            
        # print(exit_count.sum())
        # Loss statistics for current EPOCH
        train_loss = train_loss / len(train_dataloader.dataset)
        valid_loss = valid_loss / len(valid_dataloader.dataset)
        train_kw_accuracy = torch.zeros([e_count])
        valid_kw_accuracy = torch.zeros([e_count])

        # valid_acc_non_exit[e_idx] = -1 if(len(dev_dataloader.dataset) - exit_count[e_idx])==0 \
        #     else 100 * hit_non_exit[e_idx] / (len(dev_dataloader.dataset) - exit_count[e_idx])

        infer_valid_acc = torch.sum(hit_exit) / len(valid_dataloader.dataset)
        for i in range(e_count):
            train_kw_accuracy[i] = 100.0 * (train_kw_correct[i] / len(train_dataloader.dataset))
            valid_kw_accuracy[i] = 100.0 * (valid_kw_correct[i] / len(valid_dataloader.dataset))
            valid_acc_exit[i] = 100 * hit_exit[i] / exit_count[i]
            valid_exit_ratio[i] = exit_count[i] / len(valid_dataloader.dataset)
            valid_acc_non_exit[i] = -1 if (len(valid_dataloader.dataset) - exit_count[i]) == 0 \
                else 100 * hit_non_exit[i] / (len(valid_dataloader.dataset) - exit_count[i])

            # update ratios
            # if i == 0:
            #     ratios[i] = valid_exit_ratio[i].numpy()
            # elif i == (e_count - 1):
            #     ratios[i] = 1
            # else:
            #     ratios[i] = ratios[i - 1] + valid_exit_ratio[i].numpy()

        real_latency = (model.flops * valid_exit_ratio).sum().numpy()
        # print(output.shape)
        # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
        # print(f1_scores)
        print("===========================| EPOCH #{}  |===================================".format(epoch))
        print("  | TRAIN ACC: {}%\t|  TRAIN LOSS : {:.2f} |".format(
            train_kw_accuracy, train_loss))
        print("  | INFER ACC: {:.2f}%\t| ".format(infer_valid_acc * 100))
        print("  | Average Latency: {:.4f}\t| ".format(real_latency))
        for i in range(e_count):
            print(
                "            EXIT {} | VAL ACC :  {:.2f}%\t ｜ Exit Ratio: {:.4f}% | Thresholds: {:.4f} |ACC_exit: {:.2f}%\t| ACC_non_exit {:.2f}%\t | VAL LOSS : {:.2f}".format(
                    i, valid_kw_accuracy[i], valid_exit_ratio[i] * 100, thresholds[i], valid_acc_exit[i],
                    valid_acc_non_exit[i], valid_loss))
        # print("Validation path count:   ", path_count)
        # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
        print("===========================================================================")
        # for i in range(e_count):
        #     del exit_outs[i],exit_labels[i], non_exit_outs[i], non_exit_labels[i]

        ##################### 看要不要存
        save = 0
        # for i in range(e_count):
        if (infer_valid_acc > previous_infer_valid_acc):
            save = 1

        if (save == 1):
            previous_infer_valid_acc = infer_valid_acc
            best_thresolds = thresholds
            print("Saving current model...")
            arg_str = get_argstring(args)
            name = 'e2e_ee_{}_infer_{:.3f}.pt'.format(epoch, infer_valid_acc * 100)
            # model.save(args=arg_str)
            model.save(is_onnx=0, name=f"saved_model/{name}")

        # Update scheduler (for decaying learning rate)
        # scheduler.step()

    # Final test
    evaluate_ee_testset(model, test_dataloader, best_thresolds, args,name)


def partition_batch_idx(outs, label, thresh, ratio, exit_index, type=0):
    '''
    返回index
    type: 0-follow ratio, 1-follow threshold, 2-follow real infer
    '''
    #

    out = outs[exit_index]
    out = out.value
    par_batch_size = out.shape[0]
    # 按照ratio退出还是按照threshold
    max_prob_0, _ = out.max(dim=1)
    sorted_prob, sorted_indices = max_prob_0.sort(dim=0, descending=True)

    if type == 0:  # follow predefined ratio,
        # print(ratio)
        thresh = thresh[exit_index]
        if (ratio[0] == 0):
            exit_indices = sorted_indices[:round(ratio[1] * par_batch_size)]
        elif (ratio[1] == 1):
            exit_indices = sorted_indices[round(ratio[0] * par_batch_size):]
        else:
            exit_indices = sorted_indices[round(ratio[0] * par_batch_size):round(ratio[1] * par_batch_size)]
            # cur_thresh = out[exit_indices[-1]][label_kw[exit_indices[-1]]]
        cur_thresh = out[exit_indices[-1]].max().cpu().detach().numpy() if exit_indices.shape[0] != 0 else thresh
        threshold = 0.9 * thresh + 0.1 * cur_thresh  # NOTE 如果cur_thresh不detach的话会导致显存爆炸，因为把tensor+int默认丢到显存里
        thresh_idx = (sorted_prob > threshold).sum()
        exit_indices = sorted_indices[:thresh_idx]
        non_exit_indices = sorted_indices[thresh_idx:]

    # follow threshold, exit when above thresh
    elif type == 1:
        threshold = thresh[exit_index]
        thresh_idx = (sorted_prob > threshold).sum()
        exit_indices = sorted_indices[:thresh_idx]
        non_exit_indices = sorted_indices[thresh_idx:]

    # take real inference, exit not exit previously and above thresh
    else:
        num_exits = len(outs)
        early_exits = torch.zeros(par_batch_size, dtype=torch.int64)
        early_exits.fill_(num_exits - 1)
        for i in range(num_exits - 1):  # 最后一个exit不需要threshold
            mask = (sorted_prob[i].cpu() > thresh[i]) & (early_exits == num_exits - 1)
            early_exits[mask] = i
        exit_indices = torch.nonzero(early_exits == exit_index).view(-1)
        non_exit_indices = torch.nonzero(early_exits != exit_index).view(-1)
        threshold = thresh[exit_index]
        thresh_idx = (sorted_prob > threshold).sum()

    return [threshold, thresh_idx], [exit_indices, non_exit_indices]


def infer_batch(outs, label, thresholds):
    exit_outs = []
    exit_labels = []
    exit_indices = []
    # for i in range(len(outs)):
    #     out = outs[i]
    #     max_prob, _ = out.max(dim=1)
    #     for index in exit_indices:
    #         max_prob[index] = 0
    #     if i == (len(outs) - 1):    # 最后一个exit 把剩下的都exit
    #         t = max_prob > 0
    #     else:
    #         t = max_prob >= thresholds[i]
    #     exit_index = torch.squeeze(torch.nonzero(t))
    #     exit_indices.append(exit_index)
    #     exit_outs.append(out[exit_index])
    #     exit_labels.append(label[exit_index])
    outs = torch.stack(outs, dim=0)
    max_prob_0, _ = outs.max(dim=2)
    # sorted_prob, sorted_indices = max_prob_0.sort(dim=0, descending=True)
    num_exits = len(outs)
    par_batch_size = outs[0].shape[0]
    early_exits = torch.zeros(par_batch_size, dtype=torch.int64)
    early_exits.fill_(num_exits - 1)
    for i in range(num_exits - 1):  # 最后一个exit不需要threshold
        mask = (max_prob_0[i].cpu() > thresholds[i]) & (early_exits == num_exits - 1)
        early_exits[mask] = i
    for i in range(num_exits):
        exit_index = torch.nonzero(early_exits == i).view(-1)
        exit_indices.append(exit_index)
        exit_outs.append(outs[i][exit_index])
        exit_labels.append(label[exit_index])

    # debug
    # print(exit_indices)
    # count = 0
    # for k in range(num_exits):
    #     # print(k,exit_indices[k].shape)
    #     count += exit_indices[k].shape[0]
    # print(count)

    return exit_outs, exit_labels, exit_indices


# def cal_ratio(ratios,e_idx):
#     if e_idx ==0:
#         return ratios[0]
#     else:
#         ratio = 1
#         for i in range(e_idx-1):
#             ratio *= (1-ratios[i])
#         ratio *= ratios[e_idx]
#         return ratio


def evaluate_ee_testset(model, test_dataloader, thresholds, name):
    # arg_str = get_argstring(args)
    model.load(name=name)
    model.eval()
    criterion = ConfCrLoss()
    test_loss = 0.0
    test_kw_correct = [0] * e_count
    hit_exit = torch.zeros([e_count])  #
    exit_count = torch.zeros([e_count])  # count of current exit samples
    hit_non_exit = torch.zeros([e_count])
    valid_acc_exit = torch.zeros([e_count])
    valid_acc_non_exit = torch.zeros([e_count])
    valid_exit_ratio = torch.zeros([e_count])
    for batch_idx, (audio_data, label_kw) in enumerate(test_dataloader):
        if train_on_gpu:
            audio_data = audio_data.cuda()
            label_kw = label_kw.cuda()
        outs = model(x=audio_data)
        # [out0, out1, out2] = outs

        batch_size = label_kw.shape[0]

        # forward through all exit
        # outs = model(x=audio_data)

        exit_outs, exit_labels, exit_indices = infer_batch(outs, label_kw, thresholds)
        loss = 0
        b_exit_hit = []
        b_non_exit_hit = []
        for i in range(e_count):
            if (exit_outs[i].dim() == 2):
                # print(exit_outs[i].shape)
                b_exit_hit.append(float(torch.sum(torch.argmax(exit_outs[i], 1) == exit_labels[i]).item()))
                hit_exit[i] += b_exit_hit[i]
                exit_count[i] += exit_outs[i].shape[0]
                loss += criterion.ce_loss(exit_outs[i], exit_labels[i])
            else:  # batch has output
                b_exit_hit.append(torch.Tensor(0))

        # if (i == 0):
        #     outs_conti = [None, outs[i], outs[i + 1]]
        #     ratio = [0, ratios[i]]
        # elif (i == e_count - 1):
        #     outs_conti = [outs[i - 1], outs[i], None]
        #     ratio = [ratios[i - 1], 1]
        # else:
        #     outs_conti = [outs[i - 1], outs[i], outs[i + 1]]
        #     ratio = [ratios[i - 1], ratios[i]]

        test_loss += loss.item() * audio_data.size(0)
        ba = torch.zeros([e_count])
        b_hit = torch.zeros([e_count])
        for i in range(e_count):
            b_hit[i] = float(torch.sum(torch.argmax(outs[i], 1) == label_kw).item())
            ba[i] = b_hit[i] / float(audio_data.shape[0]) * 100
            test_kw_correct[i] += b_hit[i]

        # if (batch_idx % 500 == 0):
        #     print("|  Valid step #{}   | Loss_ALL: {:.4f}  | ACC: {:.2f} {:.2f} {:.2f}|  ".format(
        #         step_idx, loss, ba[0], ba[1], ba[2]))

        # valid_kw_correct += b_hit

        # break

    # Loss statistics for current EPOCH

    valid_loss = test_loss / len(test_dataloader.dataset)
    train_kw_accuracy = torch.zeros([e_count])
    valid_kw_accuracy = torch.zeros([e_count])

    # valid_acc_non_exit[e_idx] = -1 if(len(dev_dataloader.dataset) - exit_count[e_idx])==0 \
    #     else 100 * hit_non_exit[e_idx] / (len(dev_dataloader.dataset) - exit_count[e_idx])

    infer_valid_acc = torch.sum(hit_exit) / len(test_dataloader.dataset)
    for i in range(e_count):
        valid_kw_accuracy[i] = 100.0 * (test_kw_correct[i] / len(test_dataloader.dataset))
        valid_acc_exit[i] = 100 * hit_exit[i] / exit_count[i]
        valid_exit_ratio[i] = exit_count[i] / len(test_dataloader.dataset)
        valid_acc_non_exit[i] = -1 if (len(test_dataloader.dataset) - exit_count[i]) == 0 \
            else 100 * hit_non_exit[i] / (len(test_dataloader.dataset) - exit_count[i])

        # update ratios
        # if i == 0:
        #     ratios[i] = valid_exit_ratio[i].numpy()
        # elif i == (e_count - 1):
        #     ratios[i] = 1
        # else:
        #     ratios[i] = ratios[i - 1] + valid_exit_ratio[i].numpy()

    real_latency = (model.flops * valid_exit_ratio).sum().numpy()
    # print(output.shape)
    # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
    # print(f1_scores)

    print("==================| Test Set |====================")
    print("  | INFER ACC: {:.2f}%\t| ".format(infer_valid_acc * 100))
    print("  | Average Latency: {:.4f}\t| ".format(real_latency))
    for i in range(e_count):
        print(
            "            EXIT {} | VAL ACC :  {:.2f}%\t ｜ Exit Ratio: {:.2f}% | Thresholds: {:.4f} |ACC_exit: {:.2f}%\t| ACC_non_exit {:.2f}%\t | VAL LOSS : {:.2f}".format(
                i, valid_kw_accuracy[i], valid_exit_ratio[i] * 100, thresholds[i], valid_acc_exit[i],
                valid_acc_non_exit[i], valid_loss))
    print("===========================================================================")


def get_argstring(arg):
    result = ""
    if arg == "":
        return result
    # 遍历命名空间对象的属性
    for key, value in vars(arg).items():
        key = key.replace(" ", "_")  # 用下划线替换空格
        result += f"{key}_{value}_"

    # 去除最后一个逗号
    result = result[:-1]

    # 输出生成的字符串
    return result
