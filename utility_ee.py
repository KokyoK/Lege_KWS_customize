import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.utils.data as data
import speech_dataset as sd
import model as md

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
time_check = True

# train_on_gpu = False
time_check = False





if not train_on_gpu:
    print('CUDA is not available.  Using CPU ...')
else:
    print('CUDA is available!  Using GPU ...') 
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    time_check = True



def evaluate_testset(model, test_dataloader):
    # Final test
    test_loss = 0.0
    test_correct = 0.0
    model.load()
    model.eval()
    total_infer_time = 0
    path_count = [0,0,0]
    total_flops = 0

    criterion = nn.CrossEntropyLoss()
     
    for batch_idx, (audio_data, labels) in enumerate(test_dataloader):

        if train_on_gpu:
            model.cuda()
            audio_data = audio_data.cuda()
            labels = labels.cuda()

        output = model(audio_data)
        
        loss = criterion(output, labels)
        test_loss += loss.item()*audio_data.size(0)
        test_correct += (torch.sum(torch.argmax(output, 1) == labels).item())



        if(time_check):
            torch.cuda.synchronize()
            elapsed_time = start.elapsed_time(end)
            total_infer_time += elapsed_time

    test_loss = test_loss/len(test_dataloader.dataset)
    test_accuracy = 100.0*(test_correct/len(test_dataloader.dataset))

    
    print("================================================")
    print(" FINAL ACCURACY : {:.4f}% - TEST LOSS : {:.4f}".format(test_accuracy, test_loss))
    print(" Time for avg test set inference:    ",total_infer_time/len(test_dataloader.dataset))
    print(" Flops for avg test set inference:    ",total_flops / len(test_dataloader.dataset))
    # print(" Test Set path count:   ", path_count)
    print("================================================")

class OrgLoss(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, map_k, map_s,Share_W, Kws_P,Speaker_P):
        # mul = torch.matmul(map_k.squeeze(dim=2), map_s.squeeze(dim=2).permute(0,2,1))
        # o_loss = torch.norm(mul, p='fro') ** 2 / (48*48)
        o_loss = ((torch.norm(Share_W @ Kws_P.T, p='fro') + torch.norm(Share_W @ Speaker_P.T, p='fro')) ** 2) / (
                Share_W.data.shape[0] * Kws_P.data.shape[1])
        return o_loss

def train(model, root_dir, word_list, speaker_list,num_epoch):
    """
    Trains TCResNet
    """

    # Enable GPU training
    if train_on_gpu:
        model.cuda()
    
    # Loading dataset
    ap = sd.AudioPreprocessor() # Computes Log-Mel spectrogram
    train_files, dev_files, test_files = sd.split_dataset(root_dir, word_list,speaker_list )

    train_data = sd.SpeechDataset(train_files, "train", ap, word_list,speaker_list)
    dev_data = sd.SpeechDataset(dev_files, "dev", ap, word_list,speaker_list)
    test_data = sd.SpeechDataset(test_files, "test", ap, word_list,speaker_list)

    train_dataloader = data.DataLoader(train_data, batch_size=16, shuffle=True)
    # dev_dataloader = data.DataLoader(dev_data, batch_size=16, shuffle=True)
    dev_dataloader = data.DataLoader(test_data, batch_size=16, shuffle=True)
    test_dataloader = data.DataLoader(test_data, batch_size=16, shuffle=True)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3,weight_decay=1e-5)
    OLoss = OrgLoss()
    # Training
    step_idx = 0
    valid_accuracy = 0
    previous_valid_accuracy = 0

    for epoch in range(num_epoch):

        train_loss = 0.0
        valid_loss = 0.0
        valid_kw_correct = 0
        valid_id_correct = 0
        train_kw_correct = 0
        train_id_correct = 0


        model.train()

        for batch_idx, (audio_data,label_kw, label_id) in enumerate(train_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                label_kw = label_kw.cuda()
                label_id = label_id.cuda()

            optimizer.zero_grad()

            out_kw, out_id, map_kw, map_s = model(x=audio_data)

            # cal_loss

            loss_kw = criterion(out_kw, label_kw)
            loss_id = criterion(out_id, label_id)
            loss_o = OLoss(map_kw,map_s,model.share_para, model.kws_para, model.speaker_para)
            # loss_full = 1.5*loss_id + 0.5*loss_kw + loss_o
            loss_full = loss_id


            with torch.autograd.set_detect_anomaly(True):
                loss_full.backward(retain_graph=True)
                # loss_kw.backward(retain_graph=True)
                # loss_id.backward(retain_graph=True)
                # loss_o.backward(retain_graph=True)
            loss = loss_kw

            optimizer.step()

            train_loss += loss.item()*audio_data.size(0)
            kw_batch_accuracy = float(torch.sum(torch.argmax(out_kw, 1) == label_kw).item()) / float(audio_data.shape[0])
            id_batch_accuracy = float(torch.sum(torch.argmax(out_id, 1) == label_id).item()) / float(audio_data.shape[0])
            
            if (batch_idx%100 == 0):
                print("Epoch {} | Train step #{}   | Loss_ALL: {:.4f} | Loss Otho {:.4f}  | KWS ACC: {:.4f} | SPEAKER ACC: {:.4f} ".format(epoch, step_idx, loss, loss_o,kw_batch_accuracy, id_batch_accuracy))

            train_kw_correct += torch.sum(torch.argmax(out_kw, 1) == label_kw).item()
            train_id_correct += torch.sum(torch.argmax(out_id, 1) == label_id).item()
            step_idx += 1


        # Validation (1 epoch)
        model.eval()
        model.mode = "eval"
        total_infer_time = 0

        for batch_idx, (audio_data, label_kw, label_id) in enumerate(dev_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                label_kw = label_kw.cuda()
                label_id = label_id.cuda()

            optimizer.zero_grad()

            out_kw, out_id, map_kw, map_s = model(x=audio_data)

            # cal_loss
            loss_kw = criterion(out_kw, label_kw)
            loss_id = criterion(out_id, label_id)
            loss_o = OLoss(map_kw,map_s,model.share_para, model.kws_para, model.speaker_para)
            loss_full = loss_id + loss_kw + loss_o

            loss = loss_full

            train_loss += loss.item() * audio_data.size(0)
            kw_batch_accuracy = float(torch.sum(torch.argmax(out_kw, 1) == label_kw).item()) / float(
                audio_data.shape[0])
            id_batch_accuracy = float(torch.sum(torch.argmax(out_id, 1) == label_id).item()) / float(
                audio_data.shape[0])

            if (batch_idx % 100 == 0):
                print(
                    "Epoch {} | Eval  step #{}   | Loss: {:.4f}  | KWS ACC: {:.4f} | SPEAKER ACC: {:.4f} ".format(epoch,
                                                                                                                  step_idx,
                                                                                                                  loss,
                                                                                                                  kw_batch_accuracy,
                                                                                                                  id_batch_accuracy))

            valid_kw_correct += torch.sum(torch.argmax(out_kw, 1) == label_kw).item()
            valid_id_correct += torch.sum(torch.argmax(out_id, 1) == label_id).item()
            step_idx += 1

        # Loss statistics
        train_loss = train_loss/len(train_dataloader.dataset)
        valid_loss = valid_loss/len(dev_dataloader.dataset)
        train_kw_accuracy = 100.0 * (train_kw_correct / len(train_dataloader.dataset))
        train_id_accuracy = 100.0 * (train_id_correct / len(train_dataloader.dataset))
        valid_kw_accuracy = 100.0 * (valid_kw_correct / len(dev_dataloader.dataset))
        valid_id_accuracy = 100.0 * (valid_id_correct / len(dev_dataloader.dataset))
        # print(output.shape)
        # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
        # print(f1_scores)
        print("===========================================================================")
        print("EPOCH #{}     | TRAIN KWS ACC: {:.2f}% | TRAIN SPEAKER ACC: {:.2f}% | TRAIN LOSS : {:.2f}".format(epoch, train_kw_accuracy,train_id_accuracy , train_loss))
        print("             | VAL KWS ACC :  {:.2f}% | VAL   SPEAKER ACC: {:.2f}% | VAL LOSS   : {:.2f}".format(valid_kw_accuracy, valid_id_accuracy, valid_loss))
        # print("Validation path count:   ", path_count)
        # print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
        print("===========================================================================")
        
        if (valid_id_accuracy > previous_valid_accuracy):
            previous_valid_accuracy = valid_id_accuracy
            print("Saving current model...")
            model.save()
            model.save(is_onnx=0, name=f'e_{epoch}_kw_{valid_kw_accuracy:.3f}_valloss_{valid_id_accuracy:.3f}_.pt')
            # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
            # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))

        # Update scheduler (for decaying learning rate)
        # scheduler.step()

    # Final test
    evaluate_testset(model, test_dataloader)
    
