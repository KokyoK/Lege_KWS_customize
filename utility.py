import torch
torch.manual_seed(42)
import torch.nn as nn
import torch.utils.data as data
import speech_dataset as sd
import model as md

# check if CUDA is available
train_on_gpu = torch.cuda.is_available()
time_check = True

train_on_gpu = False
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
    # path_count = [0,0,0]
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


def train(model, root_dir, word_list, num_epoch):
    """
    Trains TCResNet
    """

    # Enable GPU training
    if train_on_gpu:
        model.cuda()
    
    # Loading dataset
    ap = sd.AudioPreprocessor() # Computes Log-Mel spectrogram
    train_files, dev_files, test_files = sd.split_dataset(root_dir, word_list)

    train_data = sd.SpeechDataset(train_files, "train", ap, word_list)
    dev_data = sd.SpeechDataset(dev_files, "dev", ap, word_list)
    test_data = sd.SpeechDataset(test_files, "test", ap, word_list)

    train_dataloader = data.DataLoader(train_data, batch_size=16, shuffle=True)
    dev_dataloader = data.DataLoader(dev_data, batch_size=1, shuffle=True)
    test_dataloader = data.DataLoader(test_data, batch_size=1, shuffle=True)


    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4,weight_decay=0)

    # Training
    step_idx = 0
    valid_accuracy = 0
    previous_valid_accuracy = 0

    for epoch in range(num_epoch):

        train_loss = 0.0
        valid_loss = 0.0
        valid_correct = 0.0
        train_correct = 0.0


        model.train()

        for batch_idx, (audio_data, labels) in enumerate(train_dataloader):
            if train_on_gpu:
                audio_data = audio_data.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()

            out_full = model(x=audio_data)
            loss_full = criterion(out_full, labels)

            with torch.autograd.set_detect_anomaly(True):
                loss_full.backward(retain_graph=True)
            loss = loss_full

            
            optimizer.step()
            train_loss += loss.item()*audio_data.size(0)
            # batch_accuracy = float(torch.sum(torch.argmax(out_full, dim=1) == labels).item())/float(audio_data.shape[0])
            batch_accuracy = float(torch.sum(torch.argmax(out_full.value, dim=1) == labels).item()) / float(
                audio_data.shape[0])

            
            if (batch_idx%10 == 0):
                print("Epoch {} | Train step #{}   | Loss: {:.4f}  | Accuracy: {:.4f}".format(epoch, step_idx, loss, batch_accuracy))
            train_correct += torch.sum(torch.argmax(out_full.value, dim=1) == labels).item()
            step_idx += 1

        quantized_model = torch.quantization.convert(model.eval(), inplace=False)

        quantized_model.eval()
        quantized_model.mode = "eval"

        # Validation (1 epoch)
        model.eval()
        model.mode = "eval"
        path_count = [0,0,0]
        total_infer_time = 0
        
        for batch_idx, (audio_data, labels) in enumerate(dev_dataloader):

            if train_on_gpu:
                audio_data = audio_data.cuda()
                labels = labels.cuda()
                
            start.record() if time_check else 1
            output = quantized_model(audio_data)
            path = 0
            end.record() if time_check else 1
            
            
            loss = criterion(output, labels)
            valid_loss += loss.item()*audio_data.size(0)

            batch_accuracy = (torch.sum(torch.argmax(output.value, dim=1) == labels).item())/audio_data.shape[0]
            # print("Epoch {} | Eval step #{}     | Loss: {:.4f}  | Accuracy: {:.4f}".format(epoch,batch_idx, loss, batch_accuracy))

            path_count[path] += 1
            valid_correct += torch.sum(torch.argmax(output.value, dim=1) == labels).item()

            if(time_check):
                torch.cuda.synchronize()
                elapsed_time = start.elapsed_time(end)
                total_infer_time += elapsed_time

            # print("Validation path count:   ", path_count)

        # Loss statistics
        train_loss = train_loss/len(train_dataloader.dataset)
        valid_loss = valid_loss/len(dev_dataloader.dataset)
        train_accuracy = 100.0 * (train_correct / len(train_dataloader.dataset))
        valid_accuracy = 100.0*(valid_correct/len(dev_dataloader.dataset))
        # print(output.shape)
        # f1_scores = f1_score(labels, torch.max(output.detach(), 1)[0], average=None, )
        # print(f1_scores)
        print("===========================================================================")
        print("EPOCH #{}     | TRAIN ACC: {:.2f}% | TRAIN LOSS : {:.2f}".format(epoch, train_accuracy,  train_loss))
        print("             | VAL ACC :  {:.2f}% | VAL LOSS   : {:.2f}".format(valid_accuracy,  valid_loss))
        print("Validation set inference time:    ",total_infer_time/len(dev_dataloader.dataset))
        print("===========================================================================")
        
        if (valid_accuracy > previous_valid_accuracy):
            previous_valid_accuracy = valid_accuracy
            print("Saving current model...")
            # model.save()
            model.save(is_onnx=0, name='saved_model/e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format( epoch,
                                                                                              valid_accuracy,
                                                                                              valid_loss))
            # model.save(is_onnx=0,name='saved_model/w{}b{}_e_{}_valacc_{:.3f}_valloss_{:.3f}_.pt'.format(md.qw,md.qa,epoch,valid_accuracy,valid_loss))
            # torch.save(quantized_model.state_dict(), 'saved_model/q_epoch_{}_valacc_{:.3f}_valloss_{:.3f}_.pth'.format(epoch,valid_accuracy,valid_loss))
       
        # Update scheduler (for decaying learning rate)
        # scheduler.step()

    # Final test
    evaluate_testset(model, test_dataloader)
    
