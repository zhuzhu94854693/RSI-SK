import time
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import data_loader2
from tqdm import tqdm
import random
from utils.metrics import Evaluator
import time
from network.UNet_CD import Unet
from network.class_pro_2 import class_pro
import matplotlib.pyplot as plt



start = time.time()
threshold_a = 1
seed = 45
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def update_ema_variables(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def train1(train_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2,
           data_name, save_path, net, ema_net, criterion, semicriterion, optimizer, use_ema, num_epoches):
    global best_iou
    epoch_loss = 0
    net.train(True)
    ema_net.train(True)

    length = 0
    st = time.time()
    loss_semi = torch.zeros(1)
    with tqdm(total=len(train_loader), desc=f'Eps {epoch}/{num_epoches}', unit='img') as pbar:
        for i, (A, B, mask, with_label) in enumerate(train_loader):
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()

            with_label = with_label.cuda()


            optimizer.zero_grad()
            if use_ema is False:

                if with_label.any():

                    _, _, class_1, class_2, class_3, class_4, class_5, preduA = class_pro(A[with_label], B[with_label])

                    preds, preds_class, preds_class_CD, layer1_A, layer1_AA = net(A[with_label], B[with_label], class_1, class_2, class_3, class_4, class_5, preduA, use_ema, with_label)
                    Y_CD = torch.zeros_like(Y[with_label])



                    loss = criterion(preds[:, 0, :, :].unsqueeze(1), Y[with_label]) + criterion(preds_class[:, 0, :, :].unsqueeze(1),Y[with_label])  # + criterion(preds_class_CD[:,0,:,:].unsqueeze(1), Y[with_label])
                    Y = Y[with_label]
                else:

                    continue
            else:


                _, _, class1, class2, class3, class4, class_5, preduA = class_pro(A, B)

                preds, preds_class, preds_class_CD, layer1_A, layer1_AA  = net(A, B, class1, class2, class3, class4, class_5, preduA, use_ema, with_label)
                Y_CD = torch.zeros_like(Y[with_label])
                if with_label.any():
                    loss = criterion(preds[:, 0, :, :][with_label].unsqueeze(1), Y[with_label]) + criterion(preds_class[:, 0, :, :][with_label].unsqueeze(1), Y[with_label])
                else:
                    loss = 0

            if use_ema is True and torch.sum(~with_label) > 0:

                with torch.no_grad():
                    z1 = A[~with_label]
                    z2 = B[~with_label]

                    _, _, class_z1, class_z2, class_z3, class_z4, class_z5, z1_preduA = class_pro(z1, z2)

                    pseudo_preds, pseudo_preds_class, pseudo_preds_class_CD, pseudo_layer1_A, pseudo_layer1_AA = ema_net(z1, z2, class_z1, class_z2, class_z3, class_z4, class_z5, z1_preduA, use_ema, with_label)
                    pseudo_preds = torch.sigmoid(pseudo_preds).detach()
                    pseudo_preds_class = torch.sigmoid(pseudo_preds_class).detach()
                    pseudo_preds_class_CD = torch.sigmoid(pseudo_preds_class_CD).detach()
                    Y_CD = torch.zeros_like(preds[:, 0, :, :][~with_label].unsqueeze(1))


                loss_semi = semicriterion(preds[:, 0, :, :][~with_label].unsqueeze(1), pseudo_preds_class[:, 0, :, :].unsqueeze(1)) + semicriterion(preds_class[:, 0, :, :][~with_label].unsqueeze(1), pseudo_preds[:, 0, :, :].unsqueeze(1)) + semicriterion(preds_class_CD[:, 0, :, :][~with_label].unsqueeze(1), pseudo_preds_class_CD[:, 0, :, :].unsqueeze(1))

                loss = loss + 0.2 * loss_semi + 0.2 * layer1_A + 0.2 * layer1_AA
                device = torch.device('cuda:0')
                mask = mask.to(device)
                with_label = with_label.to(device)

                Eva_train2.add_batch(mask[~with_label].cpu().numpy().astype(int), (preds[:, 0, :, :][~with_label].unsqueeze(1) > 0).cpu().numpy().astype(int))


            loss.backward()
            optimizer.step()

            with torch.no_grad():
                update_ema_variables(net, ema_net, alpha=0.99)

            epoch_loss += loss.item()

            output = F.sigmoid(preds[:, 0, :, :].unsqueeze(1))
            output_class = F.sigmoid(preds_class[:, 0, :, :].unsqueeze(1))
            threshold = (1 - output * threshold_a).detach()
            output[output >= threshold] = 1
            output[output < threshold] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_train.add_batch(target, pred)
            pbar.set_postfix(**{'LAll': loss.item(), 'LSemi': loss_semi.item()})
            pbar.update(1)
            length += 1

    IoU = Eva_train.Intersection_over_Union()[1]
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

   
    log_to_file(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            Eva_train2.Intersection_over_Union()[1], Eva_train2.Precision()[1], Eva_train2.Recall()[1],
            Eva_train2.F1()[1]))
    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            Eva_train2.Intersection_over_Union()[1], Eva_train2.Precision()[1], Eva_train2.Recall()[1],
            Eva_train2.F1()[1]))

    if use_ema is True:
        log_to_file(
            'Epoch [%d/%d],\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
                epoch, num_epoches, \
                IoU, Pre, Recall, F1))
        print(
            'Epoch [%d/%d],\n[Training]IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
                epoch, num_epoches, \
                IoU, Pre, Recall, F1))
    log_to_file("Strat validing!")
    print("Strat validing!")

    net.train(False)
    net.eval()
    ema_net.train(False)
    ema_net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()


            _, _, class_1, class_2, class_3, class_4, class_5, preduA = class_pro(A, B)

            preds, preds_class, preds_class_CD, _, _  = net(A, B, class_1, class_2, class_3, class_4, class_5, preduA, use_ema, with_label)
            output = F.sigmoid(preds[:, 0, :, :].unsqueeze(1))
            output_class = F.sigmoid(preds_class[:, 0, :, :].unsqueeze(1))
            threshold = (1 - output * threshold_a).detach()
            output[output >= threshold] = 1
            output[output < threshold] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)
            Eva_val.add_batch(target, pred)

            preds_ema, preds_ema_class, preds_ema_class_CD, _, _ = ema_net(A, B, class_1, class_2, class_3, class_4, class_5, preduA, use_ema, with_label)  # [1]
            Eva_val2.add_batch(target, (preds_ema[:, 0, :, :].unsqueeze(1) > 0).cpu().numpy().astype(int))
            length += 1

    IoU = Eva_val.Intersection_over_Union()
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    log_to_file('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))
    print('[Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1], Pre[1], Recall[1], F1[1]))

    log_to_file('[Ema Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (Eva_val2.Intersection_over_Union()[1], Eva_val2.Precision()[1], Eva_val2.Recall()[1], Eva_val2.F1()[1]))
    print('[Ema Validation] IoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (Eva_val2.Intersection_over_Union()[1], Eva_val2.Precision()[1], Eva_val2.Recall()[1], Eva_val2.F1()[1]))

    new_iou = IoU[1]
    if new_iou >= best_iou:
        best_iou = new_iou
        best_epoch = epoch
        log_to_file('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        print('Best Model Iou :%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1], F1[1], best_epoch))
        torch.save(net.state_dict(), save_path + '_best_student_iou.pth')
        torch.save(ema_net.state_dict(), save_path + '_best_teacher_iou.pth')
        print('best_epoch', epoch)

        student_dir = save_path + '_train1_' + '_best_student_iou.pth'

        student_state = {'best_student_net ': net.state_dict(),
                         'optimizer ': optimizer.state_dict(),
                         ' epoch': epoch}

        torch.save(student_state, student_dir)
        torch.save(ema_net.state_dict(), save_path + '_train1_' + '_best_teacher_iou.pth')
    log_to_file('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))



if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--train_ratio', type=float, default=0.05, help='Proportion of the labeled images')  # 修改这里！！！
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--gpu_id', type=str, default='0,1', help='train use gpu')
    parser.add_argument('--data_name', type=str, default='LEVIR',
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='Unet',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default=r'/home/user/zly/daima3/gongkai/RSI-SK/pth/t0.10/')


    opt = parser.parse_args()
    print('labeled ration=0.05,Ablation现在半监督损失函数系数为:0.2!')



    log_file_path = os.path.join(r'/home/user/zly/daima3/gongkai/RSI-SK/pth/', "training_t0.10.txt")


    with open(log_file_path, "a") as log_file:

        log_file.write("Training started...\n")



    def log_to_file(message):
        with open(log_file_path, "a") as log_file:
            log_file.write(message + "\n")


    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    if opt.data_name == 'LEVIR':
        opt.train_root = r'/home/user/zly/data2/LEVIR_clip/sample/'
        opt.val_root = r'/home/user/zly/data2/LEVIR_clip/sample/orgion/val/'

    train_loader = data_loader2.get_semiloader(opt.train_root, opt.batchsize, opt.trainsize, opt.train_ratio,
                                               num_workers=0, shuffle=True, pin_memory=False)
    val_loader = data_loader2.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=0, shuffle=False,
                                              pin_memory=False)


    Eva_train = Evaluator(num_class=2)
    Eva_train2 = Evaluator(num_class=2)
    Eva_val = Evaluator(num_class=2)
    Eva_val2 = Evaluator(num_class=2)

    model = Unet().cuda()
    ema_model = Unet().cuda()

    for param in ema_model.parameters():
        param.detach_()


    criterion = nn.BCEWithLogitsLoss().cuda()
    semicriterion = nn.BCEWithLogitsLoss().cuda()


    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0



    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])


        if epoch < 5:
            use_ema = False

        else:
            use_ema = True



        Eva_train.reset()
        Eva_train2.reset()
        Eva_val.reset()
        Eva_val2.reset()
        train1(train_loader, val_loader, Eva_train, Eva_train2, Eva_val, Eva_val2, data_name, save_path, model,
               ema_model, criterion, semicriterion, optimizer, use_ema, opt.epoch)

        lr_scheduler.step()


