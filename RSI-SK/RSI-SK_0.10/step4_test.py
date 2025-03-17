import os
import torch
import torch.nn.functional as F
import numpy as np
from utils import data_loader2
from tqdm import tqdm
from utils.metrics import Evaluator
from PIL import Image
from network.UNet_CD_test import Unet
from network.class_pro_2 import class_pro
import sys
import time
start=time.time()

threshold_a = 1.1
class DualOutput:
    def __init__(self, file_name):
        self.console = sys.stdout
        self.file = open(file_name, 'w')
    def write(self, message):
        self.console.write(message)
        self.file.write(message)

    def flush(self):
        self.console.flush()
        self.file.flush()



def test(test_loader, Eva_test, save_path, net):
    print("Strat validing!")


    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()

            _, _, class1, class2, class3, class4, class_5, preduA = class_pro(A, B)

            preds, preds_class, preds_class_CD, layer1_A, layer1_AA = net(A, B, class1, class2, class3, class4, class_5,
                                                                          preduA)

            output = F.sigmoid(preds[:, 0, :, :].unsqueeze(1))
            threshold = (1 - output * threshold_a).detach()
            output[output >= threshold] = 1
            output[output < threshold] = 0


            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy()

            for i in range(output.shape[0]):
                probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
                final_mask = probs_array * 255
                final_mask = final_mask.astype(np.uint8)
                final_savepath = save_path + filename[i] + '.png'
                im = Image.fromarray(final_mask)
                im.save(final_savepath)

            Eva_test.add_batch(target, pred)
    print('target.shape', target.shape)
    print('pred.shape', pred.shape)


    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA=Eva_test.OA()
    Kappa=Eva_test.Kappa()


    print('[Test] F1: %.4f, Precision:%.4f, Recall: %.4f, OA: %.4f, Kappa: %.4f,IoU: %.4f' % ( F1[1],Pre[1],Recall[1],OA[1],Kappa[1],IoU[1]))

    print('F1-Score: Precision: Recall: OA: Kappa: IoU: ')

    print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100, Kappa[1] * 100,IoU[1] * 100))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=1, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')
    parser.add_argument('--data_name', type=str, default='LEVIR',
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='Unet',
                        help='the test rgb images root')

    parser.add_argument('--save_path', type=str, default=r'/home/user/zly/daima3/gongkai/RSI-SK/result/t0.10/') #半监督C2F-SemiCD影像保存路径！！！


    opt = parser.parse_args()

    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')


    log_file = r'/home/user/zly/daima3/gongkai/RSI-SK/result/test_t0.10.txt'  # Change to your preferred filename
    sys.stdout = DualOutput(log_file)


    if opt.data_name == 'LEVIR':
        opt.test_root = '/home/user/zly/data2/LEVIR_clip/sample/orgion/test/'

    test_loader = data_loader2.get_test_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_test = Evaluator(num_class=2)

    if opt.model_name == 'Unet':
        model = Unet().cuda()


    save_path = r'/home/user/zly/daima3/gongkai/RSI-SK/pth/t0.10/'

    opt.load = save_path + opt.data_name + '/' + opt.model_name+ '_train1_' + '_best_student_iou.pth'
    

    if opt.load is not None:
        print('load model from ', opt.load)
        checkpoint_stud = torch.load(opt.load)
        model.load_state_dict(checkpoint_stud['best_student_net '])


    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test(test_loader, Eva_test, opt.save_path, model)

end=time.time()
print('程序测试test的时间为:',end-start)
