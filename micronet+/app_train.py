import torch
import os
import optparse
import dataset
import time
import numpy as np
from tqdm import tqdm
from self_ops import init_lookup_tables

import my_utils
import app_layers
import models
import common_models.resnet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    lossLayer = torch.nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = lossLayer(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch {}: [{}/{}]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset), loss.item()
            ))

            
def calibrate(model, device, train_loader):
    model.train()
    print('Calibrating...')
    t_start = time.time()
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        _ = model(data)
    print(f'Calibration finished in {time.time() - t_start:.2f} seconds')


best_acc = 0
def test(model, device, test_loader, epoch, model_path=None, use_top5=False):
    global best_acc
    model.eval()
    test_loss = 0
    lossLayer = torch.nn.CrossEntropyLoss(reduction='sum')

    if not use_top5:
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += lossLayer(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc = 100. * correct / len(test_loader.dataset)
        if model_path is not None:
            if test_acc > best_acc:
                best_acc = test_acc
                path = f'{model_path}_top1_acc_{best_acc}.pth'
                print(f'Epoch {epoch}: Saving model to {path}')
                torch.save(model.state_dict(), path)
        print(f'Epoch {epoch}: Test set: Average loss: {test_loss}, Accuracy: {test_acc}%, Best accuracy: {best_acc}%')
    else:
        correct_top5 = 0
        correct_top1 = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += lossLayer(output, target).item()
            # get top-5 accuracy
            maxk = max((1, 5))
            target_resize = target.view(-1, 1)
            _, pred = output.topk(maxk, 1, True, True)
            correct_top5 += torch.eq(pred, target_resize).sum().float().item()
            # get top-1 accuracy 
            pred_top1 = output.argmax(dim=1, keepdim=True)
            correct_top1 += pred_top1.eq(target.view_as(pred_top1)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_acc_top5 = 100. * correct_top5 / len(test_loader.dataset)
        test_acc_top1 = 100. * correct_top1 / len(test_loader.dataset)
        if model_path is not None:
            if test_acc_top5 > best_acc:
                best_acc = test_acc_top5
                path = f'{model_path}_top5_acc_{best_acc}.pth'
                print(f'Epoch {epoch}: Saving model to {path}')
                torch.save(model.state_dict(), path)        
        print(f'Epoch {epoch}: Test set: Average loss: {test_loss}, Top-1 Accuracy: {test_acc_top1}%, Top-5 Accuracy: {test_acc_top5}%, Best top-5 accuracy: {best_acc}%')


def GetOpt():
    parser = optparse.OptionParser()
    parser.add_option('-p', '--model_path',
                        action='store',
                        dest='model_path',
                        # default='./pretrained/cifar10_lenet_fp32_acc_76.27.pth',
                        # default='./pretrained/cifar10_vgg19_fp32_acc_92.69.pth',
                        default='./pretrained/cifar10_resnet18_fp32_acc_94.06.pth',
                        # default='./pretrained/cifar100_resnet34_fp32_acc_77.92.pth',
                        # default='./pretrained/cifar100_resnet50_fp32_acc_78.74.pth',
                        help='path to load pre-trained FP32 model')
    parser.add_option('-f', '--fix_seed',
                        action='store_true',
                        dest='fix_seed',
                        default=False,
                        help='fix random seed')
    parser.add_option('-u', '--use_ste_gradient',
                        action='store_true',
                        dest='use_ste_gradient',
                        default=False,
                        help='use straight-through estimator for gradient')
    parser.add_option('-s', '--batch_size',
                        action='store',
                        dest='batch_size',
                        default=64,
                        help='batch size')
    parser.add_option('-e', '--epochs',
                        action='store',
                        dest='epochs',
                        default=30,
                        help='epochs')
    parser.add_option('-r', '--learn_rate',
                        action='store',
                        dest='learn_rate',
                        default=0.001,
                        help='learning rate')
    parser.add_option('-d', '--decresing_lr',
                        action='store',
                        dest='decreasing_lr',
                        default='10,20',
                        help='decreasing learning rate')
    parser.add_option('-b', '--quant_bits',
                        action='store',
                        dest='quant_bits',
                        default=7,
                        help='quantization bits')
    parser.add_option('-k', '--num_workers',
                        action='store',
                        dest='num_workers',
                        default=16,
                        help='number of workers')
    parser.add_option('-l', '--lut_file_name',
                        action='store',
                        dest='lut_file_name',
                        default='./app_mults/vert_trunc/mult7u/7_7_U_SP_WT_CL_GenMul_vertical_truncation_6cols_lutfp+bp_hws16.txt',
                        help='lookup table file name')
    
    options, _ = parser.parse_args()
    return options


def main():
    # get options
    options = GetOpt()

    # get hyperparameters
    load_model_path, batch_size, epochs, lr, decreasing_lr, quant_bits, num_workers = \
        options.model_path, int(options.batch_size), int(options.epochs), float(options.learn_rate), list(map(int, options.decreasing_lr.split(','))), int(options.quant_bits), int(options.num_workers)
    print(f'batch_size: {batch_size}, epochs: {epochs}, lr: {lr}, decreasing_lr: {decreasing_lr}, quant_bits: {quant_bits}, num_workers: {num_workers}')

    # get device
    assert torch.cuda.is_available(), 'must use cuda'

    # set random seed
    if options.fix_seed:
        _seed = 199608224
    else:
        _seed = int(time.time())
    setup_seed(_seed)
    print(f'Random seed: {_seed}')

    # deal with model path
    assert load_model_path[-4:] == '.pth', 'Model path must end with .pth'
    assert load_model_path.find('fp32') != -1, 'Model path must contain fp32'
    assert os.path.exists(load_model_path), f'Model path {load_model_path} does not exist'
    save_model_path = load_model_path[:load_model_path.find('fp32')] + f'{quant_bits}bit_qat'
    save_model_path = save_model_path[save_model_path.rfind('/')+1:]
    save_model_path = './ckpt/' + save_model_path
    print('load model from', load_model_path)
    print(f'Model saving path prefix: {save_model_path}')

    # get dataset name
    if load_model_path.find('cifar10_') != -1:
        dataset_name = 'cifar10'
        use_top5 = False
    elif load_model_path.find('cifar100_') != -1:
        dataset_name = 'cifar100'
        use_top5 = True
    else:
        raise NotImplementedError
    print(f'Dataset: {dataset_name}')

    # load data
    if dataset_name == 'cifar10':
        train_loader, test_loader, train_loader_no_aug = dataset.get_cifar10(batch_size=batch_size, num_workers=num_workers)
    elif dataset_name == 'cifar100':
        train_loader, test_loader, train_loader_no_aug = dataset.get_cifar100(batch_size=batch_size, num_workers=num_workers)
    else:
        raise NotImplementedError

    # load model
    if load_model_path.find('lenet') != -1:
        model = models.Lenet()
    elif load_model_path.find('vgg19') != -1:
        model = models.VGG('VGG19')
    elif load_model_path.find('resnet18') != -1:
        model = models.resnet18()
    elif load_model_path.find('resnet34') != -1:
        model = common_models.resnet.resnet34()
    elif load_model_path.find('resnet50') != -1:
        model = common_models.resnet.resnet50()
    else:
        raise NotImplementedError
    model = model.cuda()
    device = next(model.parameters()).device
    model.load_state_dict(torch.load(load_model_path, weights_only=True, map_location=device))
    print('FP32 model')
    test(model, device, test_loader, -2, None, use_top5)

    # parameters
    q_type = 1 # 0: symmetric, 1: asymmetric
    q_level = 1 # 0: channel-wise, 1: layer-wise
    weight_observer = 0 # 0: minmax, 1: moving average
    bn_fuse = False # fuse bn into conv
    bn_fuse_calib = False # calibrate bn fuse
    pretrained_model = True # use pretrained model
    qaft = False # quantize aware fine-tuning
    ptq = False # post-training quantization
    percentile = 0.999999 # percentile for ptq
    use_ste_gradient = options.use_ste_gradient # whether use straight-through estimator for gradient

    # prepare approximate multiplication lookup tables
    lut_file_name = options.lut_file_name
    init_lookup_tables(device, lut_file_name, quant_bits)

    # apply approximation
    app_layers.prepare(
        model,
        inplace=True,
        a_bits=quant_bits,
        w_bits=quant_bits,
        q_type=q_type,
        q_level=q_level,
        weight_observer=weight_observer,
        bn_fuse=bn_fuse,
        bn_fuse_calib=bn_fuse_calib,
        pretrained_model=pretrained_model,
        qaft=qaft,
        ptq=ptq,
        percentile=percentile,
        use_ste_gradient=use_ste_gradient
    )
    # print(model)

    model = model.to(device)
    calibrate(model, device, train_loader_no_aug)
    test(model, device, test_loader, -1, None, use_top5)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0)

    # train
    t_begin = time.time()
    for epoch in range(epochs):
        if epoch in decreasing_lr:
            optimizer.param_groups[0]['lr'] *= 0.5
            print(f'AppTrain Epoch {epoch}: Learning rate decreased to {optimizer.param_groups[0]["lr"]}')
        else:
            print(f'AppTrain Epoch {epoch}: Learning rate: {optimizer.param_groups[0]["lr"]}')
        train(model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader, epoch, save_model_path, use_top5)
        my_utils.report_time_and_speed(t_begin, epoch, epochs, len(train_loader))


if __name__ == "__main__":
    main() 