import argparse
import os
from torchvision import transforms
from dataloader_fiveK import *
from torch.utils.data import DataLoader
import torch
from loss import *
from utils import *
from PIL import Image

# --- Parse hyper-parameters  --- #
parser = argparse.ArgumentParser(description='Hyper-parameters for At_Net')
parser.add_argument('-Is_save_image', help='Whether to save the image to local', default=True, type=bool)
parser.add_argument('-data_path', help='The data path', default='/home/liu/zhanglideng', type=str)
parser.add_argument('-category', help='Set image category (NYU or NTIRE?)', default='NYU', type=str)
parser.add_argument('-Is_train_result', help='Whether to save the dehazing results for enhanced training',
                    default=False,
                    type=bool)
parser.add_argument('-gth_test', help='Whether to add Gth testing', default=False, type=bool)
parser.add_argument('-batch_size', help='The batch size', default=1, type=int)
parser.add_argument('-weight', help='The loss weight', default=[1, 1, 1], type=list)
parser.add_argument('-excel_row', help='The excel row',
                    default=["num", "A", "beta", "A_l1", "t_l1", "J_l1"], type=list)
args = parser.parse_args()
Is_save_image = args.Is_save_image  # 是否保存图像测试结果
data_path = args.data_path  # 数据路径
batch_size = args.batch_size  # 测试批大小
excel_row = args.excel_row  # excel的列属性名
weight = args.weight  # 损失函数的权重
category = args.category  # 选择测试的数据集
Is_train_result = args.Is_train_result  # 是否为增强网络提供去雾结果

# 加载训练好的模型
file_path = find_pretrain('At_result')
model_path = file_path + '/At_model.pt'
net = torch.load(model_path)
net = net.cuda()
loss_net = train_loss_net().cuda()
transform = transforms.Compose([transforms.ToTensor()])

if Is_train_result:
    if category == 'NYU':
        input_path = [data_path + '/data/nyu/train_hazy_patch/', data_path + '/data/nyu/val_hazy_patch/',
                      data_path + '/data/nyu/test_hazy_patch/']
        output_path = [data_path + '/data/nyu/train_re_patch/', data_path + '/data/nyu/val_re_patch/',
                       data_path + '/data/nyu/test_re_patch/']
    else:
        input_path = [data_path + '/data/cut_ntire_cycle/train_hazy_patch/',
                      data_path + '/data/cut_ntire_cycle/val_hazy_patch/']
        output_path = [data_path + '/data/cut_ntire_cycle/train_re_patch/',
                       data_path + '/data/cut_ntire_cycle/val_re_patch/']
        '''
        test_hazy_path = data_path + '/data/cut_ntire_cycle/test_hazy/'
        test_gth_path = data_path + '/data/cut_ntire_cycle/test_gth/'
        test_path_list = [test_hazy_path, test_gth_path]
        '''
    for i in range(len(input_path)):
        if not os.path.exists(output_path[i]):
            os.makedirs(output_path[i])

        test_data = test_DataSet(transform, path=input_path[i])
        test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

        start_time = time.time()
        count = 1
        save_path = output_path[i]
        for name, haze_image in test_data_loader:
            net.eval()
            with torch.no_grad():
                haze_image = haze_image.cuda()
                A, t, J_reconstruct = net(haze_image)
                for j in range(len(name)):
                    im_output_for_save = get_image_for_save(J_reconstruct[j])
                    filename = '{}.bmp'.format(name[j])
                    im_output_for_save.save(os.path.join(save_path, filename))
            print_test_time(start_time, count, len(test_data_loader))
            count += 1
else:
    if category == 'NYU':
        test_hazy_path = data_path + '/data/nyu/test_hazy_patch/'
        test_gth_path = data_path + '/data/nyu/test_gth_patch/'
        test_t_path = data_path + '/data/nyu/test_t_patch/'
        test_A_path = data_path + '/data/nyu/test_A_patch/'
        test_path_list = [test_hazy_path, test_gth_path, test_t_path, test_A_path]
    else:
        test_hazy_path = data_path + '/data/cut_ntire_cycle/test_hazy/'
        test_gth_path = data_path + '/data/cut_ntire_cycle/test_gth/'
        test_path_list = [test_hazy_path, test_gth_path]

    # 创建用于保存测试结果的文件夹
    local_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = './{}/test_result_{}'.format(file_path, local_time)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 创建用于保存测试指标的表格文件
    excel_test_line = 1
    excel_save = './{}/test_result_{}.xls'.format(file_path, local_time)
    f, sheet_test = init_test_excel(row=excel_row)

    # 创建图像数据加载器
    if category == 'NYU':
        test_data = train_DataSet(transform, path=test_path_list, flag='test')
    else:
        test_data = Ntire_DataSet(transform, path=test_path_list, flag='test')
    test_data_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=8)

    start_time = time.time()
    count = 1

    for name, haze_image, gt_image, A_gth, t_gth in test_data_loader:
        net.eval()
        with torch.no_grad():
            haze_image = haze_image.cuda()
            gt_image = gt_image.cuda()
            A_gth = A_gth.cuda()
            t_gth = t_gth.cuda()
            A, t, J_reconstruct = net(haze_image)
            loss_image = [A, A_gth, t, t_gth, J_reconstruct, gt_image]
            loss, temp_loss = loss_net(loss_image, weight)
            if Is_save_image:
                im_output_for_save = get_image_for_save(J_reconstruct)
                filename = '{}.bmp'.format(name[0])
                im_output_for_save.save(os.path.join(save_path, filename))
            excel_test_line = write_excel_test(sheet=sheet_test, line=excel_test_line,
                                               content=name_div(name[0]) + temp_loss)
            f.save(excel_save)
        print_test_time(start_time, count, len(test_data_loader))
        count += 1
