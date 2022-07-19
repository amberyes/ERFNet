import xlwt
import time
import os
import numpy as np
from PIL import Image
import random
import math

"""
def get_args(parser):


    log = 'learning_rate: {}\n' \
          'batch_size: {}\n' \
          'epoch: {}\n' \
          'drop_rate: {}\n' \
          'category: {}\n' \
          'loss_weight: {}\n' \
          'Is_pre_model: {}\n' \
          'total_params: {}\n' \
          'save_file_name: {}\n' \
          'MAE_or_MSE: {}\n' \
          'Is_inter_train: {}\n' \
          'accumulation_steps: {}\n' \
          'norm_type: {}'.\
        format(learning_rate,
               batch_size,
               epoch,
               drop_rate,
               category,
               weight,
               Is_pre_model,
               total_params,
               save_path,
               MAE_or_MSE,
               Is_inter_train,
               accumulation_steps,
               norm_type)
    print(log)
    with open(save_path + 'detail.txt', 'w') as f:
        f.write(log)
"""


def find_n(l, n):
    """
    从l查找是否包含n作为子串的元素，返回所有元素
    """
    l1 = l.copy()
    num = l1.count(n)
    if num != 2:
        print(num)
    r = []
    for i in range(num):
        r.append(l1.index(n) + i)
        l1.remove(n)
    return r


def fiveK_name_div(name):
    return name


def data_aug(img):
    a = random.random()
    b = math.floor(random.random() * 4)
    if a >= 0.5:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.FLIP_LEFT_RIGHT)
    if b == 1:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.ROTATE_90)
    elif b == 2:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.ROTATE_180)
    elif b == 3:
        for i in range(len(img)):
            img[i] = img[i].transpose(Image.ROTATE_270)
    return img


def get_train_path(net_name, data_path, category):
    train_haze_path = data_path + 'data/' + category + 'train_haze/'
    val_haze_path = data_path + 'data/' + category + 'val_haze/'

    # train_haze_path = '/media/liu/新加卷/去雾数据集/RESIDE/OTS/train_haze/'
    # val_haze_path = '/media/liu/新加卷/去雾数据集/RESIDE/OTS/val_haze/'

    gth_path = data_path + 'data/' + category + 'gt/'

    # gth_path = '/media/liu/新加卷/去雾数据集/RESIDE/OTS/gt/'

    train_path_list = [train_haze_path, gth_path]
    val_path_list = [val_haze_path, gth_path]

    name_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = '%strain_result/%s_result_%s/' % (data_path, net_name, name_time)
    save_model_name = '%s%s_model.pt' % (save_path, net_name)  # 保存模型的路径
    excel_save = save_path + 'result.xls'  # 保存excel的路径
    mid_save_path = './mid_model/%s_model.pt' % net_name  # 保存的中间模型，用于意外停止后继续训练
    return train_path_list, val_path_list, save_path, save_model_name, excel_save, mid_save_path


def count_params(net):
    """
    打印net网络参数量
    :param net:
    :return:
    """
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: {}".format(total_params))


def create_dir(path):
    """
    是否存在路径path，不存在则创建
    :param path:
    :return:
    """
    if not os.path.exists(path):
        print("创建%s" % path)
        os.makedirs(path)


# 转换需要保存的图像
def get_image_for_save(img):
    img = img.cpu()
    img = img.numpy()
    img = np.squeeze(img)
    img = img * 255
    img[img < 0] = 0
    img[img > 255] = 255
    print(img.shape)
    if len(img.shape) == 2:
        img = np.tile(img, (3, 1, 1))
        print(img.shape)
    img = np.rollaxis(img, 0, 3)
    img = img.astype('uint8')
    img = Image.fromarray(img).convert('RGB')
    return img


# 查找训练好的模型文件
def find_pretrain(path_name):
    file_list = os.listdir('./')
    length = len(path_name)
    for i in range(len(file_list)):
        if file_list[i][:length] == path_name:
            return file_list[i]
    return 0


def set_style(name, height, bold=False):
    style = xlwt.XFStyle()
    font = xlwt.Font()
    font.name = name
    font.bold = bold
    font.color_index = 4
    font.height = height
    style.font = font
    return style


def write_excel_train(sheet, line, epoch, itr, loss, weight):
    sum_loss = 0
    sheet.write(line, 0, epoch + 1)
    sheet.write(line, 1, itr + 1)
    for i in range(len(loss)):
        sheet.write(line, i + 2, round(loss[i], 6))
        sum_loss += loss[i] * weight[i]
    sheet.write(line, 2 + len(loss), round(sum_loss, 6))
    return line + 1


def write_excel_val(sheet, line, epoch, loss):
    loss, val, train = loss
    sheet.write(line, 0, epoch + 1)
    for i in range(len(loss)):
        sheet.write(line, i + 1, round(loss[i], 6))
    sheet.write(line, 1 + len(loss), round(val, 6))
    sheet.write(line, 2 + len(loss), round(train, 6))
    return line + 1


def write_excel_every_val(sheet, line, epoch, name, loss):
    sheet.write(line, 0, epoch + 1)
    num = int(name[:4])
    if len(name) == 4:
        air_light = 0.0
        beta = 0.0
    else:
        air_light = float(name[-11:-7])
        beta = float(name[-4:])
    sheet.write(line, 1, num)
    sheet.write(line, 2, air_light)
    sheet.write(line, 3, beta)
    for i in range(len(loss)):
        sheet.write(line, i + 4, round(loss[i], 6))
    return line + 1


def fiveK_name_div(name):
    return [name]


def name_div(name):
    num = int(name.split('_')[0])
    if len(name) == 5:
        air_light = 0.0
        beta = 0.0
    else:
        air_light = float(name.split('_')[1].split('=')[1])
        beta = float(name.split('_')[2].split('=')[1])
    return [num, air_light, beta]


def write_excel_test(sheet, line, content):
    for i in range(len(content)):
        sheet.write(line, i, content[i])
    return line + 1


def init_train_excel(row):
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('train', cell_overwrite_ok=True)
    sheet2 = workbook.add_sheet('val', cell_overwrite_ok=True)
    print('写入train_excel')
    for i in range(0, len(row[0])):
        sheet1.write(0, i, row[0][i], set_style('Times New Roman', 220, True))
    print('写入val_excel')
    for i in range(0, len(row[1])):
        sheet2.write(0, i, row[1][i], set_style('Times New Roman', 220, True))
    return workbook, sheet1, sheet2


def init_test_excel(row):
    workbook = xlwt.Workbook()
    sheet1 = workbook.add_sheet('test', cell_overwrite_ok=True)
    # 通过excel保存训练结果（训练集验证集loss，学习率，训练时间，总训练时间）
    print('写入test_excel')
    for i in range(0, len(row)):
        sheet1.write(0, i, row[i], set_style('Times New Roman', 220, True))
    return workbook, sheet1


def print_time(start, progress, epoch, total, n_epoch):
    """
    :param start: 训练开始时间
    :param progress: 当前轮的进度
    :param epoch: 总轮数
    :param total: 当前轮的总批数
    :param n_epoch: 当前第几轮
    需要打印，到目前为止已经花费的时间，训练结束需要的时间。
    """
    # print("start:%d\nprogress:%d\nepoch:%d\ntotal:%d\nn_epoch:%d\n", start, progress, epoch, total, n_epoch)
    now = time.time()
    epoch_time = now - start
    etr_time = (now - start) / (n_epoch * total + progress) * epoch * total - epoch_time

    m, s = divmod(epoch_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("spend time: %d:%02d:%02d:%02d" % (d, h, m, s))
    m, s = divmod(etr_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("Estimated time remaining: %d:%02d:%02d:%02d\n" % (d, h, m, s))


def print_test_time(start, count, total):
    now = time.time()
    epoch_time = now - start
    etr_time = (now - start) / count * (total - count)

    m, s = divmod(epoch_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)
    print("spend time: %d:%02d:%02d:%02d" % (d, h, m, s))
    m, s = divmod(etr_time, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 60)
    print("Estimated time remaining: %d:%02d:%02d:%02d\n" % (d, h, m, s))
