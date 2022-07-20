import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import xlrd
import pandas as pd

WIDTH = 120
HEIGHT = WIDTH


# npz_save_path = '/home/dell/susan/Substorm/npzs_199701_LBHS/'


def test_fuc():
    print('Hey bros!')


def image_initialize(image, img_size, img_chns):
    picture = Image.open(image)
    picture = picture.resize((img_size, img_size), Image.ANTIALIAS)
    if img_chns == 3:
        picture = picture.convert('L')
    data = np.array(picture.getdata()).reshape(img_size, img_size, img_chns)
    data = data.astype(np.float32) / 255

    return data


def get_npz(img_size, data_path, path, npz_save_path, img_chns):
    num = 0
    img_list = np.array([])
    file_list = os.listdir(os.path.join(data_path, path))
    file_list.sort(key=lambda x: int(x[:-4]))
    for i in range(len(file_list)):
        image_array = image_initialize(os.path.join(os.path.join(data_path, path), file_list[i]), img_size, img_chns)
        img_list = np.append(img_list, image_array)
        num += 1
    img_list = img_list.reshape(num, img_size * img_size * img_chns)
    np.savez(npz_save_path + path + '.npz', images=img_list)
    # print('new npz')
    # print(path, '.npz', ' saved.')


# 读npz，生成basic_seq和对应的next_seq，basic_seq用于聚类
# 用于一帧一帧的训练
def get_sequence(npz_path, npz_file, basic_frames, interval_frames, img_size):
    raw_seq = np.load(os.path.join(npz_path, npz_file))['images']  # load array
    raw_len = raw_seq.shape[0]
    raw_seq = raw_seq.reshape(raw_len, img_size, img_size, 1)
    #     raw_seq = np.dot(raw_seq[...,:3], [0.299, 0.587, 0.114])
    print('raw:', raw_seq.shape)

    basic_len = raw_len - interval_frames
    next_len = basic_len

    basic_imgs = raw_seq[:basic_len]
    next_imgs = raw_seq[basic_frames:]

    basic_imgs = basic_imgs.reshape(basic_imgs.shape[0], img_size * img_size)
    next_imgs = next_imgs.reshape(next_imgs.shape[0], img_size * img_size)
    print(npz_file, ' generated:', basic_imgs.shape, ' ', next_imgs.shape)
    return basic_imgs, next_imgs


# 用于多帧的训练
def get_3chn_seq(npz_path, npz_file, basic_frames, interval_frames, img_size, img_chns):
    raw_seq = np.load(os.path.join(npz_path, npz_file))['images']  # load array
    raw_len = raw_seq.shape[0]
    raw_seq = raw_seq.reshape(raw_len, img_size, img_size, img_chns)

    basic_len = raw_len - basic_frames - interval_frames + 1
    next_len = basic_len
#     print('basic_frames:', basic_frames)
#     print('interval_frames:', interval_frames)
#     print('raw_len:', raw_len)
#     print('basic_len:', basic_len)
    basic_seq = np.zeros((basic_len, basic_frames, img_size, img_size, img_chns))
    next_seq = np.zeros((next_len, basic_frames, img_size, img_size, img_chns))

    for i in range(basic_frames):
        basic_seq[:, i, :, :] = raw_seq[i:i + basic_len]
        next_seq[:, i, :, :] = raw_seq[i + interval_frames:i + basic_len + interval_frames]

    # basic_seq = basic_seq.reshape(basic_seq.shape[0], basic_frames * HEIGHT * WIDTH * 3)
    # next_seq = next_seq.reshape(next_seq.shape[0], basic_frames * HEIGHT * WIDTH * 3)
    # print(npz_file, ' generated:', basic_seq.shape, ' ', next_seq.shape)
    return basic_seq, next_seq


def para_maxmin(src_path, col_idx, start_row, end_row):
    para_list = np.array([])

    excel_list = os.listdir(src_path)
    excel_list.sort()

    for i in range(len(excel_list)):

        excel = excel_list[i]
        # print('excel:', excel)
        workbook = xlrd.open_workbook(os.path.join(src_path, excel))
        sheet_list = workbook.sheet_names()
        sheet_list.sort()

        for j in range(len(sheet_list)):
            sheet = workbook.sheet_by_name(sheet_list[j])
            para_list = np.append(para_list, sheet.col_values(col_idx, start_row, end_row))

            # print('sheet:', sheet_list[j])
            # print(para_list.shape)
    print(para_list.shape)
    # print(para_list)
    # for i in range(len(para_list)):
    #     tmp = para_list[i]
    #     para_list[i] = float(tmp)
    # para_list = int(para_list)
    print(para_list)
    para_array = para_list.astype(np.float32)
    para_min = para_array.min()
    para_max = para_array.max()

    return para_array, para_min, para_max


def para_normalization(sheet, col_idx, start_row, end_row, size):
    para_array = np.array(sheet.col_values(col_idx, start_row, end_row))
    para_array = para_array.astype(np.float32)
    para_min = para_array.min()
    para_max = para_array.max()

    # normalization
    for i in range(len(para_array)):
        para_array[i] = (para_array[i] - para_min) / (para_max - para_min)

    para_array.resize([len(para_array), 1])
    temp_array = para_array

    for j in range(size - 1):
        para_array = np.hstack((para_array, temp_array))

    # para_array.resize([len(para_array) * size * size,])
    return para_array


# para_normalization v2.0
def normalization(para_min, para_max, para_array, size):
    # normalization
    for i in range(len(para_array)):
        para_array[i] = (para_array[i] - para_min) / (para_max - para_min)

    para_array.resize([len(para_array), 1])
    temp_array = para_array

    for j in range(size - 1):
        para_array = np.hstack((para_array, temp_array))
    # print(para_array.shape)
    return para_array


def para_sequence(para, basic_frames, interval_frames, hidden_dim):
    # raw_seq = np.load(os.path.join(npz_path, npz_file))['sequence_array']  # load array
    raw_seq = para
    raw_len = raw_seq.shape[0]
    # print('raw_len:', raw_len)
    raw_seq = raw_seq.reshape(raw_len, hidden_dim, hidden_dim)

    basic_len = raw_len - basic_frames - interval_frames + 1
    next_len = basic_len
    basic_seq = np.zeros((basic_len, basic_frames, hidden_dim, hidden_dim))
    next_seq = np.zeros((next_len, basic_frames, hidden_dim, hidden_dim))

    for i in range(basic_frames):
        basic_seq[:, i, :, :] = raw_seq[i:i + basic_len]
        next_seq[:, i, :, :] = raw_seq[i + interval_frames:i + basic_len + interval_frames]

    basic_seq = basic_seq.reshape(basic_seq.shape[0], basic_frames, hidden_dim, hidden_dim, 1)
    next_seq = next_seq.reshape(next_seq.shape[0], basic_frames, hidden_dim, hidden_dim, 1)
    # print(npz_file, ' generated:', basic_seq.shape, ' ', next_seq.shape)
    return basic_seq, next_seq
    # return basic_seq


# 对于多段视频的单一参数分别归一化后生成序列，并组成总数组
def para_seq(src_path, para_min, para_max, col_idx, start_row, end_row, basic_frames, interval_frames, dim):
    excel_list = os.listdir(src_path)
    excel_list.sort()
    for i in range(len(excel_list)):

        excel = excel_list[i]
        # print('excel:', excel)
        workbook = xlrd.open_workbook(os.path.join(src_path, excel))
        sheet_list = workbook.sheet_names()
        sheet_list.sort()

        for j in range(len(sheet_list)):
            sheet = workbook.sheet_by_name(sheet_list[j])
            para_array = np.array(sheet.col_values(col_idx, start_row, end_row))
            para_array = normalization(para_min, para_max, para_array, dim * dim)
            print('para_array:', para_array.shape)
            # 取多min间隔
            #             new_paras = np.array([])
            #             for k in range(20):
            #                 new_paras = np.append(new_paras, para_array[(k + 1) * 6 - 1])

            #             para_array = new_paras.reshape((20, 64))
            # print('new_paras:', para_array.shape)
            basic_para, next_para = para_sequence(para_array, basic_frames, interval_frames, dim)
            # print(basic_para.shape)

            if i == 0 and j == 0:
                # print('first round')
                total_basic = basic_para
                total_next = next_para
            else:
                total_basic = np.vstack((total_basic, basic_para))
                total_next = np.vstack((total_next, next_para))
            # print('sheet:', sheet_list[j])
            # print('basic_ae:', basic_ae.shape)
            # print(total_para.shape)

    return total_basic, total_next


def get_paras(para_path, basic_frames, interval_frames, hidden_dim, para_chns, filters=1):
    total_basic = np.array([])
    total_next = np.array([])
    for k in range(len(para_path)):
        data = pd.read_excel(para_path[k], sheet_name=0)
        para_list = data.values[:120, 1:]
        para_max = np.max(para_list)
        para_min = np.min(para_list)

        para_array = np.array([])
        for paras in para_list:
            new_paras = np.array([])
            for para in paras:
                new_para = (para - para_min) / (para_max - para_min)
                new_paras = np.append(new_paras, new_para)
            para_array = np.append(para_array, new_paras)

        para_array = para_array.reshape((120, 93)).T

        basic_paras = []
        next_paras = []
        for paras in para_array:
            paras = paras.reshape(paras.shape[0], 1)
            basic_len = paras.shape[0] - basic_frames - interval_frames + 1
            basic_seq = np.zeros((basic_len, basic_frames, 1))
            next_seq = np.zeros((basic_len, basic_frames, 1))

            for i in range(basic_frames):
                basic_seq[:, i, :] = paras[i:i + basic_len]
                next_seq[:, i, :] = paras[i + interval_frames:i + basic_len + interval_frames]

            next_paras.append(next_seq)
            basic_paras.append(basic_seq)

        basic_paras = np.array(basic_paras)
        next_paras = np.array(next_paras)
        print(basic_paras.shape)
        basic_paras = basic_paras.reshape((basic_paras.shape[0] * basic_paras.shape[1], basic_paras.shape[2], 1))
        next_paras = next_paras.reshape((next_paras.shape[0]*next_paras.shape[1], next_paras.shape[2], 1))

        new_basic = []
        new_next = []
        for i in range(basic_paras.shape[0]):
            new_seq = []
            new_next_seq = []
            for j in range(basic_paras.shape[1]):
                para = np.tile(basic_paras[i, j], (hidden_dim, hidden_dim, filters))
                new_seq.append(para)

                next_para = np.tile(next_paras[i,j], (hidden_dim, hidden_dim, filters))
                new_next_seq.append(next_para)
            new_seq = np.array(new_seq)
            new_next_seq = np.array(new_next_seq)
            #     print(new_seq.shape)
            new_basic.append(new_seq)
            new_next.append(new_next_seq)
        new_basic = np.array(new_basic)
        new_next = np.array(new_next)

        print(k)
        if k == 0:
            total_basic = new_basic
            total_next = new_next
        else:
            total_basic = np.concatenate((total_basic, new_basic), -1)
            total_next = np.concatenate((total_next, new_next), -1)

    return total_basic, total_next
