import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.font_manager import FontProperties
import skimage.measure as ms
from multiprocessing import Pool
import os

def pr_curve_out_noname(dir = 'tmp', mode = 'ods'):
    p = None
    r = None

    if mode == 'ods':
        total_num = np.zeros([100, 4])

        name_list = os.listdir(dir)
        for name in name_list:
            each_num = np.load(dir + '/' + name)
            total_num = total_num + each_num

        p = total_num[:, 0] / total_num[:, 1]
        r = total_num[:, 2] / total_num[:, 3]

    elif mode == 'ois':
        total_pr = []
        name_list = os.listdir(dir)
        for name in name_list:
            pr_tmp = np.zeros([100, 2])
            each_num = np.load(dir + '/' + name)
            p_each = each_num[:, 0] / each_num[:, 1]
            r_each = each_num[:, 2] / each_num[:, 3]
            pr_tmp[:, 0] = p_each
            pr_tmp[:, 1] = r_each
            total_pr.append(pr_tmp)
        total_pr = np.array(total_pr)
        total_pr = np.mean(total_pr,axis=0)
        p = total_pr[:,0]
        r = total_pr[:,1]

    p_filt = []
    r_filt = []
    p_tmp = -1
    for idx, p_each in enumerate(p):
        if p_each >= p_tmp:
            p_filt.append(p_each)
            r_filt.append(r[idx])
            p_tmp = p_each
    p_filt = np.expand_dims(np.array(p_filt), 1)
    r_filt = np.expand_dims(np.array(r_filt), 1)

    f = 2 * p_filt * r_filt / (p_filt + r_filt)
    pr = np.hstack([r_filt, p_filt])
    f_max_idx = np.argmax(f)
    f_score = np.max(f)

    print(dir + ':' + str(f_score))

    plt.figure()
    plt.plot(pr[:,0],pr[:,1],linewidth = '1')
    plt.scatter(pr[f_max_idx, 0], pr[f_max_idx, 1], alpha=0.5, marker='*')  # s为size，按每个点的坐标绘制，alpha为透明度

    plt.show()


def label_metric(arg):

    def bound_metrix(pred, gt, threhold):
        ther_pre = threhold
        min_thr = 0
        max_thr = 100
        interval = 1
        pred = pred.astype(np.float)
        gt = gt.astype(np.float)
        pred = pred / 255.0 * max_thr
        gt = gt / 255.0

        def calculate_pr(gan, gt):

            tp = 0
            all_p = np.where(gan == 1)[0].shape[0]
            tn = 0
            all_n = np.where(gt == 1)[0].shape[0]
            for i in range(gan.shape[0]):
                for j in range(gan.shape[0]):
                    if gan[i, j] == 0:
                        continue
                    re_x_min = max(0, i - int(ther_pre))
                    re_x_max = min(255, i + int(ther_pre)) + 1
                    re_y_min = max(0, j - int(ther_pre))
                    re_y_max = min(255, j + int(ther_pre)) + 1
                    out = gt[re_x_min:re_x_max, re_y_min:re_y_max]
                    x_ = min(int(ther_pre), i)
                    y_ = min(int(ther_pre), j)
                    label = False
                    for ii in range(out.shape[0]):
                        if not label:
                            for jj in range(out.shape[1]):
                                if out[ii, jj] == 1 and (ii - x_) ** 2 + (jj - y_) ** 2 < ther_pre ** 2:
                                    tp = tp + 1
                                    label = True
                                    break
                                else:
                                    p = 0
            if all_p == 0:
                gan_precion = 0
            else:
                gan_precion = tp / all_p

            for i in range(gt.shape[0]):
                for j in range(gt.shape[0]):
                    if gt[i, j] == 0:
                        continue
                    re_x_min = max(0, i - int(ther_pre))
                    re_x_max = min(255, i + int(ther_pre)) + 1
                    re_y_min = max(0, j - int(ther_pre))
                    re_y_max = min(255, j + int(ther_pre)) + 1
                    out = gan[re_x_min:re_x_max, re_y_min:re_y_max]
                    x_ = min(int(ther_pre), i)
                    y_ = min(int(ther_pre), j)
                    label = False
                    for ii in range(out.shape[0]):
                        if not label:
                            for jj in range(out.shape[1]):
                                if out[ii, jj] == 1 and (ii - x_) ** 2 + (jj - y_) ** 2 < ther_pre ** 2:
                                    tn = tn + 1
                                    label = True
                                    break
            if all_n == 0:
                gan_recall = 0
            else:
                gan_recall = tn / all_n

            #print('precison:' + str(gan_precion) + ' recall:' + str(gan_recall))
            return tp, all_p, tn, all_n #gan_precion, gan_recall

        pr = np.zeros([100, 4])

        for thre in range(min_thr, max_thr, interval):
            pred_ = np.zeros(pred.shape)
            pred_[np.where(pred >= thre)] = 1
            tp, all_p, tn, all_n = calculate_pr(pred_, gt)
            pr[thre, 0] = tp
            pr[thre, 1] = all_p
            pr[thre, 2] = tn
            pr[thre, 3] = all_n
        return pr

    gt_name, tar_name, save_dir = arg
    img_name = gt_name.split('/')[-1].split('.')[0]
    gt_image = Image.open(gt_name)
    gt_image = np.array(gt_image)
    pred_image = Image.open(tar_name)
    pred_image = np.array(pred_image)
    pr = bound_metrix(pred_image, gt_image, threhold = 2.1)
    np.save(save_dir + '/' + img_name, pr)
    print(img_name + ' OK!')


def label_metric_mask(arg):

    def bound_metrix(pred, gt, threhold):
        ther_pre = threhold
        min_thr = 0
        max_thr = 100
        interval = 1
        pred = pred.astype(np.float)
        gt = gt.astype(np.float)
        pred = pred / 255.0 * max_thr
        gt = gt / 255.0

        def calculate_pr(gan, gt):

            tp = 0
            all_p = np.where(gan == 1)[0].shape[0]
            tn = 0
            all_n = np.where(gt == 1)[0].shape[0]
            for i in range(gan.shape[0]):
                for j in range(gan.shape[0]):
                    if gan[i, j] == 0:
                        continue
                    re_x_min = max(0, i - int(ther_pre))
                    re_x_max = min(255, i + int(ther_pre)) + 1
                    re_y_min = max(0, j - int(ther_pre))
                    re_y_max = min(255, j + int(ther_pre)) + 1
                    out = gt[re_x_min:re_x_max, re_y_min:re_y_max]
                    x_ = min(int(ther_pre), i)
                    y_ = min(int(ther_pre), j)
                    label = False
                    for ii in range(out.shape[0]):
                        if not label:
                            for jj in range(out.shape[1]):
                                if out[ii, jj] == 1 and (ii - x_) ** 2 + (jj - y_) ** 2 < ther_pre ** 2:
                                    tp = tp + 1
                                    label = True
                                    break
                                else:
                                    p = 0
            if all_p == 0:
                gan_precion = 0
            else:
                gan_precion = tp / all_p

            for i in range(gt.shape[0]):
                for j in range(gt.shape[0]):
                    if gt[i, j] == 0:
                        continue
                    re_x_min = max(0, i - int(ther_pre))
                    re_x_max = min(255, i + int(ther_pre)) + 1
                    re_y_min = max(0, j - int(ther_pre))
                    re_y_max = min(255, j + int(ther_pre)) + 1
                    out = gan[re_x_min:re_x_max, re_y_min:re_y_max]
                    x_ = min(int(ther_pre), i)
                    y_ = min(int(ther_pre), j)
                    label = False
                    for ii in range(out.shape[0]):
                        if not label:
                            for jj in range(out.shape[1]):
                                if out[ii, jj] == 1 and (ii - x_) ** 2 + (jj - y_) ** 2 < ther_pre ** 2:
                                    tn = tn + 1
                                    label = True
                                    break
            if all_n == 0:
                gan_recall = 0
            else:
                gan_recall = tn / all_n

            #print('precison:' + str(gan_precion) + ' recall:' + str(gan_recall))
            return tp, all_p, tn, all_n #gan_precion, gan_recall

        pr = np.zeros([100, 4])

        for thre in range(min_thr, max_thr, interval):
            pred_ = np.zeros(pred.shape)
            pred_[np.where(pred >= thre)] = 1
            tp, all_p, tn, all_n = calculate_pr(pred_, gt)
            pr[thre, 0] = tp
            pr[thre, 1] = all_p
            pr[thre, 2] = tn
            pr[thre, 3] = all_n
        return pr

    gt_name, tar_name, mask_name, save_dir = arg
    img_name = gt_name.split('/')[-1].split('.')[0]
    gt_image = Image.open(gt_name)
    gt_image = np.array(gt_image)
    pred_image = Image.open(tar_name)
    pred_image = np.array(pred_image)
    mask_image = Image.open(mask_name)
    mask_image = np.array(mask_image)
    gt_image[np.where(mask_image == 255)] = 0
    pred_image[np.where(mask_image == 255)] = 0

    pr = bound_metrix(pred_image, gt_image, threhold = 2.1)
    np.save(save_dir + '/' + img_name, pr)
    print(img_name + ' OK!')

def build_pr_curve(dirs, mode = 'ods'):

    colors_plt = ['blue', 'deepskyblue', 'yellowgreen', 'green', 'red', 'orange']
    name_map = []
    f_idx = []
    pr_merge = []
    methods = []
    fig = plt.figure(figsize=(6, 6), dpi=300)
    for dir_idx, dir in enumerate(dirs):
        methods.append(dir)
        names = os.listdir(dir)
        total_num = np.zeros([100, 4])
        for name in names:
            if mode == 'ods':
                name_list = os.listdir(dir)
                for name in name_list:
                    each_num = np.load(dir + '/' + name)
                    total_num = total_num + each_num
                p = total_num[:, 0] / total_num[:, 1]
                r = total_num[:, 2] / total_num[:, 3]

            elif mode == 'ois':
                total_pr = []
                name_list = os.listdir(dir)
                for name in name_list:
                    pr_tmp = np.zeros([100, 2])
                    each_num = np.load(dir + '/' + name)
                    p_each = each_num[:, 0] / each_num[:, 1]
                    r_each = each_num[:, 2] / each_num[:, 3]
                    pr_tmp[:, 0] = p_each
                    pr_tmp[:, 1] = r_each
                    total_pr.append(pr_tmp)
                total_pr = np.array(total_pr)
                total_pr = np.mean(total_pr, axis=0)
                p = total_pr[:, 0]
                r = total_pr[:, 1]

        p_filt = []
        r_filt = []
        p_tmp = -1
        for idx, p_each in enumerate(p):
            if idx == len(p) - 1:
                continue
            if p_each >= p_tmp:
                p_filt.append(p_each)
                r_filt.append(r[idx])
                p_tmp = p_each
        p_filt = np.expand_dims(np.array(p_filt), 1)
        r_filt = np.expand_dims(np.array(r_filt), 1)
        f = 2 * p_filt * r_filt / (p_filt + r_filt)
        pr = np.hstack([p_filt, r_filt])
        f_max_idx = np.argmax(f)
        f_score = np.max(f).round(4)
        f_score = str(f_score)
        if len(f_score) == 5:
            f_score = f_score +'0'
        f_idx.append(f_max_idx)
        pr_merge.append(pr)
        lab = '[F1=' + f_score +']' + dir.split('_')[0]
        name_map.append(lab)
        print(dir)
        print(f_max_idx)
        print(lab)
        print(p_filt[f_max_idx])
        print(r_filt[f_max_idx])
        plt.plot(pr[:, 1], pr[:, 0], linewidth='1', label=lab, color=colors_plt[dir_idx])
        plt.scatter(pr[f_max_idx, 1], pr[f_max_idx, 0], s=150, c=colors_plt[dir_idx], alpha=0.5, marker='*')  # s为size，按每个点的坐标绘制，alpha为透明度

    if 'mask' in dir:
        plt.xlabel("Recall(Mask)", fontsize=20)
        plt.ylabel("Precision(Mask)", fontsize=20)
    else:
        plt.xlabel("Recall", fontsize=20)
        plt.ylabel("Precision", fontsize=20)
    plt.legend(loc='upper left',fontsize=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.grid()
    plt.xlim([0,1])
    plt.ylim([0,1])
    #plt.title('Performance of semiRCF Vs (Revised)semiRCF_eval')
    #plt.title('Measurement Based on New Criterion', fontsize = 20)
    plt.savefig('olc.png', bbox_inches='tight')
    #plt.show()


if __name__ == '__main__':
    dirs = []
    dirs = ['Canny', 'gPb-OWT-UCM', 'RCF', 'BDCN', 'SemiRCF-GAN','SemiRCF-MMD']

    gt_dir_out = '../data_metal/label_test/'
    mask_dir_out = '../data_metal/mask_test/'
    multi_process_num = 12

    '''
    for dir_name in dirs:
        save_dir = dir_name + '_eval'
        #save_dir = dir_name + '_mask_eval'

        name_lst = []
        name_list = os.listdir(gt_dir_out)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for name in name_list:
            if not os.path.exists(save_dir + '/' + name.split('/')[-1].split('.')[0] + '.npy'):
                name_lst.append((gt_dir_out + '/' + name, dir_name + '/' + name, save_dir))
                #name_lst.append((gt_dir_out + '/' + name, dir_name + '/' + name, mask_dir_out + '/' + name, save_dir))

        pool = Pool(multi_process_num)
        pool.map(label_metric, name_lst)
        #pool.map(label_metric_mask, name_lst)
        pool.close()
        pool.join()

        pr_curve_out_noname(save_dir, mode='ods')

    '''
    dirs = ['Canny_eval', 'gPb-OWT-UCM_eval', 'RCF_eval', 'BDCN_eval', 'SemiRCF-GAN_eval','SemiRCF-MMD_eval']
    #dirs = ['Canny_mask_eval', 'gPb-OWT-UCM_mask_eval', 'RCF_mask_eval', 'BDCN_mask_eval', 'SemiRCF-GAN_mask_eval','SemiRCF-MMD_mask_eval']

    build_pr_curve(dirs, mode = 'ods')


