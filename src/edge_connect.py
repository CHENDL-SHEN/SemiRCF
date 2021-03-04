import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from .dataset import Dataset,Dataset_unlabel,Dataset_semi,Dataset_out
from .models import EdgeModel, build_label_process
from .utils import Progbar, create_dir, stitch_images, imsave
from .metrics import PSNR, EdgeAccuracy
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import functional as FF
import cv2
import math
from scipy import signal
from multiprocessing import Pool


class EdgeConnect():
    def __init__(self, config):
        self.config = config

        if config.MODEL == 1:
            model_name = 'edge'

        self.model_name = model_name
        self.edge_model = EdgeModel(config).to(config.DEVICE)
        self.MAX_ITERS = config.MAX_ITERS

        self.edgeacc = EdgeAccuracy(config.EDGE_THRESHOLD).to(config.DEVICE)
        self.train_dataset = Dataset(config, config.TARGET_LAB_IMG_FLIST, config.TARGET_LAB_EDGE_FLIST, config.TARGET_LAB_MASK_FLIST, augment=True, training=True,  mode = self.config.data_mode)
        self.train_unlab_dataset = Dataset(config, config.TARGET_UNLAB_IMG_FLIST, config.TARGET_UNLAB_EDGE_FLIST, config.TARGET_UNLAB_MASK_FLIST, augment=True, training=True,  mode = self.config.data_mode) # eval 时用
        self.train_unlab_pseudo_dataset = Dataset_semi(config, config.TARGET_UNLAB_IMG_FLIST, config.TARGET_UNLAB_EDGE_Pseudo_FLIST, config.TARGET_UNLAB_MASK_FLIST, augment=True, training=True,  mode = self.config.data_mode)
        self.train_dataset_source = Dataset(config, config.SOURCE_LAB_IMG_FLIST, config.SOURCE_LAB_EDGE_FLIST, config.SOURCE_LAB_MASK_FLIST, augment=True, training=True,  mode = self.config.data_mode)



        if self.config.arg_train:
            self.train_dataset_arg = Dataset(config, config.TARGET_LAB_IMG_FLIST.replace(config.TARGET_LAB_IMG_FLIST.split('/')[-1],'arg_'+config.TARGET_LAB_IMG_FLIST.split('/')[-1]),
                                         config.TARGET_LAB_EDGE_FLIST.replace(config.TARGET_LAB_EDGE_FLIST.split('/')[-1],'arg_'+config.TARGET_LAB_EDGE_FLIST.split('/')[-1]),
                                         config.TARGET_LAB_MASK_FLIST.replace(config.TARGET_LAB_MASK_FLIST.split('/')[-1],'arg_'+config.TARGET_LAB_MASK_FLIST.split('/')[-1]),
                                         augment=True, training=True,  mode = self.config.data_mode)

            self.train_unlab_pseudo_dataset_arg = Dataset_semi(config, config.TARGET_UNLAB_IMG_FLIST.replace(config.TARGET_UNLAB_IMG_FLIST.split('/')[-1],'arg_'+config.TARGET_UNLAB_IMG_FLIST.split('/')[-1]),
                                         config.TARGET_UNLAB_EDGE_Pseudo_FLIST,
                                         config.TARGET_UNLAB_MASK_FLIST.replace(config.TARGET_UNLAB_MASK_FLIST.split('/')[-1],'arg_'+config.TARGET_UNLAB_MASK_FLIST.split('/')[-1]),
                                         augment=True, training=True,  mode = self.config.data_mode)

            self.train_dataset_source_arg = Dataset(config, config.SOURCE_LAB_IMG_FLIST.replace(config.SOURCE_LAB_IMG_FLIST.split('/')[-1],'arg_'+config.SOURCE_LAB_IMG_FLIST.split('/')[-1]),
                                         config.SOURCE_LAB_EDGE_FLIST.replace(config.SOURCE_LAB_EDGE_FLIST.split('/')[-1],'arg_'+config.SOURCE_LAB_EDGE_FLIST.split('/')[-1]),
                                         config.SOURCE_LAB_MASK_FLIST.replace(config.SOURCE_LAB_MASK_FLIST.split('/')[-1],'arg_'+config.SOURCE_LAB_MASK_FLIST.split('/')[-1]),
                                         augment=True, training=True,  mode = self.config.data_mode)
        else:
            self.train_dataset_arg = self.train_dataset
            self.train_unlab_pseudo_dataset_arg = self.train_unlab_pseudo_dataset
            self.train_dataset_source_arg = self.train_dataset_source



        self.samples_path = os.path.join(config.PATH, 'samples')
        self.samples_test_path = os.path.join(config.PATH, 'samples_test')
        self.results_path = os.path.join(config.PATH, 'results')
        self.out_results_path_real = os.path.join(config.PATH, 'results_out')

        self.log_file = os.path.join(config.PATH, 'log_' + model_name + '.dat')
        self.gene_pse()

    def load(self):
        if self.config.MODEL == 1:
            self.edge_model.load()


    def gene_pse(self):

        if not os.path.exists(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST):
            os.makedirs(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST)

        dir = self.config.TARGET_UNLAB_IMG_FLIST
        names_img_dir = os.listdir(dir)
        for i in names_img_dir:
            if not os.path.exists(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + i):
                gt = cv2.imread(dir + '/' + i, 0)
                gt[:, :] = 128
                cv2.imwrite(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + i, gt)
                print('Making init file for '+ str(i))

            if (not os.path.exists(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + i.replace('.png','_mask.png'))) and (self.config.data_mode == 'rgb'):
                mask = cv2.imread(self.config.TARGET_UNLAB_MASK_FLIST + '/' + i, 0)
                ones_8 = np.array([-1, -1, -1, -1, 8, -1, -1, -1, -1]).reshape([3, 3])
                ones_tuo = np.ones([2, 2])
                mask_div = signal.convolve2d(mask, ones_8, boundary='fill', mode='same')
                mask_div[np.where(mask_div != 0)] = 1
                mask_div = signal.convolve2d(mask_div, ones_tuo, boundary='fill', mode='same')
                mask_div[np.where(mask_div > 0)] = 255
                mask_div = mask_div.astype(np.uint8)
                print(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + i.replace('.png','_mask.png'))
                cv2.imwrite(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + i.replace('.png','_mask.png'), mask_div)
                print('Making mask file for '+ str(i))


        if self.config.arg_train:
            dir = self.config.TARGET_UNLAB_IMG_FLIST.replace(self.config.TARGET_UNLAB_IMG_FLIST.split('/')[-1],'arg_'+self.config.TARGET_UNLAB_IMG_FLIST.split('/')[-1])
            names_img_dir = os.listdir(dir)
            for i in names_img_dir:
                if not os.path.exists(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + i):
                    gt = cv2.imread(dir + '/' + i, 0)
                    gt[:, :] = 128
                    cv2.imwrite(self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST + '/' + i, gt)
                    print('Making arg init file for '+ str(i))

    def arg_pse_make(self):
        dir_name = os.listdir(self.config.TARGET_UNLAB_IMG_FLIST)
        arg_dir = self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST
        for name in dir_name:
            img_raw = cv2.imread(arg_dir + '/' + name)
            img_img = img_raw[::-1]
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_0_' + '.png'), img_raw)
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_0_flip' + '.png'), img_img)
            img_raw_90 = np.rot90(img_raw, 1)
            img_img_90 = np.rot90(img_img, 1)
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_90_' + '.png'), img_raw_90)
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_90_flip' + '.png'), img_img_90)
            img_raw_180 = np.rot90(img_raw, 2)
            img_img_180 = np.rot90(img_img, 2)
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_180_' + '.png'), img_raw_180)
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_180_flip' + '.png'), img_img_180)
            img_raw_270 = np.rot90(img_raw, 3)
            img_img_270 = np.rot90(img_img, 3)
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_270_' + '.png'), img_raw_270)
            cv2.imwrite(arg_dir + '/' + name.replace('.png', '_270_flip' + '.png'), img_img_270)
            print('Making pse_arg file for ' + str(name))

    def save(self):
        if self.config.MODEL == 1:
            self.edge_model.save()


    def train(self,semi=True):

        keep_training = True
        max_iteration = int(float((self.config.MAX_ITERS)))
        total = len(self.train_dataset_arg)
        total_source = len(self.train_dataset_source_arg)
        total_test = len(self.train_unlab_pseudo_dataset_arg)
        total_test_need_lab = len(self.train_unlab_pseudo_dataset)

        # 分配比例
        sample_sum = total + total_source + total_test
        totlo_iter = math.ceil(sample_sum / self.config.BATCH_SIZE_MERGE)
        iter_target_num = math.ceil(total / sample_sum * self.config.BATCH_SIZE_MERGE)
        iter_source_num = math.ceil(total_source / sample_sum * self.config.BATCH_SIZE_MERGE)
        iter_unlab_num = math.ceil(total_test / sample_sum * self.config.BATCH_SIZE_MERGE)

        train_loader = DataLoader(
            dataset=self.train_dataset_arg,
            batch_size=iter_target_num,
            num_workers=1,
            drop_last=True,
            shuffle=True
        )
        train_loader_source = DataLoader(
            dataset=self.train_dataset_source_arg,
            batch_size=iter_source_num,
            num_workers=1,
            drop_last=True,
            shuffle=True
        )

        train_unlab_pseudo_dataset = DataLoader(
            dataset=self.train_unlab_pseudo_dataset_arg,
            batch_size=iter_unlab_num,
            num_workers=1,
            drop_last=True,
            shuffle=True
        )


        # 这里的三个数据集是为了产生标记
        train_loader_each = DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            num_workers=1,
            drop_last=True,
            shuffle=True
        )
        train_loader_source_each = DataLoader(
            dataset=self.train_dataset_source,
            batch_size=1,
            num_workers=1,
            drop_last=True,
            shuffle=True
        )

        train_unlab_pseudo_dataset_each = DataLoader(
            dataset=self.train_unlab_pseudo_dataset,
            batch_size=1,
            num_workers=1,
            drop_last=True,
            shuffle=True
        )


        if total == 0:
            print('No training data was provided! Check \'TRAIN_FLIST\' value in the configuration file.')
            return

        while(keep_training):

            print('Each target Num:' + str(iter_target_num) + ' source Num: ' + str(iter_source_num) + ' unlab Num:' + str(iter_unlab_num))

            self.edge_model.train()
            self.edge_model.iteration += 1
            self.edge_model.bank_fea.reset()
            epoch = self.edge_model.iteration
            print('\n\nTraining epoch: %d' % epoch)
            print('Now lr is: ' + str(self.edge_model.gen_optimizer.param_groups[0]['lr']))
            self.log('Now lr is: ' + str(self.edge_model.gen_optimizer.param_groups[0]['lr']), name=True)

            train_loader_iter = iter(train_loader)
            train_loader_source_iter = iter(train_loader_source)
            train_pseudo_dataset_iter = iter(train_unlab_pseudo_dataset)
            totol_loss_target = 0
            totol_loss_source = 0
            totol_loss_unlab = 0
            iter_current_target = 0
            iter_current_source = 0
            iter_current_unlab = 0
            loss_target_weight = 1
            loss_source_weight = 1
            if semi:
                loss_unlab_weight = 1
            else:
                loss_unlab_weight = 0



            for each_iter in range(totlo_iter):

                items_target = train_loader_iter.next()
                iter_current_target = iter_current_target + iter_target_num
                image, edges, _ = self.cuda(*items_target)
                outputs, gen_loss, logs = self.edge_model.process(image, edges, mode=1)
                gen_loss = gen_loss / self.config.BATCH_SIZE_MERGE * loss_target_weight
                _, ls_tmp_target = logs[0]
                totol_loss_target = totol_loss_target + ls_tmp_target
                self.edge_model.backward(gen_loss)
                if iter_current_target + iter_target_num > total:
                    train_loader_iter = iter(train_loader)
                    iter_current_target = 0

                items_source = train_loader_source_iter.next()
                iter_current_source = iter_current_source + iter_source_num
                image_source, edges_source, _ = self.cuda(*items_source)
                outputs_test, gen_loss_source, logs = self.edge_model.process(image_source, edges_source, mode=2)
                gen_loss_source = gen_loss_source / self.config.BATCH_SIZE_MERGE * loss_source_weight
                _, ls_tmp_source = logs[0]
                totol_loss_source = totol_loss_source + ls_tmp_source
                self.edge_model.backward(gen_loss_source)
                if iter_current_source + iter_source_num > total_source:
                    train_loader_source_iter = iter(train_loader_source)
                    iter_current_source = 0

                items_unlab = train_pseudo_dataset_iter.next()
                iter_current_unlab = iter_current_unlab + iter_unlab_num
                image_test, edges_test, _, image_name = self.cuda_semi(*items_unlab)
                outputs_test, gen_loss_test, logs = self.edge_model.process(image_test, edges_test, mode=3)
                gen_loss_test = gen_loss_test / self.config.BATCH_SIZE_MERGE * loss_unlab_weight
                _, ls_tmp_unlab = logs[0]
                totol_loss_unlab = totol_loss_unlab + ls_tmp_unlab
                self.edge_model.backward(gen_loss_test)
                if iter_current_unlab + iter_unlab_num > total_test:
                    train_pseudo_dataset_iter = iter(train_unlab_pseudo_dataset)
                    iter_current_unlab = 0

                self.edge_model.step()

                if totol_loss_unlab > 0 and semi:

                    loss_target_weight = (totol_loss_target + totol_loss_source + totol_loss_unlab) * self.config.target_lab_weight / (self.config.target_lab_weight + self.config.source_lab_weight + self.config.target_unlab_weight) / (totol_loss_target)
                    loss_source_weight = (totol_loss_target + totol_loss_source + totol_loss_unlab) * self.config.source_lab_weight / (self.config.target_lab_weight + self.config.source_lab_weight + self.config.target_unlab_weight) / (totol_loss_source)
                    loss_unlab_weight = (totol_loss_target + totol_loss_source + totol_loss_unlab) * self.config.target_unlab_weight / (self.config.target_lab_weight + self.config.source_lab_weight + self.config.target_unlab_weight) / (totol_loss_unlab)
                else:
                    loss_target_weight = (totol_loss_target + totol_loss_source) * self.config.target_lab_weight / (self.config.target_lab_weight + self.config.source_lab_weight) / (totol_loss_target)
                    loss_source_weight = (totol_loss_target + totol_loss_source) * self.config.source_lab_weight / (self.config.target_lab_weight + self.config.source_lab_weight) / (totol_loss_source)
                    loss_unlab_weight = 0

                if each_iter % self.config.LOG_INTERVAL == 0 or each_iter == iter_target_num - 1:
                    info_target = 'Epoch: [{0}/{1}][{2}/{3}] (Tar)'.format(epoch, self.MAX_ITERS, each_iter,totlo_iter) + 'Loss tem:{0} (avg:{1}) '.format(ls_tmp_target, totol_loss_target / iter_target_num / (each_iter + 1))
                    info_source = 'Epoch: [{0}/{1}][{2}/{3}] (Sou)'.format(epoch, self.MAX_ITERS, each_iter, totlo_iter) + 'Loss tem:{0} (avg:{1}) '.format(ls_tmp_source, totol_loss_source / iter_source_num / (each_iter + 1))
                    info_unlab = 'Epoch: [{0}/{1}][{2}/{3}] (Unl)'.format(epoch, self.MAX_ITERS, each_iter,totlo_iter) + 'Loss tem:{0} (avg:{1}) '.format(ls_tmp_unlab, totol_loss_unlab / iter_unlab_num / (each_iter + 1))
                    print('loss_target_weight: ' + str(loss_target_weight) + ' loss_source_weight: ' + str(loss_source_weight) + ' loss_unlab_weight: ' + str(loss_unlab_weight))

                    print(info_target)
                    print(info_source)
                    print(info_unlab)

                    self.log(info_target, True)
                    self.log(info_source, True)
                    self.log(info_unlab, True)



            # 生成伪标
            if epoch <= self.config.epoch_tune and semi:
                print('Generater pseudo labels !')
                for idx_batch, items_source in enumerate(train_loader_each):
                    if idx_batch > self.config.target_fea_img:
                        break
                    image, edges, _ = self.cuda(*items_source)
                    self.edge_model.medium_feature(image, edges, mode=1, epoch=epoch)
                for idx_batch, items_source in enumerate(train_loader_source_each):
                    if idx_batch > self.config.source_fea_img:
                        break
                    image_source, edges_source, _ = self.cuda(*items_source)
                    self.edge_model.medium_feature(image_source, edges_source, mode=2, epoch=epoch)

                self.edge_model.eval()
                name_lst = []
                tune_flag = False
                if epoch >= self.config.epoch_tune:
                    tune_flag = True
                for idx_batch, items_source in enumerate(train_unlab_pseudo_dataset_each):
                    image_test, edges_test, _, image_name = self.cuda_semi(*items_source)
                    info_unlab = 'Epoch: [{0}/{1}][{2}/{3}]'.format(epoch, self.MAX_ITERS, idx_batch, total_test_need_lab) + 'Making: pseudo label for' + image_name[0]
                    print(info_unlab)
                    self.edge_model.medium_feature(image_test, edges_test, image_name, mode=3, tune_flag=tune_flag,
                                                   epoch=epoch)
                    #self.edge_model.gen_optimizer.zero_grad()


                    if '\\' in image_name[0]:  # for windows
                        name_lst.append((epoch, image_name[0].split('\\')[-1], self.config.region_min,
                                     self.config.region_min_second, self.config.region_min_ratio,
                                     self.config.edge_dele_ratio,
                                     self.config.edge_dele_num, self.config.threhold,
                                     self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST,
                                     self.config.perued_num, self.config.min_distance, self.config.hard_distance,
                                     self.config.weight_neg_pos, self.config.conne_dis, self.config.pos_neibour_max,
                                     tune_flag, self.config.negpos_ratio,
                                     self.config.conne_ratio, self.config.dis_max_fine, self.config.epoch_could_tune, self.config.threhold_fugai,
                                     self.config.epoch_could_pos_tune, self.config.data_mode))
                    else:  # for linux
                        name_lst.append((epoch, image_name[0].split('/')[-1], self.config.region_min,
                                     self.config.region_min_second, self.config.region_min_ratio,
                                     self.config.edge_dele_ratio,
                                     self.config.edge_dele_num, self.config.threhold,
                                     self.config.TARGET_UNLAB_EDGE_Pseudo_FLIST,
                                     self.config.perued_num, self.config.min_distance, self.config.hard_distance,
                                     self.config.weight_neg_pos, self.config.conne_dis, self.config.pos_neibour_max,
                                     tune_flag, self.config.negpos_ratio,
                                     self.config.conne_ratio, self.config.dis_max_fine, self.config.epoch_could_tune, self.config.threhold_fugai,
                                     self.config.epoch_could_pos_tune, self.config.data_mode))

                pool = Pool(self.config.multi_process_num)
                pool.map(build_label_process, name_lst)
                pool.close()
                pool.join()
                print('over!')

                if self.config.arg_train:
                    self.arg_pse_make()



            if self.edge_model.iteration >= max_iteration:
                keep_training = False
                break

            # evaluate model at checkpoints
            if self.config.EVAL_INTERVAL and epoch % self.config.EVAL_INTERVAL == 0:
                print('\nstart eval...\n')
                self.eval()

            if self.config.SAMPLE_TEST and epoch % self.config.SAMPLE_TEST == 0 and epoch > 0:
                print('\nstart test...\n')
                self.real_test_target_out()

            # save model at checkpoints
            if self.config.SAVE_INTERVAL and epoch % self.config.SAVE_INTERVAL == 0:
                self.save()


        print('\nEnd training....')

    def eval(self):
        path = os.path.join(self.samples_path, self.model_name)
        if not os.path.exists(self.samples_path):
            os.mkdir(self.samples_path)
        if not os.path.exists(path):
            os.mkdir(path)

        all_iteration = self.edge_model.iteration

        val_loader = DataLoader(
            dataset=self.train_unlab_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=True
        )

        val_train_loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=1,
            drop_last=True,
            shuffle=True
        )

        val_train_source_loader = DataLoader(
            dataset=self.train_dataset_source,
            batch_size=1,
            drop_last=True,
            shuffle=True
        )

        model = self.config.MODEL
        total = len(self.train_unlab_dataset)

        self.edge_model.eval()

        progbar = Progbar(total, width=20, stateful_metrics=['it'])
        iteration = 0

        for items in val_loader:
            iteration += 1
            image, edges,  mask = self.cuda(*items)

            # edge model
            if model == 1:
                # output, _ = self.postprocess(outputs[5])
                output, _, _ = self.edge_model(image)
                image_per_row = 1
                images = stitch_images(
                    self.postprocess(image),
                    self.postprocess(edges),
                    self.postprocess(output[0]),
                    self.postprocess(output[1]),
                    self.postprocess(output[2]),
                    self.postprocess(output[3]),
                    self.postprocess(output[4]),
                    self.postprocess(output[5]),
                    img_per_row=image_per_row
                )
                # output_bin = 255 - output
                path = os.path.join(self.samples_path, self.model_name)
                name = os.path.join(path,'test_v_' + str(all_iteration).zfill(5) + '_' + str(iteration) + ".png")

                images.save(name)
                break

            logs = [("it", iteration), ] + logs
            progbar.add(len(image), values=logs)

        for items in val_train_loader:
            iteration += 1
            image, edges,  mask = self.cuda(*items)

            # edge model
            if model == 1:
                # eval
                # output, _ = self.postprocess(outputs[5])
                output, _, _ = self.edge_model(image)
                image_per_row = 1
                images = stitch_images(
                    self.postprocess(image),
                    self.postprocess(edges),
                    self.postprocess(output[0]),
                    self.postprocess(output[1]),
                    self.postprocess(output[2]),
                    self.postprocess(output[3]),
                    self.postprocess(output[4]),
                    self.postprocess(output[5]),
                    img_per_row=image_per_row
                )
                # output_bin = 255 - output
                path = os.path.join(self.samples_path, self.model_name)
                name = os.path.join(path,'test_t_' + str(all_iteration).zfill(5) + '_' + str(iteration) + ".png")
                images.save(name)
                break

            logs = [("it", iteration), ] + logs
            progbar.add(len(image), values=logs)
            #break

        for items in val_train_source_loader:
            iteration += 1
            image, edges,  mask = self.cuda(*items)

            # edge model
            if model == 1:
                # output, _ = self.postprocess(outputs[5])
                _, output, _ = self.edge_model(image)
                image_per_row = 1
                images = stitch_images(
                    self.postprocess(image),
                    self.postprocess(edges),
                    self.postprocess(output[0]),
                    self.postprocess(output[1]),
                    self.postprocess(output[2]),
                    self.postprocess(output[3]),
                    self.postprocess(output[4]),
                    self.postprocess(output[5]),
                    img_per_row=image_per_row
                )
                # output_bin = 255 - output
                path = os.path.join(self.samples_path, self.model_name)
                name = os.path.join(path,'test_s_' + str(all_iteration).zfill(5) + '_' + str(iteration) + ".png")
                images.save(name)
                break

            logs = [("it", iteration), ] + logs
            progbar.add(len(image), values=logs)
            #break


    def real_test_target_out(self):
        self.edge_model.eval()
        all_iteration = self.edge_model.iteration
        if not os.path.exists(self.out_results_path_real):
            os.makedirs(self.out_results_path_real)
        save_dir_epoch = self.config.PATH + '/' + 'epoch_' + str(all_iteration)
        if not os.path.exists(save_dir_epoch):
            os.makedirs(save_dir_epoch)

        num_f1 = []
        model = self.config.MODEL

        out_test_dataset = Dataset_out(self.config, self.config.TEST_IMG_FLIST, mode = self.config.data_mode)

        out_test_loader = DataLoader(
            dataset=out_test_dataset,
            batch_size=1,
        )

        index = 0
        for items in out_test_loader:
            name = out_test_dataset.load_name(index)
            image_test = items.cuda()
            index += 1

            # edge model

            output, _, _ = self.edge_model(image_test)
            out_img = output[5].cpu().detach().numpy()[0, 0] * 255

            for idxx, o in enumerate(output):
                if idxx == 5:
                    output = o

            print(index, name)
            images = Image.fromarray(out_img.astype(np.uint8))
            path = os.path.join(save_dir_epoch, name)
            images.save(path)

        print('\nEnd test....')

    def real_test_out(self):
        self.edge_model.eval()
        if not os.path.exists(self.out_results_path_real):
            os.makedirs(self.out_results_path_real)

        num_f1 = []
        model = self.config.MODEL


        out_test_dataset = Dataset_out(self.config, self.config.TEST_IMG_FLIST, mode = self.config.data_mode)

        out_test_loader = DataLoader(
            dataset=out_test_dataset,
            batch_size=1,
        )

        index = 0
        for items in out_test_loader:
            name = out_test_dataset.load_name(index)
            image_test  = items.cuda()
            index += 1

            # edge model

            output, _, _ = self.edge_model(image_test)
            out_img = output[5].cpu().detach().numpy()[0,0] * 255

            for idxx, o in enumerate(output):
                if idxx == 5:
                    output = o

            print(index, name)
            images = Image.fromarray(out_img.astype(np.uint8))
            path = os.path.join(self.out_results_path_real, name)
            images.save(path)

        print('\nEnd test....')


    def log(self, logs, name=False):
        with open(self.log_file, 'a') as f:
            if name:
                f.write(logs + '\n')
            else:
                f.write('%s\n' % ' '.join([str(item[1]) for item in logs]))

    def cuda(self, *args):
        return (item.to(self.config.DEVICE) for item in args)

    def cuda_semi(self, *args):
        a, b, c, d = args
        return a.to(self.config.DEVICE), b.to(self.config.DEVICE), c.to(self.config.DEVICE), d

    def postprocess(self, img):
        # [0, 1] => [0, 255]
        img = img * 255.0
        img = img.permute(0, 2, 3, 1)
        return img.int()


    def real_fea_test(self):
        self.edge_model.eval()

        def fea_predict(fes_medium,fea,fea_weight,temp=1):
            fea_medium_new = fes_medium.permute(0, 2, 3, 1)
            batch_ = fea_medium_new.shape[0]
            fea_medium_new_ = fea_medium_new.reshape(batch_ * 256 * 256, 105)

            fea_new = (fea.permute(1, 0) * fea_weight.cuda()).permute(1, 0)
            fea_medium_new_ = fea_medium_new_ * fea_weight.cuda()

            fea_norm = torch.norm(fea_new, p=None, dim=0)
            fea_norm_ = fea_new / fea_norm
            fea_medium_new_norm = torch.norm(fea_medium_new_, p=None, dim=1)
            fea_medium_new_norm_ = (fea_medium_new_.permute(1, 0) / fea_medium_new_norm).permute(1, 0)
            pp = fea_medium_new_norm_.mm(fea_norm_)


            output = FF.softmax(pp, dim=1)
            output_ = (output + 1e-8) / temp

            dddd = output_.detach().cpu().numpy()
            dddd = dddd.reshape(batch_, 256, 256, 4)

            out_0 = dddd[0, :, :, 0]
            out_1 = dddd[0, :, :, 1]
            out_2 = dddd[0, :, :, 2]
            out_3 = dddd[0, :, :, 3]

            out_predict = out_2 + out_3

            return out_predict


        num_f1 = []

        model = self.config.MODEL
        create_dir(self.results_path)

        test_loader = DataLoader(
            dataset=self.train_unlab_dataset,
            batch_size=1,
        )

        index = 0
        for items in test_loader:
            name = self.train_unlab_dataset.load_name(index)
            image, images_gray, edges, edges_fuzzy, edges_balance, mask, mask_balance,edge_lab_fuzzy  = self.cuda(*items)
            index += 1


            output, fes_medium = self.edge_model(image)
            out_this = fea_predict(fes_medium,self.edge_model.bank_fea.fea().detach(),self.edge_model.fea_weight_this)
            out_domain = fea_predict(fes_medium, self.edge_model.domain_fea.detach(),self.edge_model.fea_weight_domain)

            images_domain = Image.fromarray((out_domain*255).astype(np.uint8))
            images_this = Image.fromarray((out_this*255).astype(np.uint8))
            images_domain.save(self.config.PATH +'/fea_domain'+name)
            images_this.save(self.config.PATH +'/fea_this'+name)

        print('\nEnd test....')