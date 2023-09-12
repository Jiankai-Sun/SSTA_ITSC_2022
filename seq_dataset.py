__author__ = 'gaozhifeng'
import numpy as np
import os
import cv2
from PIL import Image
import logging
import random

logger = logging.getLogger(__name__)

def data_provider(dataset_name, train_data_paths, valid_data_paths, batch_size,
                  img_width, seq_length, is_training=True, num_views=1, img_channel=3, baseline='1_NN_4_img_GCN',
                  eval_batch_size=1, n_epoch=1, args=None):
    train_data_list = train_data_paths.split(',')
    valid_data_list = valid_data_paths.split(',')
    try:
        test_seq_length = args.eval_num_step + args.num_past + 1
    except:
        print('No args.eval_num_step, use seq_length as test_seq_length')
        test_seq_length = seq_length
    # if dataset_name in ['circle_motion', 'sumo_sanjose-2021-11-06_20.28.57_30', 'carla_town02_20211201', 'students003']:
    input_param = {'paths': valid_data_list,
                   'image_width': img_width,
                   'minibatch_size': eval_batch_size,
                   'seq_length': test_seq_length,
                   'input_data_type': 'float32',
                   'name': dataset_name + ' iterator',
                    'num_views': num_views,
                   'img_channel': img_channel,
                   'baseline': baseline,
                   'n_epoch': n_epoch,
                   'n_cl_step': args.num_cl_step,
                   'cl_mode': args.cl_mode,}
    input_handle = DataProcess(input_param)
    if is_training:
        train_input_param = {'paths': train_data_list,
                            'image_width': img_width,
                            'minibatch_size': batch_size,
                            'seq_length': seq_length,
                            'input_data_type': 'float32',
                            'name': dataset_name + ' iterator',
                            'num_views': num_views,
                            'img_channel': img_channel,
                            'baseline': baseline,
                            'n_epoch': n_epoch,
                            'n_cl_step': args.num_cl_step,
                            'cl_mode': args.cl_mode,}
        train_input_handle = DataProcess(train_input_param)
        train_input_handle = train_input_handle.get_train_input_handle()
        train_input_handle.begin(do_shuffle=True)
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        return train_input_handle, test_input_handle
    else:
        test_input_handle = input_handle.get_test_input_handle()
        test_input_handle.begin(do_shuffle=False)
        return test_input_handle


class InputHandle:
    def __init__(self, datas, indices, input_param):
        self.name = input_param['name']
        self.input_data_type = input_param.get('input_data_type', 'float32')
        self.num_views = input_param.get('num_views', 4)
        self.minibatch_size = input_param['minibatch_size']
        self.image_width = input_param['image_width']
        self.datas = datas
        self.indices = indices
        self.current_position = 0
        self.current_batch_indices = []
        self.current_input_length = input_param['seq_length']
        self.img_channel = input_param.get('img_channel', 3)
        self.n_epoch = input_param.get('n_epoch', 1)
        self.n_cl_step = input_param.get('n_cl_step', 1)
        self.cl_mode = input_param.get('cl_mode', 1)

    def total(self):
        return len(self.indices)

    def begin(self, do_shuffle=True, epoch=None):
        logger.info("Initialization for read data ")
        if do_shuffle:
            random.shuffle(self.indices)
        if epoch:
            # self.current_position = int(self.total() * (epoch // self.n_cl_step) * self.n_cl_step / self.n_epoch)
            self.current_position = int(self.total() * epoch / self.n_epoch)
        else:
            self.current_position = 0
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def next(self):
        self.current_position += self.minibatch_size
        if self.no_batch_left():
            return None
        self.current_batch_indices = self.indices[self.current_position:self.current_position + self.minibatch_size]

    def no_batch_left(self, epoch=None):
        if self.current_position + self.minibatch_size >= self.total():
            return True
        # elif epoch is not None and self.current_position != 0 and \
        #         self.current_position + self.minibatch_size >= \
        #         self.total() * min(((epoch // self.n_cl_step) + 1) * self.n_cl_step / self.n_epoch, 1.):
        #     return True
        elif epoch is not None and self.current_position != 0 and \
                self.current_position + self.minibatch_size >= \
                self.total() * min((epoch + self.n_cl_step) / self.n_epoch, 1.):
            return True
        else:
            return False

    def get_batch(self):
        if self.no_batch_left():
            logger.error(
                "There is no batch left in " + self.name + ". Consider to user iterators.begin() to rescan from the beginning of the iterators")
            return None
        input_batch = np.zeros(
            (self.minibatch_size, self.current_input_length, self.image_width, self.image_width,
             self.img_channel * self.num_views)).astype(self.input_data_type)
        for i in range(self.minibatch_size):
            batch_ind = self.current_batch_indices[i]
            begin = batch_ind
            end = begin + self.current_input_length
            data_slice = self.datas[begin:end, :, :, :]
            # print(input_batch[i, :self.current_input_length, :, :, :].shape, data_slice.shape)
            input_batch[i, :self.current_input_length, :, :, :] = data_slice
            
        input_batch = input_batch.astype(self.input_data_type)
        return input_batch

    def print_stat(self):
        logger.info("Iterator Name: " + self.name)
        logger.info("    current_position: " + str(self.current_position))
        logger.info("    Minibatch Size: " + str(self.minibatch_size))
        logger.info("    total Size: " + str(self.total()))
        logger.info("    current_input_length: " + str(self.current_input_length))
        logger.info("    Input Data Type: " + str(self.input_data_type))

class DataProcess:
    def __init__(self, input_param):
        self.paths = input_param['paths']
        self.image_width = input_param['image_width']

        self.train_person = os.listdir(self.paths[0])
        self.test_person = os.listdir(self.paths[0])  

        self.input_param = input_param
        self.seq_len = input_param['seq_length']
        self.num_views = input_param.get('num_views', 4)
        self.img_channel = input_param.get('img_channel', 3)
        self.baseline = input_param.get('baseline', '1_NN_4_img_GCN')
        self.n_epoch = input_param.get('n_epoch', 1)
        self.n_cl_step = input_param.get('n_cl_step', 1)
        self.cl_mode = input_param.get('cl_mode', 1)

    def load_data(self, paths, mode='train'):
        '''
        frame -- action -- person_seq(a dir)
        :param paths: action_path list
        :return:
        '''

        path = paths[0]
        print('begin load data ' + str(path))

        frames_np = []
        frames_file_name = []
        c_dir_list = os.listdir(path)
        for c_dir in c_dir_list:
            c_dir_path = os.path.join(path, c_dir)
            p_c_dir_list = os.listdir(c_dir_path)
            p_c_dir_list.sort()
            for file in p_c_dir_list:
                frame_im = Image.open(os.path.join(c_dir_path, file))
                frame_np = np.array(frame_im)  # (1000, 1000) numpy array
                frames_np.append(frame_np)
                frames_file_name.append(file)
        if self.baseline == '1_NN_1_img_no_GCN':
            frames_np_4to1 = []
            for view_idx_start in range(0, len(frames_np), 4):
                temporal1_img = np.concatenate((frames_np[view_idx_start + 0], frames_np[view_idx_start + 1]), axis=1)
                temporal2_img = np.concatenate((frames_np[view_idx_start + 2], frames_np[view_idx_start + 3]), axis=1)
                frame = np.concatenate([temporal1_img, temporal2_img], axis=0)
                frame = cv2.resize(frame, (self.image_width, self.image_width)) / 255.
                if view_idx_start < 100 and False:
                    os.makedirs('tmp', exist_ok=True)
                    # print('frame.shape: ', frame.shape)  # frame.shape:  (128, 128, 3)
                    frame = frame[:, :, ::-1]
                    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(os.path.join('tmp', '{}.png'.format(view_idx_start)), frame * 255.)
                frames_np_4to1.append(frame)
            data = np.asarray(frames_np_4to1)
        else:
            frames_np = np.asarray(frames_np)
            data = np.zeros((frames_np.shape[0], self.image_width, self.image_width, self.img_channel))
            for i in range(len(frames_np)):
                temp = np.float32(frames_np[i])
                data[i, :, :, :] = cv2.resize(temp, (self.image_width, self.image_width)) / 255
            new_data = np.zeros((frames_np.shape[0] // self.num_views, self.image_width, self.image_width, self.img_channel * self.num_views))
            for i in range(self.num_views):
                new_data[:, :, :, self.img_channel*i:self.img_channel*(i+1)] = data[i::self.num_views][:frames_np.shape[0] // self.num_views]
            data = new_data
        # is it a begin index of sequence
        indices = []
        index = len(data) - 1
        while index >= self.seq_len - 1:
            indices.append(index - self.seq_len + 1)
            index -= 1
        print("there are " + str(data.shape[0]) + " pictures with shape {}".format(data.shape))
        print("there are " + str(len(indices)) + " sequences")
        return data, indices

    def get_train_input_handle(self):
        train_data, train_indices = self.load_data(self.paths, mode='train')
        return InputHandle(train_data, train_indices, self.input_param)

    def get_test_input_handle(self):
        test_data, test_indices = self.load_data(self.paths, mode='test')
        return InputHandle(test_data, test_indices, self.input_param)

