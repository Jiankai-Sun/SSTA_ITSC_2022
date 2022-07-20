import os, sys, cv2
import numpy as np
import moviepy.editor as mpy
from moviepy.editor import *
import matplotlib.pyplot as plt

def img2video(src_path='', target_path='', epoch='15000', bs='', dataset_name=''):
    epoch_path = os.path.join(src_path, epoch)
    all_test_case = sorted(os.listdir(epoch_path))
    frame_list = []
    all_temporal1 = ['gt_{0:02d}_00.png'.format(i) for i in range(1, 20+1)]
    all_temporal2 = ['gt_{0:02d}_00.png'.format(i) for i in range(1, 10+1)] + ['pd_{0:02d}_00.png'.format(i) for i in range(10+1, 20+1)]
    for temporal1, temporal2 in zip(all_temporal1, all_temporal2):
        temporal1_img_list = []
        for test_case in all_test_case:
            test_case_path = os.path.join(epoch_path, test_case, temporal1)
            print('Loading img from {}.'.format(test_case_path))
            BGRimage = cv2.imread(test_case_path)
            RGBimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)
            temporal1_img_list.append(RGBimage)
        temporal1_img = np.concatenate(temporal1_img_list, axis=1)
        temporal2_img_list = []
        for test_case in all_test_case:
            test_case_path = os.path.join(epoch_path, test_case, temporal2)
            print('Loading img from {}.'.format(test_case_path))
            BGRimage = cv2.imread(test_case_path)
            RGBimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)
            temporal2_img_list.append(RGBimage)
        temporal2_img = np.concatenate(temporal2_img_list, axis=1)
        diff_img = cv2.absdiff(temporal2_img, temporal1_img)
        frame = np.concatenate([temporal1_img, np.ones_like(temporal1_img)[:4],
                                temporal2_img, np.ones_like(temporal1_img)[:4], diff_img], axis=0)
        frame_list.append(frame)

    clip = mpy.ImageSequenceClip(frame_list, fps=1)
    clip.write_videofile(os.path.join(target_path, 'vis_{}_epoch_{}_{}.mp4'.format(dataset_name, epoch, bs)), fps=1)

def CARLA_img2video_single(src_path='', target_path='', epoch='20000', bs=''):
    epoch_path = os.path.join(src_path)
    all_test_case = sorted(os.listdir(epoch_path))[-1000:]
    frame_list = []
    # all_temporal1 = ['gt_{0:02d}_00.png'.format(i) for i in range(1, 20+1)]
    # all_temporal2 = ['gt_{0:02d}_00.png'.format(i) for i in range(1, 10+1)] + ['pd_{0:02d}_00.png'.format(i) for i in range(10+1, 20+1)]
    for test_case in all_test_case:
        test_case_path = os.path.join(epoch_path, test_case)
        print('Loading img from {}.'.format(test_case_path))
        BGRimage = cv2.imread(test_case_path)
        RGBimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)
        frame_list.append(RGBimage)

    clip = mpy.ImageSequenceClip(frame_list, fps=1)
    clip.write_videofile(os.path.join(target_path, 'CARLA_single_{}.mp4'.format(bs)), fps=1)

def astar_img2video_single(src_path='', target_path='', bs=''):
    epoch_path = os.path.join(src_path)
    all_test_case = sorted(os.listdir(epoch_path))[-1000:]
    frame_list = []
    # all_temporal1 = ['gt_{0:02d}_00.png'.format(i) for i in range(1, 20+1)]
    # all_temporal2 = ['gt_{0:02d}_00.png'.format(i) for i in range(1, 10+1)] + ['pd_{0:02d}_00.png'.format(i) for i in range(10+1, 20+1)]
    for test_case in all_test_case:
        test_case_path = os.path.join(epoch_path, test_case)
        print('Loading img from {}.'.format(test_case_path))
        BGRimage = cv2.imread(test_case_path)
        RGBimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)
        frame_list.append(RGBimage)

    clip = mpy.ImageSequenceClip(frame_list, fps=1)
    clip.write_videofile(os.path.join(target_path, 'astar_{}.mp4'.format(bs)), fps=1)


def img2video_multiple_views(src_path='', target_path='', epoch='15000', num_views=4, bs='', dataset_name='', speed_up=8):
    epoch_path = os.path.join(src_path, epoch)
    all_test_case = sorted(os.listdir(epoch_path))
    # print(all_test_case)
    frame_list = []
    selected_frame_list = []
    selected_frame_index = 0
    if dataset_name in ['circle_motion', 'carla_town02_20211201', 'carla_town02_20220119', 'circle_motion_20220120',
                        'circle_motion_bg_change_20220128', 'circle_motion_bg_change_20220130', 'two_circle_motion_overlap',
                        'carla_town02_2_view_20220205_split',
                        ]:
        all_temporal1 = ['gt_{0:02d}'.format(i) for i in range(1, 10 + 1)]
        all_temporal2 = ['pd_{0:02d}'.format(i) for i in range(1, 10 + 1)]
        all_message = ['msg_{0:02d}'.format(i) for i in range(1, 10 + 1)]
    elif dataset_name in ['carla_town02_4_view_20220205_split', 'carla_town02_8_view_20220205_split','carla_town02_4_view_20220205_color_split', 'carla_town02_8_view_20220205_color_split',]:
        all_temporal1 = ['gt_{0:02d}'.format(i) for i in range(1, 3 + 1)]
        all_temporal2 = ['pd_{0:02d}'.format(i) for i in range(1, 3 + 1)]
        all_message = ['msg_{0:02d}'.format(i) for i in range(1, 3 + 1)]
    elif dataset_name in ['carla_town02_8_view_20220303_color_split_2', 'carla_town02_8_view_20220303_color_split_4', 'carla_town02_8_view_20220303_color_split_8',
                          'carla_town02_8_view_20220415_color_left_right', 'carla_town02_8_view_20220415_left_right', 'carla_manual_20220509', 'carla_town02_2_view_20220708_color_merge_split_2']:
        # all_temporal1 = ['gt_{0:02d}'.format(i) for i in range(1, 49 + 1)]
        # all_temporal2 = ['pd_{0:02d}'.format(i) for i in range(1, 49 + 1)]
        # all_message = ['msg_{0:02d}'.format(i) for i in range(1, 49 + 1)]
        all_temporal1 = ['gt_{0:02d}'.format(i) for i in range(1, 99 + 1)]
        all_temporal2 = ['pd_{0:02d}'.format(i) for i in range(1, 99 + 1)]
        all_message = ['msg_{0:02d}'.format(i) for i in range(1, 99 + 1)]
    elif dataset_name in ['carla_town02_8_view_20220713_color_split_2', 'carla_town02_8_view_20220713_color_split_4', 'carla_town02_8_view_20220713_color_split_8']:
        all_temporal1 = ['gt_{0:02d}'.format(i) for i in range(1, 26 + 1)]
        all_temporal2 = ['pd_{0:02d}'.format(i) for i in range(1, 26 + 1)]
        all_message = ['msg_{0:02d}'.format(i) for i in range(1, 26 + 1)]

        # all_temporal1 = ['gt_{0:02d}'.format(i) for i in range(1, 29 + 1)]
        # all_temporal2 = ['pd_{0:02d}'.format(i) for i in range(1, 29 + 1)]
        # all_message = ['msg_{0:02d}'.format(i) for i in range(1, 29 + 1)]
    else:
        all_temporal1 = ['gt_{0:02d}'.format(i) for i in range(1, 20+1)]
        all_temporal2 = ['gt_{0:02d}'.format(i) for i in range(1, 10+1)] + ['pd_{0:02d}'.format(i) for i in range(10+1, 20+1)]
        all_message = [''.format(i) for i in range(1, 20 + 1)]
    for temporal1, temporal2, message_name in zip(all_temporal1, all_temporal2, all_message):
        temporal1_img_list = []
        for test_case in all_test_case:
            test_case_path = os.path.join(epoch_path, test_case, temporal1)
            RGBimage_list = []
            for view_idx in range(num_views):
                view_test_case_path = test_case_path + '_{0:02d}.png'.format(view_idx)
                print('Loading img from {}.'.format(view_test_case_path))
                BGRimage_each_view = cv2.imread(view_test_case_path)
                RGBimage_each_view = cv2.cvtColor(BGRimage_each_view, cv2.COLOR_BGR2RGB)
                RGBimage_list.append(RGBimage_each_view)
            '''
            RGBimage_col1 = np.concatenate((RGBimage_list[0], RGBimage_list[1]), axis=1)
            RGBimage_col2 = np.concatenate(list(RGBimage_list)[len(RGBimage_list) // 2:], axis=1)
            RGBimage_row = np.concatenate((RGBimage_col1, RGBimage_col2), axis=0)
            '''
            if num_views == 4:
                RGBimage_col2 = np.concatenate((RGBimage_list[0], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[1]), axis=1)
                RGBimage_col1 = np.concatenate((RGBimage_list[2], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[3]), axis=1)
                RGBimage_row = np.concatenate((RGBimage_col1, np.ones_like(RGBimage_col1[:1]), RGBimage_col2), axis=0)
                RGBimage = np.concatenate((RGBimage_row, np.ones_like(RGBimage_row)[:, :4]), axis=1)
            elif num_views == 8:
                RGBimage_col2 = np.concatenate((RGBimage_list[0], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[1],
                                                np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[4], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[5]), axis=1)
                RGBimage_col1 = np.concatenate((RGBimage_list[2], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[3],
                                                np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[6], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[7]), axis=1)
                RGBimage_row = np.concatenate((RGBimage_col1, np.ones_like(RGBimage_col1[:1]), RGBimage_col2), axis=0)
                RGBimage = np.concatenate((RGBimage_row, np.ones_like(RGBimage_row)[:, :4]), axis=1)
            elif num_views == 2:
                RGBimage_col1 = np.concatenate(
                    (RGBimage_list[0], np.ones_like(RGBimage_list[0][:, :1]) * 220, RGBimage_list[1]), axis=1)
                RGBimage_col2 = RGBimage_col1
                RGBimage_row = RGBimage_col1
                RGBimage = np.concatenate((RGBimage_row, np.ones_like(RGBimage_row)[:, :4]), axis=1)
            # print('RGBimage_col1.shape: {}, RGBimage_col2.shape: {}, RGBimage_row.shape: {}, RGBimage.shape: {}, len(RGBimage_each_view): {}'
            #       .format(RGBimage_col1.shape, RGBimage_col2.shape, RGBimage_row.shape, RGBimage.shape, len(RGBimage_list)))
            temporal1_img_list.append(RGBimage)
        selected_temporal1_img = temporal1_img_list[selected_frame_index]
        temporal1_img = np.concatenate(temporal1_img_list, axis=1)
        temporal2_img_list = []
        for test_case in all_test_case:
            test_case_path = os.path.join(epoch_path, test_case, temporal2)
            RGBimage_list = []
            for view_idx in range(num_views):
                view_test_case_path = test_case_path + '_{0:02d}.png'.format(view_idx)
                print('Loading img from {}.'.format(view_test_case_path))
                BGRimage_each_view = cv2.imread(view_test_case_path)
                RGBimage_each_view = cv2.cvtColor(BGRimage_each_view, cv2.COLOR_BGR2RGB)
                RGBimage_list.append(RGBimage_each_view)
            if num_views == 4:
                RGBimage_col2 = np.concatenate((RGBimage_list[0], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[1]), axis=1)
                RGBimage_col1 = np.concatenate((RGBimage_list[2], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[3]), axis=1)
                RGBimage_row = np.concatenate((RGBimage_col1, np.ones_like(RGBimage_col1[:1]), RGBimage_col2), axis=0)
                RGBimage = np.concatenate((RGBimage_row, np.ones_like(RGBimage_row)[:, :4]), axis=1)
            elif num_views == 8:
                print(len(RGBimage_list))
                RGBimage_col2 = np.concatenate((RGBimage_list[0], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[1],
                                                np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[4], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[5]), axis=1)
                RGBimage_col1 = np.concatenate((RGBimage_list[2], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[3],
                                                np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[6], np.ones_like(RGBimage_list[0][:, :1]), RGBimage_list[7]), axis=1)
                RGBimage_row = np.concatenate((RGBimage_col1, np.ones_like(RGBimage_col1[:1]), RGBimage_col2), axis=0)
                RGBimage = np.concatenate((RGBimage_row, np.ones_like(RGBimage_row)[:, :4]), axis=1)
            elif num_views == 2:
                RGBimage_col1 = np.concatenate(
                    (RGBimage_list[0], np.ones_like(RGBimage_list[0][:, :1]) * 220, RGBimage_list[1]), axis=1)
                RGBimage_col2 = RGBimage_col1
                RGBimage_row = RGBimage_col1
                RGBimage = np.concatenate((RGBimage_row, np.ones_like(RGBimage_row)[:, :4]), axis=1)
            temporal2_img_list.append(RGBimage)
        selected_temporal2_img = temporal2_img_list[selected_frame_index]
        temporal2_img = np.concatenate(temporal2_img_list, axis=1)
        diff_img = cv2.absdiff(temporal2_img, temporal1_img)
        print(np.all(cv2.absdiff(temporal2_img, temporal1_img) == cv2.absdiff(temporal1_img, temporal2_img)))
        print('diff_img.max(), diff_img.min(): ', diff_img.max(), diff_img.min(), temporal2_img.max(),
              temporal2_img.min(), temporal1_img.max(), temporal1_img.min())
        frame = np.concatenate([temporal1_img, np.ones_like(temporal1_img)[:4], temporal2_img, np.ones_like(temporal1_img)[:4], diff_img], axis=0)
        if dataset_name in ['circle_motion', 'carla_town02_20211201', 'carla_town02_20220119', 'circle_motion_20220120',
                            'circle_motion_bg_change_20220128', 'circle_motion_bg_change_20220130', 'two_circle_motion_overlap',
                            'carla_town02_8_view_20220205_split', 'carla_town02_4_view_20220205_split',
                            'carla_town02_2_view_20220205_split', 'carla_town02_8_view_20220205_color_split', 
                            'carla_town02_4_view_20220205_color_split',
                            'carla_town02_2_view_20220205_color_split',
                            'carla_town02_8_view_20220303_color_split_2', 'carla_town02_8_view_20220303_color_split_4', 'carla_town02_8_view_20220303_color_split_8',
                            'carla_town02_8_view_20220415_color_left_right', 'carla_town02_8_view_20220415_left_right', 'carla_manual_20220509',
                            'carla_town02_2_view_20220708_color_merge_split_2', 'carla_town02_8_view_20220713_color_split_8']:
            message_list = []
            for test_case in all_test_case:
                test_case_path = os.path.join(epoch_path, test_case, message_name)
                # print('Loading img from {}.'.format(test_case_path))
                # BGRimage = cv2.imread(test_case_path)
                # RGBimage = cv2.cvtColor(BGRimage, cv2.COLOR_BGR2RGB)
                single_message_list = []
                for view_idx in range(num_views):
                    view_test_case_path = test_case_path + '_{0:02d}.png'.format(view_idx)
                    print('Loading message from {}.'.format(view_test_case_path))
                    message_each_view = cv2.imread(view_test_case_path)
                    # message_each_view = cv2.cvtColor(message_each_view, cv2.COLOR_BGR2RGB)
                    # print('message_each_view.shape: ', message_each_view.shape)  # (128, 128, 3)
                    single_message_list.append(message_each_view)
                if num_views == 4:
                    message_col2 = np.concatenate(
                        (single_message_list[0], np.ones_like(single_message_list[0][:, :1]), single_message_list[1]), axis=1)
                    message_col1 = np.concatenate(
                        (single_message_list[2], np.ones_like(single_message_list[0][:, :1]), single_message_list[3]), axis=1)
                    message_row = np.concatenate((message_col1, np.ones_like(message_col1[:1]), message_col2),
                                                  axis=0)
                    each_message = np.concatenate((message_row, np.ones_like(message_row)[:, :4]), axis=1)
                elif num_views == 8:
                    message_col2 = np.concatenate(
                        (single_message_list[0], np.ones_like(single_message_list[0][:, :1]), single_message_list[1],
                         np.ones_like(single_message_list[0][:, :1]), single_message_list[4], np.ones_like(single_message_list[0][:, :1]), single_message_list[5]),
                        axis=1)
                    message_col1 = np.concatenate(
                        (single_message_list[2], np.ones_like(single_message_list[0][:, :1]), single_message_list[3],
                         np.ones_like(single_message_list[0][:, :1]), single_message_list[6], np.ones_like(single_message_list[0][:, :1]), single_message_list[7]),
                        axis=1)
                    message_row = np.concatenate((message_col1, np.ones_like(message_col1[:1]), message_col2),
                                                 axis=0)
                    each_message = np.concatenate((message_row, np.ones_like(message_row)[:, :4]), axis=1)
                elif num_views == 2:
                    message_col1 = np.concatenate(
                        (single_message_list[0], np.ones_like(single_message_list[0][:, :1]) * 220, single_message_list[1]), axis=1)
                    message_col2 = message_col1
                    message_row = message_col1
                    each_message = np.concatenate((message_row, np.ones_like(message_row)[:, :4]), axis=1)
                message_list.append(each_message)
            selected_message = message_list[selected_frame_index]
            message_img = np.concatenate(message_list, axis=1)
            # print(frame.shape, message_img.shape, )  # (392, 2610, 3) (128, 2610, 3)
            frame = np.concatenate([frame, np.ones_like(temporal1_img)[:4], message_img], axis=0)

            selected_frame = np.concatenate([selected_temporal1_img, selected_temporal2_img,
                                             cv2.absdiff(selected_temporal1_img, selected_temporal2_img), selected_message], axis=0)
            selected_frame_list.append(selected_frame)
        frame_list.append(frame)

    clip = mpy.ImageSequenceClip(frame_list, fps=speed_up)
    clip.write_videofile(os.path.join(target_path, 'vis_{}_epoch_{}_{}_{}x.mp4'.format(dataset_name, epoch, bs, speed_up)), fps=speed_up)
    selected_frame_all = np.concatenate(selected_frame_list, axis=1)
    selected_frame_all = cv2.cvtColor(selected_frame_all, cv2.COLOR_BGR2RGB)
    cv2.imwrite(os.path.join(target_path, 'vis_{}_epoch_{}_{}.png'.format(dataset_name, epoch, bs)), selected_frame_all)
    print('Saved to {}'.format(os.path.join(target_path, 'vis_{}_epoch_{}_{}.png'.format(dataset_name, epoch, bs))))

def img2video_multiple_types(src_path='', target_path='', num_views=1, fps=1, history_length=10):
    template_array = np.ones((210, 400, 3))  # (79,394,3)
    img_idx = 0
    # all_test_case = sorted(os.listdir(epoch_path))[294:5834]  # [::2]  # [:3000]

    temporal1_img_list = []
    # for i, test_case in enumerate(all_test_case):
    while os.path.exists(os.path.join(src_path, 't2no_{0:06d}.png'.format(img_idx))):
        # RGB
        # try:
        tmp_list = []
        # t2no
        view_test_case_path = os.path.join(src_path, 't2no_{0:06d}.png'.format(img_idx))
        print('Loading img from {}'.format(view_test_case_path))
        BGRimage_each_view = cv2.imread(view_test_case_path)
        RGBimage_each_view = BGRimage_each_view
        empty_array = np.ones_like(template_array)
        t2no_img = RGBimage_each_view
        empty_array[:t2no_img.shape[0], :t2no_img.shape[1]] = t2no_img

        view_test_case_path = os.path.join(src_path, 'rgb_{0:06d}.png'.format(img_idx))
        print('Loading img from {}'.format(view_test_case_path))
        BGRimage_each_view = (cv2.imread(view_test_case_path)).astype('uint8')
        RGBimage_each_view = cv2.cvtColor(BGRimage_each_view, cv2.COLOR_BGR2RGB)
        RGBimage = RGBimage_each_view
        # tmp_list.append(empty_array)
        # empty_array = np.ones_like(template_array)
        # empty_array[0:0 + RGBimage.shape[0], t2no_img.shape[1]:(RGBimage.shape[1] + t2no_img.shape[1])] = RGBimage
        empty_array[t2no_img.shape[0]:(RGBimage.shape[0] + t2no_img.shape[0]), 0:0 + RGBimage.shape[1]] = RGBimage
        # tmp_list.append(empty_array)
        # t2no wo ego
        view_test_case_path = os.path.join(src_path, 't2no_wo_ego_{0:06d}.png'.format(img_idx))
        print('Loading img from {}'.format(view_test_case_path))
        t2no_wo_ego = cv2.imread(view_test_case_path)  #  * 255 / history_length
        # plt.title('t2no w/o ego')
        # plt.imshow(t2no_wo_ego)
        # plt.show()
        # empty_array = np.ones_like(template_array)
        # empty_array[RGBimage.shape[0]:(RGBimage.shape[0]+t2no_wo_ego.shape[0]), t2no_img.shape[1]:(t2no_img.shape[1]+t2no_wo_ego.shape[1])] = t2no_wo_ego
        empty_array[
        t2no_img.shape[0]:(t2no_img.shape[0] + t2no_wo_ego.shape[0]), RGBimage.shape[1]:(RGBimage.shape[1] + t2no_wo_ego.shape[1]),] = t2no_wo_ego
        tmp_list.append(empty_array)
        # occupied_area
        # view_test_case_path = os.path.join(src_path, 'occupied_area_{0:06d}.png'.format(img_idx))
        # print('Loading img from {}.'.format(view_test_case_path))
        # BGRimage_each_view = (cv2.imread(view_test_case_path) * 255 / history_length).astype('uint8')
        # RGBimage_each_view = BGRimage_each_view
        # RGBimage = RGBimage_each_view
        # empty_array = np.ones_like(template_array)
        # empty_array[:RGBimage.shape[0], :RGBimage.shape[1]] = RGBimage
        # tmp_list.append(empty_array)
        temporal1_img_list.extend(tmp_list)
        # except:

        img_idx = img_idx + 1

    frame_list = temporal1_img_list

    clip = mpy.ImageSequenceClip(frame_list, fps=fps)
    clip.write_videofile(os.path.join(target_path, 'vis_.mp4'.format()), fps=fps)
    print('Saved to {}'.format(os.path.join(target_path, 'vis_.mp4'.format())))

def video_speed_up(root_dir='/Users/sunjiankai/Downloads/',
                   src_path='vis_carla_town02_8_view_20220205_split_epoch_100_50_steps_1',
                   speed_up=8):
    '''

    Args:
        src_path:
        target_path: vis_carla_town02_8_view_20220205_split_epoch_100_50_steps_2.mp4
        speed_up:

    Returns:

    '''
    video = VideoFileClip(os.path.join(root_dir, src_path+'.mp4'), audio=False)
    print("old fps: {}".format(video.fps))
    video = video.set_fps(video.fps * speed_up)
    new_video = video.speedx(speed_up)
    print("new fps: {}".format(new_video.fps))
    # new_video = video.fl_time(lambda t: speed_up*t, appl_to=['mask', 'video', 'audio']).set_end(video.duration / speed_up)
    new_video.write_videofile(os.path.join(root_dir, src_path+'_{}x.mp4'.format(speed_up)))

if __name__ == "__main__":
    # dir_name = 'bair_predrnn_v2_train_eval'
    # dataset_name = 'sumo_sanjose-2021-11-06_20.28.57_30'
    # dataset_name = 'carla_town02_20211106'
    # dataset_name = 'carla_town02_20220119'
    # dataset_name = 'circle_motion_20220120'
    # dataset_name = 'circle_motion_bg_change_20220128'
    # dataset_name = 'circle_motion_bg_change_20220130'
    # dataset_name = 'two_circle_motion_overlap'
    # dataset_name = 'carla_town02_8_view_20220303_color_split_2'
    # dataset_name = 'carla_town02_8_view_20220415_color_left_right'
    # dataset_name = 'carla_town02_8_view_20220415_left_right'
    # dataset_name = 'carla_manual_20220509'
    dataset_name = 'carla_town02_8_view_20220713_color_split_2' # 'carla_town02_8_view_20220713_color_split_4'  # 'carla_town02_8_view_20220713_color_split_2'
    # results/20220508-212823/carla_manual_20220509
    # 20211226-213445/circle_motion/100/10
    # bs_name_list = ['1_NN_1_img_no_GCN']
    '''
    src_path_list = ['vis_carla_town02_8_view_20220205_split_epoch_100_50_steps_1', 'vis_carla_town02_8_view_20220205_split_epoch_100_50_steps_2',
                     'vis_carla_town02_4_view_20220205_split_epoch_100_50_step_2', 'vis_carla_town02_4_view_20220205_split_epoch_100_50_step_1',
                     'vis_carla_town02_2_view_20220205_split_epoch_100_1', 'vis_carla_town02_2_view_20220205_split_epoch_100_2']  # [0:1]
    for src_path in src_path_list:
        # print(src_path)
        video_speed_up(src_path=src_path, speed_up=16)

    img2video_multiple_types(
        src_path='/media/data/jack/Programs/msl-traffic-prediction/packages/carla_tools/save_img_dir',
        target_path='/media/data/jack/Programs/msl-traffic-prediction/packages/carla_tools')
    '''
    astar_img2video_single(src_path='/media/data/jack/Programs/msl-traffic-prediction/packages/carla_tools/save_img_dir_grid',
        target_path='/media/data/jack/Programs/msl-traffic-prediction/packages/carla_tools')
    # img2video_multiple_types(src_path='/media/data/jack/Programs/msl-traffic-prediction/packages/carla_tools/save_img_dir',
    #                          target_path='/media/data/jack/Programs/msl-traffic-prediction/packages/carla_tools')
    # '''
    if dataset_name in ['circle_motion', 'carla_town02_20211201', 'carla_town02_20220119', 'circle_motion_20220120',
                        'circle_motion_bg_change_20220128', 'circle_motion_bg_change_20220130', 'two_circle_motion_overlap']:
        bs_name_list = [""]
        num_views = 2
        epoch = '100'
        dir_name = '20220124-144002'
    elif dataset_name in ['carla_town02_4_view_20220205_split', 'carla_town02_8_view_20220303_color_split_4', 'carla_town02_8_view_20220713_color_split_4']:
        bs_name_list = [""]
        num_views = 4
        epoch = '10'
        # dir_name = '20220206-162103'
        dir_name = '20220714-073513'
    elif dataset_name in ['carla_town02_8_view_20220205_split', 'carla_town02_8_view_20220303_color_split_8', 'carla_town02_8_view_20220713_color_split_8']:
        bs_name_list = [""]
        num_views = 8
        epoch = '10'
        dir_name = '20220716-192532' # 20220211-071720
    elif dataset_name in ['carla_town02_2_view_20220205_split', 'carla_town02_8_view_20220303_color_split_2',
                          'carla_town02_8_view_20220415_color_left_right', 'carla_town02_8_view_20220415_left_right',
                          'carla_manual_20220509', 'carla_town02_2_view_20220708_color_merge_split_2', 'carla_town02_8_view_20220713_color_split_2']:
        bs_name_list = [""]
        num_views = 2
        epoch = '20'
        dir_name = '20220718-061932'  # '20220509-182500'  # '20220508-175114'  # '20220211-071720'
    else:
        bs_name_list = ['4_NN_4_img_no_GCN', '1_NN_4_img_GCN', '1_NN_4_img_no_GCN', '4_NN_4_img_GCN', '1_NN_1_img_no_GCN',] # 4_NN_4_img_FC, 4_NN_4_img_Identity
        num_views = 4
        epoch = '15000'
        dir_name = '20220124-144002'
    for bs_name in bs_name_list:
        # bs_name = '4_NN_4_img_no_GCN'
        # dir_name = 'results/carla_town03_'+bs_name
        if dataset_name in ['circle_motion', 'carla_town02_20211201', 'carla_town02_20220119', 'circle_motion_20220120',
                            'circle_motion_bg_change_20220128', 'circle_motion_bg_change_20220130', 'two_circle_motion_overlap',
                            'carla_town02_4_view_20220205_split', 'carla_town02_8_view_20220205_split', 'carla_town02_2_view_20220205_split',
                            'carla_town02_8_view_20220205_color_split_8', 'carla_town02_8_view_20220205_color_split_4', 'carla_town02_8_view_20220303_color_split_2',
                            'carla_town02_8_view_20220303_color_split_4', 'carla_town02_8_view_20220303_color_split_8', 'carla_town02_8_view_20220415_color_left_right',
                            'carla_town02_8_view_20220415_left_right', 'carla_manual_20220509', 'carla_town02_2_view_20220708_color_merge_split_2',
                            'carla_town02_8_view_20220713_color_split_2', 'carla_town02_8_view_20220713_color_split_4', 'carla_town02_8_view_20220713_color_split_8']:
            #  ../FedML_v4/fedml_experiments/standalone/decentralized_pred/results/20211226-213445/circle_motion/100/10
            src_path = os.path.join('..', 'FedML_v4', 'fedml_experiments', 'standalone', 'decentralized_pred', 'results',
                                    dir_name, dataset_name)  # 20220109-220502  20220109-212140
            target_path = os.path.join('..', 'FedML_v4', 'fedml_experiments', 'standalone', 'decentralized_pred', 'results',
                                       dir_name, dataset_name)
        else:
            dir_name = 'results/'+ dataset_name + '_' + bs_name
            src_path = os.path.join('..', '..', '..', '..', 'Downloads', 'viz', dir_name)
            target_path = os.path.join('..', '..', '..', '..', 'Downloads', 'viz', dir_name)
        # img2video(src_path=src_path, target_path=target_path, bs='1_NN_1_img_no_GCN')
        # CARLA_img2video_single(src_path=os.path.join('..', '..', '..', '..', 'Downloads', 'viz', '_out_2'), target_path=target_path, bs='dataset')
        '''
        if bs_name == '1_NN_1_img_no_GCN':
            img2video(src_path=src_path, target_path=target_path, bs=bs_name, dataset_name=dataset_name)
        else:
            img2video_multiple_views(src_path=src_path, target_path=target_path, bs=bs_name, dataset_name=dataset_name,
                                     epoch=epoch, num_views=num_views)
        '''