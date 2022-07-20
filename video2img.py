from typing import IO
import os
import pathlib

from PIL import Image, ImageDraw
from moviepy.editor import *
import numpy as np
import gc
import cv2
import matplotlib.pyplot as plt
import pylab
import pickle, json

DEFAULT_WIDTH = int(64 * 4)  # 800
DEFAULT_HEIGHT = int(64 * 4)  # 600
SAVE_WIDTH = 128
SAVE_HEIGHT = 128
CROP_WIDTH = 128 * 2
CROP_HEIGHT = 128 * 2

def write_images(
        # tracks: Tracks,
        # interaction_map: Map,
        fileobj: IO,
        input_video_path: str = '',
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        step: int = 1,
        save_gif: bool = True,
        save_jpeg: bool = False,
        save_png: bool = True,
        resize: bool = False,
        crop: bool = True,
        first_skip: int = 5,
        last_skip: int = 4,
        img_per_traj: int = 256,
        len_gif: int = 200,
):
    video = VideoFileClip(input_video_path)
    frames = list(range(int(video.fps * video.duration)))
    print('fps: {}, duration: {}, num_frames: {}, video (width,heigth): {}'
          .format(video.fps, video.duration, video.fps * video.duration, video.size))
    if crop:
        DEFAULT_WIDTH, DEFAULT_HEIGHT = video.size
        left = int((DEFAULT_WIDTH - CROP_WIDTH) / 2) - 150
        top = int((DEFAULT_HEIGHT - CROP_HEIGHT) / 2) + 100
        right = left + CROP_WIDTH
        bottom = top + CROP_HEIGHT
        print('left: {}, top: {}, right: {}, bottom: {}'.format(left, top, right, bottom))
        imgs = [
            Image.fromarray(video.get_frame(f / video.fps)).crop((left, top, right, bottom))
                .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
            for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
                                             and f < int(video.fps * (video.duration - last_skip))
        ]
    elif resize:
        imgs = [
            Image.fromarray(video.get_frame(f / video.fps)).resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
            for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
                                             and f < int(video.fps * (video.duration - last_skip))
        ]
    else:
        imgs = [
            Image.fromarray(video.get_frame(f / video.fps))
            for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
                                             and f < int(video.fps * (video.duration - last_skip))
        ]

    if save_gif:
        imgs[:len_gif][0].save(fp=fileobj + '.gif', format="GIF", append_images=imgs[:len_gif],
                           save_all=True, duration=len_gif, loop=1)
    if save_jpeg:
        os.makedirs(os.path.join(fileobj, 'jpeg'), exist_ok=True)
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            im.save(os.path.join(fileobj, "{0:03d}.jpeg".format(idx)), "JPEG")
    if save_png:
        os.makedirs(fileobj, exist_ok=True)
        train_path = os.path.join(fileobj, "train")
        eval_path = os.path.join(fileobj, "eval")
        test_path = os.path.join(fileobj, "test")
        for item in [train_path, eval_path, test_path]:
            os.makedirs(item, exist_ok=True)

        traj_start_idx = 0
        png_idx = 0
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            if idx > img_per_traj * traj_start_idx - 1:
                traj_start_idx += 1
                png_idx = 0
            traj_folder_name = 'traj_{}_to_{}'.format(
                img_per_traj * (traj_start_idx - 1), img_per_traj * traj_start_idx - 1)
            # train / val / test: 8 / 1 / 1
            if img_per_traj * traj_start_idx - 1 < (len(imgs) / 10 * 1):
                os.makedirs(os.path.join(test_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(test_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, 0)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 1) <= (img_per_traj * traj_start_idx - 1) and (img_per_traj * traj_start_idx - 1) < (
                    len(imgs) / 10 * 2):
                os.makedirs(os.path.join(
                    eval_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(eval_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, 0)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 2) <= (img_per_traj * traj_start_idx - 1):
                os.makedirs(os.path.join(
                    train_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(train_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, 0)), "PNG")
                png_idx += 1

def write_images_multiple_views(
        fileobj: IO,
        input_video_path: str = '',
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        step: int = 1,
        save_gif: bool = True,
        save_jpeg: bool = False,
        save_png: bool = True,
        resize: bool = False,
        crop: bool = True,
        first_skip: int = 5,
        last_skip: int = 5,
        num_view: int = 4,
        dataset: str = 'sanjose',
        img_per_traj: int = 256,
        assigned_fps: int = 30,
        min_length: int = 1000,
        len_gif: int = 200,
):
    '''
    dataset: oval / sanjose / carla
    '''
    video = VideoFileClip(input_video_path)
    print('input_video_path: {}'.format(input_video_path))
    if input_video_path in ['circle_motion.gif', 'fluid_flow_1.gif', 'circle_motion_20220120.mp4']:
        video = concatenate_videoclips([video] * 6)
    elif input_video_path in ['circle_motion_bg_change_20220128.gif', 'circle_motion_bg_change_20220130']:
        video = concatenate_videoclips([video] * 1)
    elif input_video_path in ['two_circle_motion_overlap.mp4']:
        video = concatenate_videoclips([video] * 3)
    elif input_video_path in ['fluid_flow_2.gif']:
        print(input_video_path)
        video = concatenate_videoclips([video] * 25)
    frames = list(range(int(video.fps * video.duration)))
    if video.fps < assigned_fps:
        assigned_fps = video.fps
    min_length = min(len(frames), min_length * num_view * video.fps // assigned_fps)
    min_length = int(min_length - min_length % img_per_traj)
    print('fps: {}, duration: {}, num_frames: {}, video (width,heigth): {}, min_length: {}'
          .format(video.fps, video.duration, video.fps * video.duration, video.size, min_length))
    if crop:
        DEFAULT_WIDTH, DEFAULT_HEIGHT = video.size # WIDTH --, HEIGHT |
        LOCAL_CROP_WIDTH = CROP_WIDTH
        LOCAL_CROP_HEIGHT = CROP_HEIGHT
        left = int((DEFAULT_WIDTH - CROP_WIDTH) / 2) - 150
        top = int((DEFAULT_HEIGHT - CROP_HEIGHT) / 2) + 100
        right = left + LOCAL_CROP_WIDTH
        bottom = top + LOCAL_CROP_HEIGHT
        if dataset == 'oval':
            LOCAL_CROP_WIDTH = DEFAULT_WIDTH // 2  # 594
            LOCAL_CROP_HEIGHT = DEFAULT_HEIGHT // 2  # 413
            left = int((DEFAULT_WIDTH / 2 - LOCAL_CROP_WIDTH))
            top = int((DEFAULT_HEIGHT / 2 - LOCAL_CROP_HEIGHT))
            right = left + LOCAL_CROP_WIDTH
            bottom = top + LOCAL_CROP_HEIGHT
        elif dataset == 'sanjose':
            LOCAL_CROP_WIDTH = DEFAULT_WIDTH // 2  # 594
            LOCAL_CROP_HEIGHT = DEFAULT_HEIGHT // 2  # 413
            left = int((DEFAULT_WIDTH / 2 - LOCAL_CROP_WIDTH))
            top = int((DEFAULT_HEIGHT / 2 - LOCAL_CROP_HEIGHT))
            right = left + LOCAL_CROP_WIDTH
            bottom = top + LOCAL_CROP_HEIGHT
        elif dataset in ['circle_motion', 'sumo_sanjose-2021-11-06_20.28.57', 'fluid_flow_1',
                         'fluid_flow_2', 'students003', 'circle_motion_20220120',
                         'circle_motion_bg_change_20220128', 'circle_motion_bg_change_20220130', 'two_circle_motion_overlap']:
            LOCAL_CROP_WIDTH = DEFAULT_WIDTH // 2  # 594
            LOCAL_CROP_HEIGHT = DEFAULT_HEIGHT  # 413
            left = int((DEFAULT_WIDTH / 2 - LOCAL_CROP_WIDTH))
            top = int(0)
            right = left + LOCAL_CROP_WIDTH
            bottom = top + LOCAL_CROP_HEIGHT
        print('left: {}, top: {}, right: {}, bottom: {}'.format(left, top, right, bottom))
        # imgs = [
        #     Image.fromarray(video.get_frame(f / video.fps)).crop((left, top, right, bottom))
        #         .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
        #     for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
        #                                      and f < int(video.fps * (video.duration - last_skip))
        # ]
        imgs = []
        imgs_gif = []
        for i, f in enumerate(frames):
            if i >= min_length:
                print('Processed {} frames. Stopped.'.format(i))
                break
            if i % step == 0 and f > int(video.fps * first_skip) \
                    and f < int(video.fps * (video.duration - last_skip)):
                if num_view == 4:
                    x_biases = [0, LOCAL_CROP_WIDTH, 0, LOCAL_CROP_WIDTH]
                    y_biases = [0, 0, LOCAL_CROP_HEIGHT, LOCAL_CROP_HEIGHT]
                    '''
                    0  |  1
                    -------
                    2  |  3
                    '''
                    img_gif_list = []
                    for i, (x_bias, y_bias) in enumerate(zip(x_biases, y_biases)):
                        # x_bias = CROP_WIDTH
                        # y_bias = CROP_HEIGHT
                        left_bias = left + x_bias
                        right_bias = right + x_bias
                        top_bias = top + y_bias
                        bottom_bias = bottom + y_bias
                        cropped_img = Image.fromarray(video.get_frame(f / video.fps))\
                            .crop((left_bias, top_bias, right_bias, bottom_bias)).resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                        imgs.append(cropped_img)
                        if len(imgs_gif) < len_gif * video.fps // assigned_fps and save_gif:
                            img_gif_list.append(np.array(cropped_img))
                    if len(imgs_gif) < len_gif * video.fps // assigned_fps and save_gif:
                        temporal1_img = np.concatenate([img_gif_list[0], img_gif_list[1]], axis=1)
                        temporal2_img = np.concatenate([img_gif_list[2], img_gif_list[3]], axis=1)
                        img_gif = np.concatenate([temporal1_img, temporal2_img], axis=0)
                        imgs_gif.append(Image.fromarray(img_gif)) # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                elif num_view == 2:
                    x_biases = [0, LOCAL_CROP_WIDTH, ]
                    y_biases = [0, 0,]
                    '''
                    0  |  1
                    '''
                    img_gif_list = []
                    for j, (x_bias, y_bias) in enumerate(zip(x_biases, y_biases)):
                        left_bias = left + x_bias
                        right_bias = right + x_bias
                        top_bias = top + y_bias
                        bottom_bias = bottom + y_bias
                        cropped_img = Image.fromarray(video.get_frame(f / video.fps))\
                            .crop((left_bias, top_bias, right_bias, bottom_bias)).resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                        imgs.append(cropped_img)
                        if len(imgs_gif) < len_gif * video.fps // assigned_fps and save_gif:
                            img_gif_list.append(np.array(cropped_img))
                    if len(imgs_gif) < len_gif * video.fps // assigned_fps and save_gif:
                        img_gif = np.concatenate([img_gif_list[0], np.ones_like(img_gif_list[0][:, :1]), img_gif_list[1]], axis=1)
                        imgs_gif.append(Image.fromarray(img_gif)) # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
        orig_imgs = imgs
        imgs = []
        for i in range(0, len(orig_imgs), int(num_view * video.fps // assigned_fps)):
            imgs.extend(orig_imgs[i:i + num_view])
        print('len(imgs): {}'.format(len(imgs)))
    elif resize:
        imgs = [
            Image.fromarray(video.get_frame(f / video.fps)).resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
            for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
                                             and f < int(video.fps * (video.duration - last_skip))
        ]
    else:
        imgs = [
            Image.fromarray(video.get_frame(f / video.fps))
            for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
                                             and f < int(video.fps * (video.duration - last_skip))
        ]

    if save_gif:
        if num_view > 0:
            print('Generating GIFs ...')
            orig_imgs_gif = imgs_gif[:int(len_gif * video.fps // assigned_fps)]
            imgs_gif = []
            for i in range(0, len(orig_imgs_gif), int(num_view * video.fps // assigned_fps)):
                imgs_gif.extend(orig_imgs_gif[i:i+num_view])
            gc.collect()
            print('len(imgs_gif): ', len(imgs_gif),)
            imgs_gif[:len_gif][0].save(fp=fileobj + '.gif', format="GIF", append_images=imgs_gif[:len_gif],
                               save_all=True, duration=len_gif, loop=1)
            print('Saved to {}'.format(fileobj + '.gif'))
        else:
            imgs[:len_gif][0].save(fp=fileobj + '.gif', format="GIF", append_images=imgs[:len_gif],
                               save_all=True, duration=len(imgs[:len_gif]), loop=1)
    if save_jpeg:
        os.makedirs(os.path.join(fileobj, 'jpeg'), exist_ok=True)
        view_id = 0
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            im.save(os.path.join(fileobj, "{0:03d}_{1:02d}.jpeg".format(idx, view_id)), "JPEG")
    if save_png:
        os.makedirs(fileobj, exist_ok=True)
        train_path = os.path.join(fileobj, "train")
        eval_path = os.path.join(fileobj, "eval")
        test_path = os.path.join(fileobj, "test")
        view_id = 0
        for item in [train_path, eval_path, test_path]:
            os.makedirs(item, exist_ok=True)

        traj_start_idx = 0
        png_idx = 0
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            if idx > img_per_traj * traj_start_idx - 1:
                traj_start_idx += 1
                png_idx = 0
            traj_folder_name = 'traj_{}_to_{}'.format(
                img_per_traj * (traj_start_idx - 1), img_per_traj * traj_start_idx - 1)
            # train / val / test: 8 / 1 / 1
            if img_per_traj * traj_start_idx - 1 < (len(imgs) / 10 * 1):
                os.makedirs(os.path.join(test_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(test_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 1) <= (img_per_traj * traj_start_idx - 1) \
                    and (img_per_traj * traj_start_idx - 1) < (len(imgs) / 10 * 2):
                os.makedirs(os.path.join(
                    eval_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(eval_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 2) <= (img_per_traj * traj_start_idx - 1):
                os.makedirs(os.path.join(
                    train_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(train_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1

def images2images_multiple_views_merge(
        save_dir: str,
        input_multiview_img_path: str = '',
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        step: int = 1,
        save_gif: bool = True,
        save_jpeg: bool = False,
        save_png: bool = True,
        resize: bool = False,
        crop: bool = True,
        first_skip: int = 5,
        last_skip: int = 5,
        num_view: int = 4,
        dataset: str = 'carla',
        img_per_traj: int = 256,
        len_gif: int = 200,
):
    '''
    For CARLA
    dataset: oval / carla
    '''
    # video = VideoFileClip(input_video_path)
    # frames = list(range(int(video.fps * video.duration)))
    # print('fps: {}, duration: {}, num_frames: {}, video (width,heigth): {}'
    #       .format(video.fps, video.duration, video.fps * video.duration, video.size))
    first_subdir_name = 'weather_default'
    second_subdir_name = 'weather_HardRainNoon'
    view_path = [os.path.join(input_multiview_img_path, first_subdir_name, '_out_{}'.format(i)) for i in range(num_view)]
    print('len(view_path): ', len(view_path))
    view_img_name_list = []
    for i, each_view_path in enumerate(view_path):
        sub_view_img_name_list = []
        for j, item in enumerate(sorted(os.listdir(each_view_path))):
            view_img_name = os.path.join(each_view_path, item)
            if j > 100:
                view_img_name = view_img_name.replace(first_subdir_name, second_subdir_name)
            sub_view_img_name_list.append(view_img_name)
        view_img_name_list.append(sub_view_img_name_list)
    # print('len(view_img_name_list): ', len(view_img_name_list), view_img_name_list[0])
    min_length = min([len(each_view_img_name_list) for each_view_img_name_list in view_img_name_list])
    # min_length = int(min_length - min_length % img_per_traj)
    print('min_length: ', min_length)
    # min_length = 100
    frames = []
    for i in range(min_length):
        for view_i in range(num_view):
            frames.append(view_img_name_list[view_i][i])
    print('len(frames): ', len(frames))
    if crop:
        view_1_img_path = os.listdir(view_path[0])[0]
        im_bgr = cv2.imread(os.path.join(view_path[0], view_1_img_path))
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        print('im_rgb.shape: ', im_rgb.shape)
        # im_rgb.shape:  (720, 1280, 3)
        DEFAULT_HEIGHT, DEFAULT_WIDTH = im_rgb.shape[:2]
        LOCAL_CROP_WIDTH = CROP_WIDTH
        LOCAL_CROP_HEIGHT = CROP_HEIGHT
        left = int((DEFAULT_WIDTH - CROP_WIDTH) / 2) - 150
        top = int((DEFAULT_HEIGHT - CROP_HEIGHT) / 2) + 100
        right = left + LOCAL_CROP_WIDTH
        bottom = top + LOCAL_CROP_HEIGHT
        if dataset == 'oval':
            LOCAL_CROP_WIDTH = DEFAULT_WIDTH // 2  # 594
            LOCAL_CROP_HEIGHT = DEFAULT_HEIGHT // 2  # 413
            left = int((DEFAULT_WIDTH / 2 - LOCAL_CROP_WIDTH))
            top = int((DEFAULT_HEIGHT / 2 - LOCAL_CROP_HEIGHT))
            right = left + LOCAL_CROP_WIDTH
            bottom = top + LOCAL_CROP_HEIGHT
        elif dataset == 'carla':
            LOCAL_CROP_WIDTH = min(im_rgb.shape[:2]) # 594
            LOCAL_CROP_HEIGHT = min(im_rgb.shape[:2])  # 413
            left = int((DEFAULT_WIDTH / 2 - LOCAL_CROP_WIDTH / 2))
            top = int((DEFAULT_HEIGHT / 2 - LOCAL_CROP_HEIGHT / 2))
            right = left + LOCAL_CROP_WIDTH
            bottom = top + LOCAL_CROP_HEIGHT
        print('left: {}, top: {}, right: {}, bottom: {}'.format(left, top, right, bottom))
        # imgs = [
        #     Image.fromarray(video.get_frame(f / video.fps)).crop((left, top, right, bottom))
        #         .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
        #     for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
        #                                      and f < int(video.fps * (video.duration - last_skip))
        # ]
        frames_list = []

        imgs = []
        imgs_gif = []
        for i, f in enumerate(frames):
            if i % step == 0 and i >= int(first_skip * num_view) \
                    and i <= int((min_length - last_skip) * num_view):
                x_bias = 0
                y_bias = 0
                left_bias = left + x_bias
                right_bias = right + x_bias
                top_bias = top + y_bias
                bottom_bias = bottom + y_bias
                print(f)
                cropped_img = Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)) \
                    .crop((left_bias, top_bias, right_bias, bottom_bias)).resize((SAVE_WIDTH, SAVE_HEIGHT),
                                                                                 Image.ANTIALIAS)
                imgs.append(cropped_img)
                if num_view == 8:
                    if len(imgs_gif) < len_gif and (i + 1) % num_view == 0 and len(imgs) >= num_view and save_gif:
                        '''
                        0  |  1  |  2  |  3
                        -------------------
                        4  |  5  |  6  |  7
                        '''
                        # print(len(imgs_gif) < 100, i % num_view == 0, len(imgs))
                        # temporal1_img = np.concatenate([imgs[-8], imgs[-7], imgs[-6], imgs[-5]], axis=1)
                        # temporal2_img = np.concatenate([imgs[-4], imgs[-3], imgs[-2], imgs[-1]], axis=1)
                        temporal1_img = np.concatenate([imgs[-6], imgs[-5], imgs[-2], imgs[-1]], axis=1)
                        temporal2_img = np.concatenate([imgs[-8], imgs[-7], imgs[-4], imgs[-3]], axis=1)
                        img_gif = np.concatenate([temporal1_img, temporal2_img], axis=0)
                        imgs_gif.append(Image.fromarray(img_gif))  # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                elif num_view == 4:
                    if len(imgs_gif) < len_gif and (i+1) % num_view == 0 and len(imgs) >= num_view and save_gif:
                        '''
                        0  |  1
                        -------
                        2  |  3
                        '''
                        temporal1_img = np.concatenate([imgs[-2], imgs[-1]], axis=1)
                        temporal2_img = np.concatenate([imgs[-4], imgs[-3]], axis=1)
                        img_gif = np.concatenate([temporal1_img, temporal2_img], axis=0)
                        imgs_gif.append(Image.fromarray(img_gif)) # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                elif num_view == 2:
                    if len(imgs_gif) < len_gif and (i+1) % num_view == 0 and len(imgs) >= num_view and save_gif:
                        '''
                        0  |  1
                        '''
                        temporal2_img = np.concatenate([imgs[-2], imgs[-1]], axis=1)
                        img_gif = temporal2_img
                        imgs_gif.append(Image.fromarray(img_gif)) # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                else:
                    raise NotImplementedError('num_view {} not implemented'.format(num_view))
        if min_length > img_per_traj:
            imgs = imgs[:int(min_length - min_length % img_per_traj)*num_view]
    elif resize:
        imgs = [
            Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)).resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
            for i, f in enumerate(frames) if i % step == 0 and i > int(first_skip * num_view)
                                             and i < int((min_length - last_skip) * num_view)
        ]
    else:
        imgs = [
            Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
            for i, f in enumerate(frames) if i % step == 0 and i > int(first_skip)
                                             and i < int((min_length - last_skip))
        ]

    if save_gif:
        if num_view > 0:
            print('Generating GIFs ...')
            imgs_gif = imgs_gif[:len_gif]
            gc.collect()
            imgs_gif[:len_gif][0].save(fp=save_dir + '.gif', format="GIF", append_images=imgs_gif[:len_gif],
                                   save_all=True, duration=len_gif, loop=1)
        else:
            imgs[:len_gif][0].save(fp=save_dir + '.gif', format="GIF", append_images=imgs[:len_gif],
                               save_all=True, duration=len_gif, loop=1)
        print('Saved to {}'.format(save_dir + '.gif'))
    if save_jpeg:
        os.makedirs(os.path.join(save_dir, 'jpeg'), exist_ok=True)
        view_id = 0
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            im.save(os.path.join(save_dir, "{0:03d}_{1:02d}.jpeg".format(idx, view_id)), "JPEG")
    if save_png:
        os.makedirs(save_dir, exist_ok=True)
        train_path = os.path.join(save_dir, "train")
        eval_path = os.path.join(save_dir, "eval")
        test_path = os.path.join(save_dir, "test")
        view_id = 0
        for item in [train_path, eval_path, test_path]:
            os.makedirs(item, exist_ok=True)
        traj_start_idx = 0
        png_idx = 0
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            if idx > img_per_traj * traj_start_idx - 1:
                traj_start_idx += 1
                png_idx = 0
            traj_folder_name = 'traj_{}_to_{}'.format(
                img_per_traj * (traj_start_idx - 1), img_per_traj * traj_start_idx - 1)
            # train / val / test: 8 / 1 / 1
            if img_per_traj * traj_start_idx - 1 < (len(imgs) / 10 * 1):
                os.makedirs(os.path.join(test_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(test_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 1) <= (img_per_traj * traj_start_idx - 1) and (img_per_traj * traj_start_idx - 1) < (
                    len(imgs) / 10 * 2):
                os.makedirs(os.path.join(
                    eval_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(eval_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 2) <= (img_per_traj * traj_start_idx - 1):
                os.makedirs(os.path.join(
                    train_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(train_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
        print('Saved to {}'.format(save_dir))

def images2images_multiple_views(
        save_dir: str,
        input_multiview_img_path: str = '',
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        step: int = 1,
        save_gif: bool = True,
        save_jpeg: bool = False,
        save_png: bool = True,
        resize: bool = False,
        crop: bool = True,
        first_skip: int = 5,
        last_skip: int = 5,
        num_view: int = 4,
        dataset: str = 'carla',
        img_per_traj: int = 256,
        len_gif: int = 200,
):
    '''
    For CARLA
    dataset: oval / carla
    '''
    # video = VideoFileClip(input_video_path)
    # frames = list(range(int(video.fps * video.duration)))
    # print('fps: {}, duration: {}, num_frames: {}, video (width,heigth): {}'
    #       .format(video.fps, video.duration, video.fps * video.duration, video.size))
    view_path = [os.path.join(input_multiview_img_path, '_out_{}'.format(i)) for i in range(num_view)]
    print('len(view_path): ', len(view_path))
    view_img_name_list = []
    for each_view_path in view_path:
        view_img_name_list.append([os.path.join(each_view_path, item) for item in sorted(os.listdir(each_view_path))])
    # print('len(view_img_name_list): ', len(view_img_name_list), view_img_name_list[0])
    min_length = min([len(each_view_img_name_list) for each_view_img_name_list in view_img_name_list])
    # min_length = int(min_length - min_length % img_per_traj)
    print('min_length: ', min_length)
    # min_length = 100
    frames = []
    for i in range(min_length):
        for view_i in range(num_view):
            frames.append(view_img_name_list[view_i][i])
    print('len(frames): ', len(frames))
    if crop:
        view_1_img_path = os.listdir(view_path[0])[0]
        im_bgr = cv2.imread(os.path.join(view_path[0], view_1_img_path))
        im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        print('im_rgb.shape: ', im_rgb.shape)
        # im_rgb.shape:  (720, 1280, 3)
        DEFAULT_HEIGHT, DEFAULT_WIDTH = im_rgb.shape[:2]
        LOCAL_CROP_WIDTH = CROP_WIDTH
        LOCAL_CROP_HEIGHT = CROP_HEIGHT
        left = int((DEFAULT_WIDTH - CROP_WIDTH) / 2) - 150
        top = int((DEFAULT_HEIGHT - CROP_HEIGHT) / 2) + 100
        right = left + LOCAL_CROP_WIDTH
        bottom = top + LOCAL_CROP_HEIGHT
        if dataset == 'oval':
            LOCAL_CROP_WIDTH = DEFAULT_WIDTH // 2  # 594
            LOCAL_CROP_HEIGHT = DEFAULT_HEIGHT // 2  # 413
            left = int((DEFAULT_WIDTH / 2 - LOCAL_CROP_WIDTH))
            top = int((DEFAULT_HEIGHT / 2 - LOCAL_CROP_HEIGHT))
            right = left + LOCAL_CROP_WIDTH
            bottom = top + LOCAL_CROP_HEIGHT
        elif dataset == 'carla':
            LOCAL_CROP_WIDTH = min(im_rgb.shape[:2]) # 594
            LOCAL_CROP_HEIGHT = min(im_rgb.shape[:2])  # 413
            left = int((DEFAULT_WIDTH / 2 - LOCAL_CROP_WIDTH / 2))
            top = int((DEFAULT_HEIGHT / 2 - LOCAL_CROP_HEIGHT / 2))
            right = left + LOCAL_CROP_WIDTH
            bottom = top + LOCAL_CROP_HEIGHT
        print('left: {}, top: {}, right: {}, bottom: {}'.format(left, top, right, bottom))
        # imgs = [
        #     Image.fromarray(video.get_frame(f / video.fps)).crop((left, top, right, bottom))
        #         .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
        #     for i, f in enumerate(frames) if i % step == 0 and f > int(video.fps * first_skip)
        #                                      and f < int(video.fps * (video.duration - last_skip))
        # ]
        frames_list = []

        imgs = []
        imgs_gif = []
        for i, f in enumerate(frames):
            if i % step == 0 and i >= int(first_skip * num_view) \
                    and i <= int((min_length - last_skip) * num_view):
                x_bias = 0
                y_bias = 0
                left_bias = left + x_bias
                right_bias = right + x_bias
                top_bias = top + y_bias
                bottom_bias = bottom + y_bias
                print(f)
                cropped_img = Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)) \
                    .crop((left_bias, top_bias, right_bias, bottom_bias)).resize((SAVE_WIDTH, SAVE_HEIGHT),
                                                                                 Image.ANTIALIAS)
                imgs.append(cropped_img)
                if num_view == 8:
                    if len(imgs_gif) < len_gif and (i + 1) % num_view == 0 and len(imgs) >= num_view and save_gif:
                        '''
                        0  |  1  |  2  |  3
                        -------------------
                        4  |  5  |  6  |  7
                        '''
                        # print(len(imgs_gif) < 100, i % num_view == 0, len(imgs))
                        # temporal1_img = np.concatenate([imgs[-8], imgs[-7], imgs[-6], imgs[-5]], axis=1)
                        # temporal2_img = np.concatenate([imgs[-4], imgs[-3], imgs[-2], imgs[-1]], axis=1)
                        temporal1_img = np.concatenate([imgs[-6], imgs[-5], imgs[-2], imgs[-1]], axis=1)
                        temporal2_img = np.concatenate([imgs[-8], imgs[-7], imgs[-4], imgs[-3]], axis=1)
                        img_gif = np.concatenate([temporal1_img, temporal2_img], axis=0)
                        imgs_gif.append(Image.fromarray(img_gif))  # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                elif num_view == 4:
                    if len(imgs_gif) < len_gif and (i+1) % num_view == 0 and len(imgs) >= num_view and save_gif:
                        '''
                        0  |  1
                        -------
                        2  |  3
                        '''
                        temporal1_img = np.concatenate([imgs[-2], imgs[-1]], axis=1)
                        temporal2_img = np.concatenate([imgs[-4], imgs[-3]], axis=1)
                        img_gif = np.concatenate([temporal1_img, temporal2_img], axis=0)
                        imgs_gif.append(Image.fromarray(img_gif)) # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                elif num_view == 2:
                    if len(imgs_gif) < len_gif and (i+1) % num_view == 0 and len(imgs) >= num_view and save_gif:
                        '''
                        0  |  1
                        '''
                        temporal2_img = np.concatenate([imgs[-2], imgs[-1]], axis=1)
                        img_gif = temporal2_img
                        imgs_gif.append(Image.fromarray(img_gif)) # .resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
                else:
                    raise NotImplementedError('num_view {} not implemented'.format(num_view))
        if min_length > img_per_traj:
            imgs = imgs[:int(min_length - min_length % img_per_traj)*num_view]
    elif resize:
        imgs = [
            Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)).resize((SAVE_WIDTH, SAVE_HEIGHT), Image.ANTIALIAS)
            for i, f in enumerate(frames) if i % step == 0 and i > int(first_skip * num_view)
                                             and i < int((min_length - last_skip) * num_view)
        ]
    else:
        imgs = [
            Image.fromarray(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB))
            for i, f in enumerate(frames) if i % step == 0 and i > int(first_skip)
                                             and i < int((min_length - last_skip))
        ]

    if save_gif:
        if num_view > 0:
            print('Generating GIFs ...')
            imgs_gif = imgs_gif[:len_gif]
            gc.collect()
            imgs_gif[:len_gif][0].save(fp=save_dir + '.gif', format="GIF", append_images=imgs_gif[:len_gif],
                                   save_all=True, duration=len_gif, loop=1)
        else:
            imgs[:len_gif][0].save(fp=save_dir + '.gif', format="GIF", append_images=imgs[:len_gif],
                               save_all=True, duration=len_gif, loop=1)
        print('Saved to {}'.format(save_dir + '.gif'))
    if save_jpeg:
        os.makedirs(os.path.join(save_dir, 'jpeg'), exist_ok=True)
        view_id = 0
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            im.save(os.path.join(save_dir, "{0:03d}_{1:02d}.jpeg".format(idx, view_id)), "JPEG")
    if save_png:
        os.makedirs(save_dir, exist_ok=True)
        train_path = os.path.join(save_dir, "train")
        eval_path = os.path.join(save_dir, "eval")
        test_path = os.path.join(save_dir, "test")
        view_id = 0
        for item in [train_path, eval_path, test_path]:
            os.makedirs(item, exist_ok=True)
        traj_start_idx = 0
        png_idx = 0
        for idx, im in enumerate(imgs):
            if idx % 1000 == 0:
                print('Saving {} / {}'.format(idx, len(imgs)))
            if idx > img_per_traj * traj_start_idx - 1:
                traj_start_idx += 1
                png_idx = 0
            traj_folder_name = 'traj_{:04d}_to_{:04d}'.format(
                img_per_traj * (traj_start_idx - 1), img_per_traj * traj_start_idx - 1)
            # train / val / test: 8 / 1 / 1
            if img_per_traj * traj_start_idx - 1 < (len(imgs) / 10 * 1):
                os.makedirs(os.path.join(test_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(test_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 1) <= (img_per_traj * traj_start_idx - 1) and (img_per_traj * traj_start_idx - 1) < (
                    len(imgs) / 10 * 2):
                os.makedirs(os.path.join(
                    eval_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(eval_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
            elif (len(imgs) / 10 * 2) <= (img_per_traj * traj_start_idx - 1):
                os.makedirs(os.path.join(
                    train_path, traj_folder_name), exist_ok=True)
                im.save(os.path.join(train_path, traj_folder_name,
                                     "{0:03d}_{1:02d}.png".format(png_idx, view_id)), "PNG")
                png_idx += 1
        print('Saved to {}'.format(save_dir))


def generate_sine_data():
    xlim = 40

    # define functions
    x = np.arange(0, xlim, 0.1)
    y = np.sin(x)
    # write the data out to a file
    sinedata = open('sinedata.md', 'w')
    json.dump(y.tolist(), sinedata)
    sinedata.close()
    # interactive mode on
    pylab.ion()

    # set the data limits
    plt.xlim(0, xlim)
    plt.ylim(-1, 1)

    # plot the first 200 points in the data
    plt.plot(x[0:200], y[0:200])
    # plot the remaining data incrementally
    for i in range(200, len(y)):
        plt.scatter(x[i], y[i])
        plt.pause(0.0005)

if __name__ == "__main__":
    root_path = '../../InteractionSimulator/datasets/'
    # input_video_path = 'sumo_first_take_output.mov'
    # write_images(fileobj='{}_{}'.format(scene_name, session), input_video_path=input_video_path)
    # input_video_path = '/home/jksun/Downloads/sumo_first_take.mov'
    # write_images_multiple_views(fileobj='{}_{}_multiview_fps300_v0'.format(scene_name, session), input_video_path=input_video_path)
    # input_video_name = 'sumo_oval_2car_1000ms_delay'
    # input_video_name = 'fluid_flow_2'
    # input_video_name = 'circle_motion_20220120'
    # input_video_name = 'circle_motion_bg_change_20220128'
    # input_video_name = 'circle_motion_bg_change_20220130'
    # input_video_name = 'two_circle_motion_overlap'
    # input_video_name = 'carla_town02'
    # input_video_name = 'carla_town02_8_view_20220205'
    # input_video_name = 'Ko_PER_seq2'
    # input_video_name = 'carla_town02_8_view_20220303_color'
    # input_video_name = 'carla_town02_8_view_20220415_color_left_right/left'
    # input_video_name = 'carla_town02_8_view_20220415_left_right/left'
    # input_video_name = 'carla_manual_5'
    # input_video_name = 'carla_town02_2_view_20220708_color_merge'
    input_video_name = 'carla_town02_8_view_20220713_color'
    assigned_fps = 30
    # input_video_path = '{}.mp4'.format(input_video_name)  # 'oval_2.mp4'
    # input_video_name = 'circle_motion'
    if input_video_name in ['circle_motion', 'fluid_flow_1', 'fluid_flow_2', 'circle_motion_bg_change_20220128',
                            'circle_motion_bg_change_20220130']:
        input_video_path = '{}.gif'.format(input_video_name)  # 'oval_2.mp4'
        num_view = 2
    elif input_video_name in ['sumo_sanjose-2021-11-06_20.28.57', 'circle_motion_20220120',
                              'two_circle_motion_overlap']:
        input_video_path = '{}.mp4'.format(input_video_name)  # 'oval_2.mp4'
        num_view = 2
    elif input_video_name == 'students003':
        input_video_path = '{}.avi'.format(input_video_name)  # 'oval_2.mp4'
        num_view = 2
    elif input_video_name in ['carla_town02_8_view_20220205']:
        input_video_path = '{}'.format(input_video_name)  # 'oval_2.mp4'
        num_view = 2
    elif input_video_name in ['carla_town02_8_view_20220303_color', 'carla_town02_8_view_20220415_color_left_right/left',
                              'carla_town02_8_view_20220415_left_right/left', 'carla_town02_8_view_20220713_color']:
        input_video_path = '{}'.format(input_video_name)  # 'oval_2.mp4'
        num_view = 8
    elif input_video_name in ['Ko_PER_seq2']:
        input_video_path = '{}'.format(input_video_name)  # 'oval_2.mp4'
        num_view = 2
    elif input_video_name in ['carla_manual_1', 'carla_manual_2', 'carla_manual_3',
                              'carla_manual_4', 'carla_manual_5']:
        input_video_path = os.path.join('carla_manual', '{}.mp4'.format(input_video_name))  # 'oval_2.mp4'
        num_view = 1
    elif input_video_name in ['carla_town02_2_view_20220708_color_merge']:
        input_video_path = '{}'.format(input_video_name)  # 'oval_2.mp4'
        num_view = 2
    else:
        input_video_path = '{}.gif'.format(input_video_name)
        num_view = 4
    if 'carla_manual' in input_video_name:
        write_images(fileobj='{}_{}_view_{}'.format(input_video_name, num_view, assigned_fps),
                    input_video_path=input_video_path,
                    save_gif=True,
                    save_jpeg=False,
                    save_png=True,
                    resize=True,
                    crop=False,
                    first_skip=0,
                    last_skip=0,)
    elif 'merge' in input_video_name:
        images2images_multiple_views_merge(save_dir='{}_split_{}'.format(input_video_name, num_view), input_multiview_img_path=input_video_name,
                                     first_skip=0, last_skip=0, num_view=num_view)
    elif 'carla' in input_video_name or 'Ko_PER_seq2' in input_video_name:
        images2images_multiple_views(save_dir='{}_split_{}'.format(input_video_name, num_view), input_multiview_img_path=input_video_name,
                                     first_skip=0, last_skip=0, num_view=num_view)
    else:
        write_images_multiple_views(fileobj='{}_{}_view_{}'.format(input_video_name, num_view, assigned_fps),
                                    input_video_path=input_video_path, num_view=num_view, first_skip=0, last_skip=0,
                                    assigned_fps=assigned_fps, dataset=input_video_name)

    # generate_sine_data()
