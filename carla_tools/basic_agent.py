# Copyright (c) # Copyright (c) 2018-2020 CVC.
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

""" This module implements an agent that roams around a track following random
waypoints and avoiding other vehicles.
The agent also responds to traffic lights. """

import queue
import numpy as np
import cv2, os, sys
import carla
from agents.navigation.agent import Agent, AgentState
from agents.navigation.local_planner import LocalPlanner
from local_planner import ImageBasedLocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO
import matplotlib.pyplot as plt
from sklearn.utils import Bunch
import torch

class BasicAgent(Agent):
    """
    BasicAgent implements a basic agent that navigates scenes to reach a given
    target destination. This agent respects traffic lights and other vehicles.
    """

    def __init__(self, vehicle, target_speed=20, agent_id=0, vis=False, save_dir='save_img_dir', planner_type='ImageBased'):
        """

        :param vehicle: actor to apply to local planner logic onto
        """
        super(BasicAgent, self).__init__(vehicle)

        self._proximity_tlight_threshold = 5.0  # meters
        self._proximity_vehicle_threshold = 10.0  # meters
        self._state = AgentState.NAVIGATING
        self.planner_type = planner_type
        args_lateral_dict = {
            'K_P': 1,
            'K_D': 0.4,
            'K_I': 0,
            'dt': 1.0/20.0}
        if self.planner_type == 'ImageBased':
            self._local_planner = ImageBasedLocalPlanner(
                self._vehicle, opt_dict={'target_speed': target_speed,
                                         'lateral_control_dict': args_lateral_dict})
        else:
            self._local_planner = LocalPlanner(
                self._vehicle, opt_dict={'target_speed' : target_speed,
                'lateral_control_dict':args_lateral_dict})
        self._hop_resolution = 2.0
        self._path_seperation_hop = 2
        self._path_seperation_threshold = 0.5
        self._target_speed = target_speed
        self._grp = None
        self.traffic_light_obs = False
        self.his_length = 10
        self.q_E = queue.Queue(self.his_length)
        self.q_E_t2nd = queue.Queue(self.his_length)
        self.step_index = 0
        self.width = 50
        self.img_title = self.draw_img_title(30, 128)  # self.draw_img_title(100, 600)
        self.vis = vis
        self.first_t2no_img = None
        self.agent_id = agent_id
        self.use_pytorch_model = True # False
        self.distance_thres = 30
        self.occupied_area_thres = 1500  # 970 # 30
        self.device = 'cuda:0'
        self.collision_detection_mode = 'robot'  # 'contour'
        self.start_vis_frame_index = 10 # 150 # 65  # np.inf  # 65
        self.ego_car_size = [8, 27]  # [8, 15]
        self.forehead_distance = [5 + self.ego_car_size[0], 10 + self.ego_car_size[1]]
        self.save_img_dir = save_dir
        self.t2no_t2nd_method = 'v2'
        if self.save_img_dir is not None:
            os.makedirs(self.save_img_dir, exist_ok=True)
        if self.use_pytorch_model:
            # ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred', 'results/20220605-224950/carla_town02_2_view_20220524_color_waypoint')
            ckpt_dir = os.path.join('..', 'FedML_v4/fedml_experiments/standalone/decentralized_pred',
                                    'results/20220704-204138/carla_town02_2_view_20220524_color_waypoint')
            model_0_path = os.path.join(ckpt_dir, "model_0.pt")
            # h_units = [int(x) for x in '16,16'.split(',')]
            # args = Bunch(img_width=128)
            # input_dim = 5
            # act = 'relu'
            # self.model_0 = CONV_NN_v2(input_dim, h_units, act, args)
            # self.model_0.load_state_dict(torch.load(model_0_path))
            self.model_0 = torch.load(model_0_path)

    def draw_img_title(self, height=100, width=128, color=[255, 255, 255], text_size=0.5):
        canvas = np.zeros((height, width))
        print('canvas.shape: ', canvas.shape)  # (600, 600)
        text_bg = "BG"
        text_frame = "Frame T"
        text_diff_t2no = "Diff T2NO"
        text_diff_t2nd = "Diff T2ND"
        text_t2no = "T2NO"
        text_t2nd = "T2ND"
        img_title_bg = cv2.putText(canvas.copy(), text_bg, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX, text_size,
                                   color)
        img_title_frame = cv2.putText(canvas.copy(), text_frame, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX,
                                      text_size, color)
        img_title_diff_t2no = cv2.putText(canvas.copy(), text_diff_t2no, (int(0.25 * self.width), int(height * 0.6)),
                                          cv2.FONT_HERSHEY_COMPLEX, text_size, color)
        img_title_diff_t2nd = cv2.putText(canvas.copy(), text_diff_t2nd, (int(0.25 * self.width), int(height * 0.6)),
                                          cv2.FONT_HERSHEY_COMPLEX, text_size, color)
        img_title_t2no = cv2.putText(canvas.copy(), text_t2no, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX,
                                     text_size, color)
        img_title_t2nd = cv2.putText(canvas.copy(), text_t2nd, (int(0.25 * self.width), int(height * 0.6)), cv2.FONT_HERSHEY_COMPLEX,
                                     text_size, color)
        concat_title = np.concatenate(
            [img_title_bg, np.zeros_like(img_title_bg[:, :4]), img_title_frame, np.zeros_like(img_title_bg[:, :4]),
             img_title_diff_t2no, np.zeros_like(img_title_bg[:, :4]),
             img_title_diff_t2nd, np.zeros_like(img_title_bg[:, :4]), img_title_t2no,
             np.zeros_like(img_title_bg[:, :4]), img_title_t2nd], axis=1)
        return concat_title

    def set_destination(self, location):
        """
        This method creates a list of waypoints from agent's position to destination location
        based on the route returned by the global router
        """

        start_waypoint = self._map.get_waypoint(self._vehicle.get_location())
        end_waypoint = self._map.get_waypoint(
            carla.Location(location[0], location[1], location[2]))

        route_trace = self._trace_route(start_waypoint, end_waypoint)

        self._local_planner.set_global_plan(route_trace)

    def _trace_route(self, start_waypoint, end_waypoint):
        """
        This method sets up a global router and returns the optimal route
        from start_waypoint to end_waypoint
        """

        # Setting up global router
        if self._grp is None:
            dao = GlobalRoutePlannerDAO(self._vehicle.get_world().get_map(), self._hop_resolution)
            grp = GlobalRoutePlanner(dao)
            grp.setup()
            self._grp = grp

        # Obtain route plan
        route = self._grp.trace_route(
            start_waypoint.transform.location,
            end_waypoint.transform.location)

        return route

    def run_step(self, debug=False, view_img=None):
        """
        Execute one step of navigation.
        :return: carla.VehicleControl
        """

        # is there an obstacle in front of us?
        hazard_detected = False

        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehicles
        actor_list = self._world.get_actors()
        vehicle_list = actor_list.filter("*vehicle*")
        '''
        # check possible obstacles
        vehicle_state, vehicle = self._is_vehicle_hazard(vehicle_list)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD [{}])'.format(vehicle.id))

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True
        '''
        # check visually possible obstacles
        vehicle_state, grid_img = self._is_vehicle_hazard_visual(vehicle_list, view_img)
        if vehicle_state:
            if debug:
                print('!!! VEHICLE BLOCKING AHEAD)'.format())

            self._state = AgentState.BLOCKED_BY_VEHICLE
            hazard_detected = True

        # check for the state of the traffic lights
        if self.traffic_light_obs:
            lights_list = actor_list.filter("*traffic_light*")
            light_state, traffic_light = self._is_light_red(lights_list)
            if light_state:
                if debug:
                    print('=== RED LIGHT AHEAD [{}])'.format(traffic_light.id))

                self._state = AgentState.BLOCKED_RED_LIGHT
                hazard_detected = True

        if hazard_detected:
            control = self.emergency_stop()
        else:
            self._state = AgentState.NAVIGATING
            # standard local planner behavior
            if self.planner_type == 'ImageBased':
                control = self._local_planner.run_step(grid_img=grid_img, debug=debug)
            else:
                control = self._local_planner.run_step(debug=debug)

        return control

    def done(self):
        """
        Check whether the agent has reached its destination.
        :return bool
        """
        return self._local_planner.done()

    def _is_vehicle_hazard_visual(self, vehicle_list, view_img, t2no_thres=70.,
                                  ego_vehicle_location_offset=[500, 866], pause_time=2, step_thres=10):
        """

        :param vehicle_list: list of potential obstacle to check
        :return: a tuple given by (bool_flag, vehicle), where
                 - bool_flag is True if there is a vehicle ahead blocking us
                   and False otherwise
                 - vehicle is the blocker object itself
        """

        ego_vehicle_location = self._vehicle.get_location()
        print('ego_vehicle_location: ', ego_vehicle_location)
        # ego_vehicle_waypoint = self._map.get_waypoint(ego_vehicle_location)
        view_img = cv2.resize(view_img, (int(128), int(128)))
        if self.save_img_dir is not None:
            cv2.imwrite(os.path.join(self.save_img_dir, 'rgb_{0:06d}.png'.format(self.step_index)), cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB))
        frame = cv2.cvtColor(view_img, cv2.COLOR_BGR2GRAY)  # BGR2GRAY
        B = cv2.imread(
            os.path.join('../tools/carla_town02_8_view_20220524_color_waypoint', '_out_0', '00000240.png')).astype(
            np.uint8)
        B = cv2.resize(B, (int(128), int(128)))
        B = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)  # BGR2GRAY
        # print('B.shape: {}, frame.shape: {}'.format(B.shape, frame.shape))  # (600, 600)
        diff_img_T = (cv2.absdiff(B, frame) > t2no_thres) * 255.  # [False, True]
        diff_img_t2nd_T = (cv2.absdiff(B, frame) < t2no_thres) * 255.  # [False, True]

        # T2NO / T2ND
        if self.use_pytorch_model:
            # num_hidden = [int(x) for x in '16,16'.split(',')]
            # # x_batch = cv2.resize(view_img, (int(128), int(128)))
            x_batch = view_img[None, None, :, :, :]
            # # print('x_batch.shape: ', x_batch.shape)  # (1, 1, 128, 128, 3)
            # batch = x_batch.shape[0]
            # height = x_batch.shape[2]
            # width = x_batch.shape[3]
            #
            # h_t_0 = []
            # c_t_0 = []
            # delta_c_list_0 = []
            # delta_m_list_0 = []
            #
            # for i in range(len(num_hidden)):
            #     zeros = torch.zeros([batch, num_hidden[i], height, width]).to(self.device)
            #     h_t_0.append(zeros)
            #     c_t_0.append(zeros)
            #     delta_c_list_0.append(zeros)
            #     delta_m_list_0.append(zeros)
            #
            # memory_0 = None  # torch.zeros([batch, num_hidden[0], height, width]).to(args.device)
            # # x_0_t = x_batch
            # message_0 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(self.device)
            # message_1 = torch.zeros((x_batch.shape[0], 1, x_batch.shape[2], x_batch.shape[3], 1)).to(self.device)
            x_0_t_pred_prev = torch.tensor(x_batch).float().to(self.device) / 255 # x_0_t[:, 0:1]
            # x_0_t_pred, message_0, memory_0, h_t_0, c_t_0, delta_c_list_0, delta_m_list_0 = \
            #     self.model_0(x_0_t_pred_prev, message_0, message_1, memory_0, h_t_0, c_t_0, delta_c_list_0,
            #             delta_m_list_0)
            # print('x_0_t_pred_prev: ', x_0_t_pred_prev)
            x_0_t_pred = self.model_0(x_0_t_pred_prev)
            # x_0_t_pred_prev = x_0_t_pred.detach()
            print('x_0_t_pred.shape: ', x_0_t_pred.shape)  # torch.Size([1, 1, 128, 128, 10])
            # x_0_t_pred.shape:  torch.Size([1, 11, 128, 128])
            t2no_img = np.argmax(x_0_t_pred[0,].detach().cpu().numpy(), axis=0) # [:, :, None]
            print('t2no_img.shape: ', t2no_img.shape)
            # print('t2no_img.shape: ', t2no_img.shape, t2no_img.max(), t2no_img.min())  # (128, 128, 1) 1 1
            # t2no_img = cv2.resize(t2no_img.astype('float32'), (int(600), int(600)))

            if self.vis and self.step_index > self.start_vis_frame_index:
                # t2nd_img_vis = ((10 - t2no_img) * 255. / (self.his_length + 1)).astype('uint8')
                t2no_img_vis = (t2no_img * 255. / (self.his_length + 1)).astype('uint8')
                t2nd_img_vis = np.zeros_like(t2no_img_vis).astype('uint8')
                # t2no_img_vis = ((10 - t2no_img) * 255 / self.his_length).astype('uint8')
                # t2nd_img_vis = (t2no_img * 255 / self.his_length).astype('uint8')

                # print('B.shape, frame.shape, diff_img.shape, diff_img_t2nd.shape, t2no_img.shape, t2nd_img.shape: ',
                #       B.shape, frame.shape, diff_img.shape, diff_img_t2nd.shape, t2no_img.shape, t2nd_img.shape)
                concat_img = np.concatenate(
                    [B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T, np.zeros_like(B[:, :4]),
                     diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]), t2nd_img_vis], axis=1)

                final = np.concatenate((self.img_title, concat_img), axis=0)
                # print('final.shape: ', concat_img.shape)  # final.shape:  (600, 3620)
                final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                plt.figure(figsize=(30 * 4, 30))
                plt.title('BG, Frame T, Diff_T2NO, Diff_T2ND, T2NO, T2ND')
                plt.imshow(final, cmap='gray', )
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')
            if self.save_img_dir is not None:
                # t2nd_img_vis = ((10 - t2no_img) * 255. / (self.his_length + 1)).astype('uint8')
                t2no_img_vis = (t2no_img * 255. / (self.his_length + 1)).astype('uint8')
                t2nd_img_vis = np.zeros_like(t2no_img_vis).astype('uint8')
                concat_img = np.concatenate(
                    [B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T, np.zeros_like(B[:, :4]),
                     diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]), t2nd_img_vis],
                    axis=1)

                final = np.concatenate((self.img_title, concat_img), axis=0)
                final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                cv2.imwrite(os.path.join(self.save_img_dir, 't2no_{0:06d}.png'.format(self.step_index)), final)
        else:
            if self.t2no_t2nd_method == 'v1':
                # while not self.q_E.full():
                #     self.q_E.put(np.zeros_like(diff_img))
                while not self.q_E_t2nd.full():
                    self.q_E_t2nd.put(diff_img_t2nd_T)
                    # self.q_E_t2nd.put(np.zeros_like(diff_img_t2nd))

                q_E_t2nd_array = np.asarray([ele for ele in reversed(list(self.q_E_t2nd.queue))] + [np.ones_like(diff_img_t2nd_T)])
                t2nd_img = np.argmax(np.asarray([ele for ele in reversed(list(self.q_E_t2nd.queue))] + [np.ones_like(diff_img_t2nd_T)]), axis=0)
                print('self.step_index: {}, (t2nd_img.min(): {}, t2nd_img.max(): {})'.format(self.step_index, t2nd_img.min(), t2nd_img.max()))  # 1, (0, self.his_length)
                t2no_img = q_E_t2nd_array.shape[0] - t2nd_img
                print('self.step_index: {}, (t2no_img.min(): {}, t2no_img.max(): {})'.format(self.step_index, t2no_img.min(), t2no_img.max()))  # 1, 2

                if self.vis and self.step_index > self.start_vis_frame_index:
                    t2nd_img_vis = (t2nd_img * 255 / self.his_length).astype('uint8')
                    t2no_img_vis = (t2no_img * 255 / self.his_length).astype('uint8')

                # if self.step_index == 0:
                #     self.q_E.get()
                #     # self.q_E_t2nd.get()
                #     self.q_E.put(diff_img)
                #     # self.q_E_t2nd.put(diff_img_t2nd)
                #     # t2no_img = (np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)) * int(
                #     #     255 / self.his_length)
                #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                #     t2no_img = np.argmax(q_E_array, axis=0)
                #     t2no_img = (np.ones_like(t2no_img) * q_E_array.shape[0] - t2no_img)
                #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max())  # (0, 10)
                #     # self.first_t2no_img = t2no_img
                #     if self.vis:
                #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')
                # else:
                #     # t2no_img_binary = np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)
                #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                #     t2no_img_argmax = np.argmax(q_E_array, axis=0)
                #     t2no_img_binary = (np.ones_like(t2no_img_argmax) * q_E_array.shape[0] - t2no_img_argmax)  #  * int(255 / self.his_length)
                #     t2no_img = (t2no_img_binary > 0) * (
                #                 t2no_img_binary - self.his_length + min(self.step_index, self.his_length - 1) + 1)
                #     mask = (t2no_img == (self.step_index + 1))
                #     print('mask: ', mask)
                #     t2no_img[mask] = self.his_length
                #     # t2no_img = np.minimum(t2no_img, self.first_t2no_img)
                #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max()) # 1, 2
                #     if self.vis:
                #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')

                # print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
                #       'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
                # t2no_img.shape:  (600, 600) diff_img.shape:  (600, 600) diff_img.max():  255.0 diff_img.min():  0.0
                # bg_error[each_img] = np.sum(diff_img).tolist()
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diff_' + each_img, ), diff_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
                # cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + each_img, ), t2no_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
                # if mode not in ['static', 'mean']:
                #     cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + each_img, ), inverse_mask)
                # cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
                # print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img, ), bg_error))
                    concat_img = np.concatenate([B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T, np.zeros_like(B[:, :4]),
                                                 diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]), t2nd_img_vis], axis=1)
                    print('self.img_title.shape, concat_img.shape: ', self.img_title.shape, concat_img.shape)
                    final = np.concatenate((self.img_title, concat_img), axis=0)
                    final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                    # cv2.imshow('Image', final)
                    # cv2.waitKey()
                    plt.figure(figsize=(30 * 4, 30))
                    plt.title('BG, Frame T, Diff_T2NO, Diff_T2ND, T2NO, T2ND')
                    plt.imshow(final, cmap='gray',)
                    plt.show()
                    if pause_time is not None:
                        plt.ion()
                        plt.pause(pause_time)
                        plt.close('all')
                self.q_E_t2nd.get()
            else:  # self.t2no_t2nd_method == 'v2':
                while not self.q_E.full():
                    self.q_E.put(diff_img_T)
                while not self.q_E_t2nd.full():
                    self.q_E_t2nd.put(diff_img_t2nd_T)
                    # self.q_E_t2nd.put(np.zeros_like(diff_img_t2nd))
                # self.q_E.get()
                # self.q_E.put(diff_img)

                q_E_t2no_array = np.asarray(
                    [ele for ele in list(self.q_E.queue)] + [np.ones_like(diff_img_T) * 255.])
                t2no_img = np.argmax(q_E_t2no_array, axis=0)
                print('self.step_index: {}, (t2no_img.min(): {}, t2no_img.max(): {})'.format(self.step_index,
                                                                                             t2no_img.min(),
                                                                                             t2no_img.max()))  # 1, 2

                q_E_t2nd_array = np.asarray([ele for ele in list(self.q_E_t2nd.queue)])  # + [np.ones_like(diff_img_t2nd)])
                t2nd_img = np.argmax(q_E_t2nd_array, axis=0)
                print('self.step_index: {}, (t2nd_img.min(): {}, t2nd_img.max(): {})'.format(self.step_index,
                                                                                             t2nd_img.min(),
                                                                                             t2nd_img.max()))  # 1, (0, self.his_length)
                infty_mask = np.logical_or((np.abs(t2no_img - self.his_length) < 1e-2), (np.abs(t2nd_img) < 1e-2))
                # print('np.sum(infty_mask): ', np.sum(infty_mask))
                t2nd_img[infty_mask] = self.his_length
                # t2no_img = q_E_t2nd_array.shape[0] - t2nd_img

                self.q_E.get()
                diff_img_t2nd_t = self.q_E_t2nd.get()

                if self.vis and self.step_index > self.start_vis_frame_index:
                    t2nd_img_vis = (t2nd_img * 255 / (self.his_length + 1)).astype('uint8')
                    t2no_img_vis = (t2no_img * 255 / (self.his_length + 1)).astype('uint8')

                    # if self.step_index == 0:
                    #     self.q_E.get()
                    #     # self.q_E_t2nd.get()
                    #     self.q_E.put(diff_img)
                    #     # self.q_E_t2nd.put(diff_img_t2nd)
                    #     # t2no_img = (np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)) * int(
                    #     #     255 / self.his_length)
                    #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                    #     t2no_img = np.argmax(q_E_array, axis=0)
                    #     t2no_img = (np.ones_like(t2no_img) * q_E_array.shape[0] - t2no_img)
                    #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max())  # (0, 10)
                    #     # self.first_t2no_img = t2no_img
                    #     if self.vis:
                    #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')
                    # else:
                    #     # t2no_img_binary = np.argmax(np.asarray([ele for ele in list(self.q_E.queue)]), axis=0)
                    #     q_E_array = np.asarray([ele for ele in reversed(list(self.q_E.queue))])
                    #     t2no_img_argmax = np.argmax(q_E_array, axis=0)
                    #     t2no_img_binary = (np.ones_like(t2no_img_argmax) * q_E_array.shape[0] - t2no_img_argmax)  #  * int(255 / self.his_length)
                    #     t2no_img = (t2no_img_binary > 0) * (
                    #                 t2no_img_binary - self.his_length + min(self.step_index, self.his_length - 1) + 1)
                    #     mask = (t2no_img == (self.step_index + 1))
                    #     print('mask: ', mask)
                    #     t2no_img[mask] = self.his_length
                    #     # t2no_img = np.minimum(t2no_img, self.first_t2no_img)
                    #     print('self.step_index, t2no_img.max(): ', self.step_index, t2no_img.max()) # 1, 2
                    #     if self.vis:
                    #         t2no_img = (t2no_img * 255 / self.his_length).astype('uint8')

                    # print('t2no_img.shape: ', t2no_img.shape, 'diff_img.shape: ', diff_img.shape,
                    #       'diff_img.max(): ', diff_img.max(), 'diff_img.min(): ', diff_img.min())
                    # t2no_img.shape:  (600, 600) diff_img.shape:  (600, 600) diff_img.max():  255.0 diff_img.min():  0.0
                    # bg_error[each_img] = np.sum(diff_img).tolist()
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'diff_' + each_img, ), diff_img)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'diffT2ND_' + each_img, ), diff_img_t2nd)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 't2no_' + each_img, ), t2no_img)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 't2nd_' + each_img, ), t2nd_img)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'frame_' + each_img, ), frame)
                    # if mode not in ['static', 'mean']:
                    #     cv2.imwrite(os.path.join(save_dir, each_view, 'mask_' + each_img, ), inverse_mask)
                    # cv2.imwrite(os.path.join(save_dir, each_view, 'bg_' + each_img, ), B)
                    # print('Save to {}, diff error: {}'.format(os.path.join(save_dir, each_view, each_img, ), bg_error))
                    concat_img = np.concatenate(
                        [B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T,
                         np.zeros_like(B[:, :4]),
                         diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]),
                         t2nd_img_vis], axis=1)
                    print('self.img_title.shape, concat_img.shape: ', self.img_title.shape, concat_img.shape)
                    final = np.concatenate((self.img_title, concat_img), axis=0)
                    final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                    # cv2.imshow('Image', final)
                    # cv2.waitKey()
                    plt.figure(figsize=(30 * 4, 30))
                    plt.title('BG, Frame T, Diff_T2NO, Diff_T2ND, T2NO, T2ND')
                    plt.imshow(final, cmap='gray', )
                    plt.show()
                    if pause_time is not None:
                        plt.ion()
                        plt.pause(pause_time)
                        plt.close('all')

            if self.save_img_dir is not None:
                # t2nd_img_vis = ((10 - t2no_img) * 255 / self.his_length).astype('uint8')
                t2nd_img_vis = (t2nd_img * 255 / (self.his_length + 1)).astype('uint8')
                t2no_img_vis = (t2no_img * 255 / (self.his_length + 1)).astype('uint8')
                concat_img = np.concatenate(
                    [B, np.zeros_like(B[:, :4]), frame, np.zeros_like(B[:, :4]), diff_img_T, np.zeros_like(B[:, :4]),
                     diff_img_t2nd_T, np.zeros_like(B[:, :4]), t2no_img_vis, np.zeros_like(B[:, :4]), t2nd_img_vis],
                    axis=1)

                final = np.concatenate((self.img_title, concat_img), axis=0)
                final = cv2.resize(final, (int(final.shape[1] / 2), int(final.shape[0] / 2)))
                cv2.imwrite(os.path.join(self.save_img_dir, 't2no_{0:06d}.png'.format(self.step_index)), final)


        # for target_vehicle in vehicle_list:
        #     # do not account for the ego vehicle
        #     if target_vehicle.id == self._vehicle.id:
        #         continue
        #
        #     # if the object is not in our lane it's not an obstacle
        #     target_vehicle_waypoint = self._map.get_waypoint(target_vehicle.get_location())
        #     if target_vehicle_waypoint.road_id != ego_vehicle_waypoint.road_id or \
        #             target_vehicle_waypoint.lane_id != ego_vehicle_waypoint.lane_id:
        #         continue
        #
        #     if is_within_distance_ahead(target_vehicle.get_transform(),
        #                                 self._vehicle.get_transform(),
        #                                 self._proximity_vehicle_threshold):
        #         return (True, target_vehicle)

        # if is_within_distance_ahead(target_vehicle.get_transform(), self._vehicle.get_transform(),
        #                                 self._proximity_vehicle_threshold):
        #     return (True, 0)
        # else:
        if self.collision_detection_mode == 'contour':  # 'contour'
            # center of the contour detection
            # https://pyimagesearch.com/2016/02/01/opencv-center-of-contour/
            # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
            diff_img_3_channel = np.float32(np.concatenate((t2no_img[:, :, None], t2no_img[:, :, None], t2no_img[:, :, None]), axis=2))
            # print('diff_img_3_channel.shape: ', diff_img_3_channel.shape, diff_img_3_channel.max(), diff_img_3_channel.min())
            # gray = cv2.cvtColor(diff_img_3_channel, cv2.COLOR_BGR2GRAY)
            gray = t2no_img
            kernel = np.ones((5, 5), dtype=np.uint8)
            gray = cv2.dilate(gray, kernel, 6)
            gray = cv2.erode(gray, kernel, iterations=4)
            blur = cv2.GaussianBlur(gray, (63, 63), cv2.BORDER_DEFAULT)
            ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV)
            # bg_thresh = np.zeros_like(thresh)
            # bg_thresh[10:-10, 10:-10] = thresh[10:-10, 10:-10]
            # thresh = bg_thresh
            # print('blur.max(), blur.min(): ', blur.max(), blur.min())
            # plt.imshow(blur)
            # plt.show()
            # print('thresh.max(), thresh.min(): ', thresh.max(), thresh.min())
            # plt.imshow(thresh)
            # plt.show()
            contours, hierarchies = cv2.findContours(thresh.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            # retval, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh.astype(np.uint8), connectivity=4)
            blank = np.zeros(thresh.shape[:2], dtype='uint8')
            cv2.drawContours(blank, contours, -1, (255, 0, 0), 1)
            center_list = []
            for idx, i in enumerate(contours):
                M = cv2.moments(i)
                area = cv2.contourArea(i)
                print('area: ', area)
                if M['m00'] != 0:
                    # print('M: ', M)
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    skip_flag = False
                    if np.sum(np.abs(cx - 300) + np.abs(cy - 300)) < 20:
                        print(f"skip x: {cx} y: {cy}")
                        skip_flag = True
                    for (prev_x, prev_y) in center_list:
                        if np.sum(np.abs(cx - prev_x) + np.abs(cy - prev_y)) < 80:
                            print(f"skip x: {cx} y: {cy}")
                            skip_flag = True
                    if not skip_flag:
                        center_list.append((cx, cy))
                        cv2.drawContours(diff_img_3_channel, [i], -1, (0, 255, 0), 2)
                        cv2.circle(diff_img_3_channel, (cx, cy), 7, (0, 0, 255), -1)
                        cv2.putText(diff_img_3_channel, "center", (cx - 20, cy - 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                        print(f"x: {cx} y: {cy}")
            # for (i, label) in enumerate(labels):
            #     if label == 0:  # background
            #         continue
            #     numPixels = stats[i][-1]
            #     cx, cy = int(centroids[i][0]), int(centroids[i][1])
            #     print('centroids[i]: ', centroids[i])
            #     if numPixels > 0:
            #         cv2.circle(diff_img_3_channel, (cx, cy), 7, (0, 0, 255), -1)
            #         cv2.putText(diff_img_3_channel, "center", (cx - 20, cy - 20),
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            #         center_list.append((cx, cy))
            # concat_img = np.concatenate(
            #     [blur, np.zeros_like(blur[:, :4]), thresh, np.zeros_like(thresh[:, :4]), diff_img_3_channel[:, :, 0]], axis=1)
            if self.vis and self.step_index > self.start_vis_frame_index:
                blur = np.repeat(blur[:, :, None], 3, axis=-1)
                thresh = np.repeat(thresh[:, :, None], 3, axis=-1)
                print('blur.shape, thresh.shape, diff_img_3_channel.shape: ', blur.shape, thresh.shape, diff_img_3_channel.shape)
                concat_img = np.concatenate(
                    [blur, np.zeros_like(blur[:, :4]), thresh, np.zeros_like(thresh[:, :4]), diff_img_3_channel],
                    axis=1)
                plt.figure(figsize=(30 * 4, 30))
                plt.title('Thresh')
                plt.imshow(concat_img)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')

            if self.save_img_dir is not None:
                blur = np.repeat(blur[:, :, None], 3, axis=-1)
                thresh = np.repeat(thresh[:, :, None], 3, axis=-1)
                print('blur.shape, thresh.shape, diff_img_3_channel.shape: ', blur.shape, thresh.shape,
                      diff_img_3_channel.shape)
                concat_img = np.concatenate(
                    [blur, np.zeros_like(blur[:, :4]), thresh, np.zeros_like(thresh[:, :4]), diff_img_3_channel],
                    axis=1)
                cv2.imwrite(os.path.join(self.save_img_dir, 'contour_{0:06d}.png'.format(self.step_index)), concat_img)

            center_list = sorted(center_list, key=lambda x: x[1])
            print(center_list)

            self.step_index = self.step_index + 1
            if self.agent_id == 0 and len(center_list) > 1:
                vehicle_0 = center_list[0]
                vehicle_1 = center_list[1]
                if np.sum(np.abs(vehicle_1[0] - vehicle_0[0]) + np.abs(
                        vehicle_1[1] - vehicle_0[1])) < self.distance_thres:
                    return (True, None)
                else:
                    return (False, None)
            else:
                return (False, None)
        elif self.collision_detection_mode == 'robot':
            print('ego_vehicle_location: ', ego_vehicle_location)
            # ego_vehicle_location: Location(x=132.030029, y=201.170212, z=0.182030)
            # local_ego_vehicle_location: (100.63402784916391, 47.810590826946736)
            # ego_vehicle_location:  Location(x=132.030029, y=201.170212, z=0.182030)
            local_ego_vehicle_location = ((ego_vehicle_location.x / 0.22 - ego_vehicle_location_offset[0]),
                                          (ego_vehicle_location.y / 0.22 - ego_vehicle_location_offset[1]))  # [vertical, horizontal]
            # local_ego_vehicle_location:  (95.03753662109375, 291.1569118499756)
            print('local_ego_vehicle_location: ', local_ego_vehicle_location)
            t2no_img_wo_ego = t2no_img
            if self.vis and self.step_index > self.start_vis_frame_index:
                plt.title('T2NO')
                plt.imshow(t2no_img_wo_ego)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')

            t2no_img_wo_ego[
                            int(local_ego_vehicle_location[0]-self.ego_car_size[0]):int(local_ego_vehicle_location[0]+self.ego_car_size[0]),
                            int(local_ego_vehicle_location[1]-self.ego_car_size[1]):int(local_ego_vehicle_location[1]+self.ego_car_size[1]),
                            ] = t2no_img.max()
            start_point = (int(local_ego_vehicle_location[1]-self.ego_car_size[1]), int(local_ego_vehicle_location[0]-self.ego_car_size[0]))
            end_point = (int(local_ego_vehicle_location[1]+self.ego_car_size[1]), int(local_ego_vehicle_location[0]+self.ego_car_size[0]))
            t2no_img_wo_ego = cv2.rectangle(t2no_img_wo_ego.astype('uint8'), start_point, end_point, (0, 0, int(t2no_img_wo_ego.max())), 1)
            occupied_area = self.his_length - t2no_img_wo_ego[int(local_ego_vehicle_location[0]-self.forehead_distance[0]):int(local_ego_vehicle_location[0]+self.forehead_distance[0]),
                                              int(local_ego_vehicle_location[1]-self.forehead_distance[1]):int(local_ego_vehicle_location[1]+self.forehead_distance[1]),
                                              ]
            # occupied_area = t2no_img_wo_ego[int(local_ego_vehicle_location[0]-self.forehead_distance[0]):int(local_ego_vehicle_location[0]+self.forehead_distance[0]),
            #                                   int(local_ego_vehicle_location[1]-self.forehead_distance[1]):int(local_ego_vehicle_location[1]+self.forehead_distance[1]),
            #                                   ]
            # occupied = np.sum(occupied_area) > self.occupied_area_thres
            try:
                print('t2no_img_wo_ego.shape, occupied_area.shape: ', t2no_img_wo_ego.shape, occupied_area.shape,
                      't2no_img_wo_ego.min(): {}, t2no_img_wo_ego.max(): {}, occupied_area.min(): {}, occupied_area.max(): {}'
                      .format(t2no_img_wo_ego.min(), t2no_img_wo_ego.max(), occupied_area.min(), occupied_area.max()))
                print('np.sum(occupied_area): ', np.sum(occupied_area))
                # t2no_img_wo_ego_vis = t2no_img_wo_ego

                occupied_area_vis = (occupied_area * 255. / (self.his_length + 1)).astype('uint8')
            except:
                occupied_area = np.zeros((10, 10))
                occupied_area_vis = (occupied_area * 255. / (self.his_length + 1)).astype('uint8')
            t2no_img_wo_ego_vis = (t2no_img_wo_ego * 255. / (self.his_length + 1)).astype('uint8')
            print('t2no_img_wo_ego_vis.min(): {}, t2no_img_wo_ego_vis.max(): {}'
                  .format(t2no_img_wo_ego_vis.min(), t2no_img_wo_ego_vis.max()), 255. / (self.his_length + 1))
            # plt.title('T2NO w/o Ego')
            # plt.imshow(t2no_img_wo_ego_vis)
            # plt.show()
            if self.vis and self.step_index > self.start_vis_frame_index:
                plt.title('T2NO w/o Ego')
                plt.imshow(t2no_img_wo_ego_vis)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')

                plt.title('Occupied Area')
                plt.imshow(occupied_area_vis)
                plt.show()
                if pause_time is not None:
                    plt.ion()
                    plt.pause(pause_time)
                    plt.close('all')
            if self.save_img_dir is not None:
                cv2.imwrite(os.path.join(self.save_img_dir, 't2no_wo_ego_{0:06d}.png'.format(self.step_index)), t2no_img_wo_ego_vis)
                # cv2.imwrite(os.path.join(self.save_img_dir, 'occupied_area_{0:06d}.png'.format(self.step_index)),
                #             occupied_area_vis)

            self.step_index = self.step_index + 1
            # grid_img = t2no_img > 0
            print('t2no_img.max(), t2no_img.min(): ', t2no_img.max(), t2no_img.min())
            # plt.imshow(t2no_img)
            # plt.show()
            grid_img = t2no_img <= step_thres
            if self.agent_id == 0:
                if np.sum(occupied_area) > self.occupied_area_thres:
                    return (True, grid_img)
                else:
                    return (False, grid_img)
            else:
                return (False, grid_img)
        # x: 229 y: 476
        # x: 498 y: 472
        # return (False, None)
