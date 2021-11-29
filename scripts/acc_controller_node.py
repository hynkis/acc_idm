#!/usr/bin/env python
#
# acc controller
#
#  Created on: Sept 3, 2020
#      Author: Hyunki Seong

import rospy
from visualization_msgs.msg import Marker
from nav_msgs.msg import Path
# from eurecar_lcm_to_ros_publisher.msg import eurecar_can_t, eurecar_con_input_t
from ackermann_msgs.msg import AckermannDrive
# from eurecar_msgs.msg import frenetPath

# for KETI IONIQ
from control_msgs.msg import RadarData, VehicleState
from driving_msgs.msg import VehicleCmd

# for ACC algorithms
from intelligent_driver_model import intelligent_driver_model as IDM

import numpy as np
import matplotlib.pyplot as plt
import math
import sys
from enum import Enum

RAD2DEGREE = 57.2958

STEERING_ANGLE_NORM_RANGE_MIN = 0  # in deg
STEERING_ANGLE_NORM_RANGE_MAX = 180  # in deg
STEERING_ANGLE_NORM_RANGE = (
    STEERING_ANGLE_NORM_RANGE_MAX - STEERING_ANGLE_NORM_RANGE_MIN
)

class LATERAL_CONTROL_TYPE(Enum):
    E2E = 1

class CONTROL_TYPE(Enum):
    PURE_PURSUIT = 1
    STANLEY = 2

class CIPV_MSG_TYPE(Enum):
    KETI_RADAR = 1

class VEHICLE_STATE_MSG_TYPE(Enum):
    KETI_CAN = 1


class REFERENCE_PATH_TYPE(Enum):
    NAV_PATH = 1
    EURECAR_FRENET = 2


STEERING_RATIO = 14.5

class C_ROS_ACC_CONTROLLER(object):
    """
    ROS Gateway
        Subscribe:
            - Ego speed
            - cipv info

        Publish
            - Control command
    """

    def __init__(self, cipv_msg_type, can_msg_type, lateral_control_type):
        rospy.init_node("ROS_acc_control_cmd_generator", anonymous=True)
        rospy.loginfo("Set cipv msg type        : %s", cipv_msg_type.name)
        rospy.loginfo("Set can msg type         : %s", can_msg_type.name)
        rospy.loginfo("Set lateral control type : %s", lateral_control_type.name)

        self.rate = rospy.Rate(100)

        # Subscriber
        # 1 : cipv
        if cipv_msg_type == CIPV_MSG_TYPE.KETI_RADAR:
            self.cipv_sub = rospy.Subscriber(
                # "/vehicle_radar_ghost", RadarData, self.callback_cipv
                "/vehicle_radar", RadarData, self.callback_cipv
            )
        else:
            print("Invalid cipv msg type")
            assert False

        # 2 : CAN
        if can_msg_type == VEHICLE_STATE_MSG_TYPE.KETI_CAN:
            self.can_sub = rospy.Subscriber(
                "/vehicle_state", VehicleState, self.callback_can
            )
        else:
            print("Invalid can msg type")
            assert False

        # 3 : E2E
        if lateral_control_type == LATERAL_CONTROL_TYPE.E2E:
            self.lat_cmd_type = rospy.Subscriber(
                "vehicle_cmd_e2e", VehicleCmd, self.callback_lat_control
            )
        else:
            print("Invalid can msg type")
            assert False

        # Publisher
        self.pub_control_cmd = rospy.Publisher(
            "vehicle_cmd_kaist", VehicleCmd, queue_size=1
        )

        # Ego CAN data
        self.ego_accel_x         = 0.0 # [m/s2]
        self.ego_yaw_rate        = 0.0
        self.ego_speed_mps       = 0.0 # [m/s]
        self.ego_steer_angle_deg = 0.0 # steering wheel angle [deg]

        # CIPV
        self.cipv_status      = 0     # [0,1] for whether detect or not
        self.lat_dist_to_cipv = 0.0   # [m]
        self.lon_dist_to_cipv = 150.0 # [m] 150 for default value when no detection
        self.rel_vel_to_cipv  = 0.0   # [m/s]

        # Lateral Control
        self.steer_angle_cmd = 0.0    # steering wheel angle [deg]

        # Longitudinal Control
        self.DESIRED_SPEED = 50 / 3.6 # desired speed for ACC [m/s] E2E inference with 50 km/h

        # Acc command steper
        self.acc_cmd_que = []
        self.acc_step_limit = 5 # [m/s^2]

    def callback_can(self, msg):
        # CAN from KETI IONIQ
        self.ego_accel_x         = msg.a_x
        self.ego_yaw_rate        = msg.yaw_rate
        self.ego_speed_mps       = msg.v_ego / 3.6  # [km/h] to [m/s]
        self.ego_steer_angle_deg = msg.steer_angle

    def callback_cipv(self, msg):
        # CIPV from KETI IONIQ Radar
        self.cipv_status      = msg.obj_status   
        self.lat_dist_to_cipv = msg.obj_lat_pos # [m]
        self.lon_dist_to_cipv = msg.obj_lon_dist # [m]
        self.rel_vel_to_cipv  = msg.obj_rel_vel  # [m/s]

    def callback_lat_control(self, msg):
        # driving_msgs.VehicleCmd
        self.steer_angle_cmd = msg.steer_angle_cmd

    def calc_acc_cmd(self, ego_v, current_dist, other_v_rel):
        # Calculate ACC command using IDM
        
        # Intelligent Driver Model
        s0            = 5      # [m] desired gap dist
        s1            = 0      # [m] jam dist (zero is okay)
        v_desired     = self.DESIRED_SPEED
        time_headway  = 2.0    # [sec] time gap. 1.6
        accel_max     = 2.0    # [m/s2] maximum accel. 0.73
        decel_desired = 5.0    # [m/s2] desired deceleration. 1.67
        delta         = 4      # accel exponent
        l             = 0      # [m] vehicle length for calc gap 5 --> zero cuz current_dist is already gap dist
        
        params = [s0, s1, v_desired, time_headway, accel_max, decel_desired, delta, l]
        accel_decel_cmd = IDM(ego_v, current_dist, other_v_rel, params)
        
        if(len(self.acc_cmd_que) == 0):
            self.acc_cmd_que.append(accel_decel_cmd)
        else:
            past_acc_cmd = self.acc_cmd_que[0]
            
            if accel_decel_cmd > 0.0 and past_acc_cmd > 0.0:
                cliped_acc_cmd = np.minimum(accel_decel_cmd, past_acc_cmd + self.acc_step_limit)
                accel_decel_cmd = cliped_acc_cmd

            elif accel_decel_cmd > 0.0 and past_acc_cmd <= 0.0:
                cliped_acc_cmd = np.minimum(accel_decel_cmd, 0.0 + self.acc_step_limit)
                accel_decel_cmd = cliped_acc_cmd

            self.acc_cmd_que.pop()
            self.acc_cmd_que.append(accel_decel_cmd)

        return accel_decel_cmd

    def publish_control_cmd(self, steer_angle_cmd, accel_decel_cmd):
        # Collect lat/lon cmds and publish
        msg_cmd = VehicleCmd()
        msg_cmd.steer_angle_cmd = steer_angle_cmd
        msg_cmd.accel_decel_cmd = accel_decel_cmd
        self.pub_control_cmd.publish(msg_cmd)



def main():
    # Set type
    cipv_msg_type        = CIPV_MSG_TYPE.KETI_RADAR
    can_msg_type         = VEHICLE_STATE_MSG_TYPE.KETI_CAN
    lateral_control_type = LATERAL_CONTROL_TYPE.E2E

    reference_path_type = REFERENCE_PATH_TYPE.NAV_PATH
    # ros_controller = C_ROS_CONTROLLER(control_type, reference_path_type)
    ros_controller = C_ROS_ACC_CONTROLLER(cipv_msg_type, can_msg_type, lateral_control_type)

    while not rospy.is_shutdown():
        # Calculate commends
        steer_angle_cmd = ros_controller.steer_angle_cmd
        accel_decel_cmd = ros_controller.calc_acc_cmd(ros_controller.ego_speed_mps, ros_controller.lon_dist_to_cipv, ros_controller.rel_vel_to_cipv)
        print("Calculated commends. Steer : %.3f, Accel : %.3f" %(steer_angle_cmd, accel_decel_cmd))
        # Publish commend
        ros_controller.publish_control_cmd(steer_angle_cmd, accel_decel_cmd)
        ros_controller.rate.sleep()


if __name__ == "__main__":
    main()
    rospy.spin()
