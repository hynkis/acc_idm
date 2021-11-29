#
# intelligent driver model
#
#  Created on: Sept 3, 2020
#      Author: Hyunki Seong

import math
import numpy as np

def intelligent_driver_model(ego_v, current_dist, other_v_rel, params, stop_flag=False):
    """
    Intelligent Driver Model (IDM)
        @ Parameters [params]
        - s0 : jam distance front car
        - s1 : jam distance rear car
        - v_desired : desired ego velocity
        - time_headway : time headway (== time gap)
        - accel_max     : max acceleration
        - decel_desired : desired deceleration
        - delta         : acceleration exponent
        - l             : vehicle length
    """
    # Parameters
    s0, s1, v_desired, time_headway, accel_max, decel_desired, delta, l \
                        = params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7]

    # Filtering for sign issue
    other_v_rel = - other_v_rel # in IDM, other_v_rel == ego_v - other_v
    if stop_flag:
        # When stop flag is true
        other_v_rel = ego_v
    ego_v       = max(ego_v, 0)

    # Desired gap
    s_star = s0 + s1 * math.sqrt(ego_v / v_desired) + time_headway * ego_v + ego_v*other_v_rel / (2 * math.sqrt(accel_max * decel_desired))
    s_curr = max(current_dist - l, 1e-3) # gap distance
    if stop_flag:
        # When stop flag is true
        s_curr = 0.5 * s0 # half of the jam distance

    accel  = accel_max * (1 - (ego_v / v_desired)**delta - (s_star/s_curr)**2)
    accel  = np.clip(accel, -decel_desired, accel_max)

    return accel