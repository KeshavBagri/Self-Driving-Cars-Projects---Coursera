#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower demo.
"""

import cutils
import numpy as np
import control

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
        desired_speed = 0
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                    self._waypoints[i][0] - self._current_x,
                    self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake
    
    def distance(self, x, y, idx):
        d = np.sqrt((x-self._waypoints[idx][0])**2 + (y-self._waypoints[idx][1])**2)
        return d 
    
    def closest_cal(self, x, y):
        min_idx       = 0
        min_dist      = float("inf")
        for i in range(len(self._waypoints)):
            dist = self.distance(x, y, i)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        min_idx = len(self._waypoints)-1 if (min_idx >= len(self._waypoints)-1) else min_idx
        return min_idx
    
    def closest_cal(self, x, y):
        min_idx       = 0
        min_dist      = float("inf")
        for i in range(len(self._waypoints)):
            dist = self.distance(x, y, i)
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        min_idx = len(self._waypoints)-1 if (min_idx >= len(self._waypoints)-1) else min_idx
        return min_idx
    
    def check_left(self, x, y, ind):
        x1 = self._waypoints[ind][0]
        y1 = self._waypoints[ind][1]
        x2 = self._waypoints[ind-1][0]
        y2 = self._waypoints[ind-1][1]
        k = (x-x1)*(y2-y1)-(y-y1)*(x2-x1)
        return True if k>0 else False
    
    def calculate_steer(self, t, x, y, yaw, v_f):
        self.vars.create_var('c', 0.0)
        self.vars.create_var('k', 0.0)
        self.vars.create_var('h_previous', 0.0)
        self.vars.create_var('yaw_prev', 0.0)
        self.vars.create_var('e_prev', 0.0)
        self.vars.create_var('psi_previous', 0.0)
        t_diff = t - self.vars.t_previous
        closest_index = self.closest_cal(x,y)
        e = self.distance(x,y,closest_index)
        heading = self.head_calculator(closest_index)
        print(heading," ", yaw, " ", self._current_speed, " ", self._desired_speed, " ", e)
        if self.vars.h_previous>1 and heading<=-1:
            self.vars.c += 1.0
        elif self.vars.h_previous<-1 and heading>=1:
            self.vars.c -= 1.0

        yaw = self._pi * yaw / abs(yaw) - yaw
        
        if self.vars.yaw_prev<=-2.5 and yaw>2.5:
            self.vars.k -= 1.0
        elif self.vars.yaw_prev>=2.5 and yaw<-2.5:
            self.vars.k += 1.0
        
        psi = (heading + self.vars.c*self._pi + yaw + self.vars.k*self._2pi)
        cons = -1 if self.check_left(x, y, closest_index) else 1
        e = e*cons
        #delta = psi + cons * np.arctan(self._kStan * e / v_f)
        X = [[e], [(e - self.vars.e_prev)/t_diff], [psi], [(psi - self.vars.psi_previous)/t_diff]]
        K = self.LQR(v_f)
        delta = K @ X
        self.vars.e_prev = e
        self.vars.h_previous = heading
        self.vars.yaw_prev = yaw
        self.vars.psi_previous = psi
        print (X)
        print (delta, " ", self.vars.c)
        return (delta)
    
    def LQR(self, Vx):
        m = 1140
        Iz = 1436.24
        Lf = 1.165
        Lr = 1.165
        Cf = 155494.663
        Cr = 155494.663
        if Vx<=1.5:
            Vx = 1.5
        A = [[0,                     1,                0,                            0],
             [0,       -(Cf+Cr)/(m*Vx),        (Cf+Cr)/m,         (Lr*Cr-Lf*Cf)/(m*Vx)],
             [0,                     0,                0,                            1],
             [0, (Lr*Cr-Lf*Cf)/(Iz*Vx), (Lf*Cf-Lr*Cr)/Iz, -(Lf*Lf*Cf+Lr*Lr*Cr)/(Iz*Vx)]]
        B = [[0],
             [Cf/m],
             [0],
             [Lf*Cf/Iz]]
        Q = [[50, 0, 0, 0],
             [0, 5, 0, 0],
             [0, 0, 10, 0],
             [0, 0, 0, 0]]
        R = [[50]]
        K, S, E = control.lqr(A, B, Q, R)
        return K
    
    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        ######################################################
        ######################################################
        # MODULE 7: DECLARE USAGE VARIABLES HERE
        ######################################################
        ######################################################
        """
            Use 'self.vars.create_var(<variable name>, <default value>)'
            to create a persistent variable (not destroyed at each iteration).
            This means that the value can be stored for use in the next
            iteration of the control loop.

            Example: Creation of 'v_previous', default value to be 0
            self.vars.create_var('v_previous', 0.0)

            Example: Setting 'v_previous' to be 1.0
            self.vars.v_previous = 1.0

            Example: Accessing the value from 'v_previous' to be used
            throttle_output = 0.5 * self.vars.v_previous
        """
        self.vars.create_var('v_previous', 0.0)
        self.vars.create_var('time_previous', 0.0)
        self.vars.create_var('throttle_previous', 0.0)
        self.vars.create_var('integral_previous', 0.0)
        self.vars.create_var('err_previous', 0.0)

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """

            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LONGITUDINAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a longitudinal controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change these outputs with the longitudinal controller. Note that
            # brake_output is optional and is not required to pass the
            # assignment, as the car will naturally slow down over time.
            #PID
            #gains
            if t<=0.75:
                k_p=1.01
                k_i=0
                k_d=0.06
            else:
                k_p=1.15
                k_i=0.45
                k_d=0.05
            #feedback
            err=v_desired-v #error
            ts = t - self.vars.time_previous
            
            integral = self.vars.integral_previous+(err*ts)
            deriv = (err-self.vars.err_previous)/ts

            final = (k_p*err)+(k_i*integral) + (k_d*deriv)
            #feedforward:
            look_ahead = waypoints[len(waypoints)-1]
            desired_vel = look_ahead[2]
            if desired_vel <=6:
                feedfwd = 0.10 + desired_vel/6*(0.6-0.10)
            elif v_desired<=11.5:
                feedfwd = 0.6 + desired_vel/(11.5-5)*(0.8-0.6)
            else:
                feedfwd = 0.8 + (desired_vel-11.5)/85

            throttle = final + feedfwd
            throttle = min(throttle,1)
            throttle = max(throttle,0)
            throttle_output = throttle
            ######################################################
            ######################################################
            # MODULE 7: IMPLEMENTATION OF LATERAL CONTROLLER HERE
            ######################################################
            ######################################################
            """
                Implement a lateral controller here. Remember that you can
                access the persistent variables declared above here. For
                example, can treat self.vars.v_previous like a "global variable".
            """
            
            # Change the steer output with the lateral controller. 
            steer_output    = 0
            
            #stanley implementation:-
            '''
            k_e=3
            k_s=15

            #heading error calculation:-
            yaw_h=np.arctan2(waypoints[-1][1]-waypoints[0][1],waypoints[-1][0]-waypoints[0][0])
            diff_h=yaw_h-yaw
            if diff_h>np.pi:
                diff_h = diff_h-2*np.pi
            if diff_h<-np.pi:
                diff_h = diff_h+2*np.pi

            #crosstrack error calculation:-
            xy=np.array([x, y])
            err_ct = np.min(np.sum((xy - np.array(waypoints)[:, :2])**2, axis=1))
            yaw_ct = np.arctan2(y-waypoints[0][1],x-waypoints[0][0])

            diff_ct = yaw_h-yaw_ct
            if diff_ct>np.pi:
                diff_ct = diff_ct-2*np.pi
            if diff_ct<-np.pi:
                diff_ct = diff_ct+2*np.pi
            if diff_ct>0:
                err_ct = abs(err_ct)
            else:
                err_ct = -abs(err_ct)
            yaw_diff_ct=np.arctan((k_e*err_ct)/(k_s+v))

            #steering control law:-
            steer_expect = diff_h + yaw_diff_ct
            if steer_expect > np.pi:
                steer_expect = steer_expect-2 * np.pi
            if steer_expect < - np.pi:
                steer_expect = steer_expect+2 * np.pi
            steer_expect = np.min([1.22, steer_expect])
            steer_expect = np.max([-1.22, steer_expect])

            steer_output=steer_expect
            '''
            '''
            #pure pursuit:-
            kp_d = 0.1
            L = 3
            min_ld = 5.0

            rear_x = x - (L * np.cos(yaw)/2)
            rear_y = y - (L * np.sin(yaw)/2)
            look_ahead = max(min_ld, kp_d * v)
            for w in waypoints:
                d = math.sqrt((w[0] - rear_x)**2 + (w[1] - rear_y)**2)
                if d > look_ahead:
                    c = w
                    break
                else:
                    c = waypoints[0]
            alpha = math.atan2(c[1] - rear_y, c[0] - rear_x) - yaw

            steer_output = math.atan2(2 * L * np.sin(alpha), look_ahead)
            steer_output = min(1.22, steer_output)
            steer_output = max(-1.22, steer_output)
            '''
            #PID Steering:-
            k_p_s = 1.0
            k_i_s = 5.0
            k_d_s = 1.0
            k_e=3
            k_s=15
            
            yaw_h=np.arctan2(waypoints[-1][1]-waypoints[0][1],waypoints[-1][0]-waypoints[0][0])
            diff_h=yaw_h-yaw
            if diff_h>np.pi:
                diff_h = diff_h-2*np.pi
            if diff_h<-np.pi:
                diff_h = diff_h+2*np.pi

            #crosstrack error calculation:-
            xy=np.array([x, y])
            err_ct = np.min(np.sum((xy - np.array(waypoints)[:, :2])**2, axis=1))
            yaw_ct = np.arctan2(y-waypoints[0][1],x-waypoints[0][0])

            diff_ct = yaw_h-yaw_ct
            if diff_ct>np.pi:
                diff_ct = diff_ct-2*np.pi
            if diff_ct<-np.pi:
                diff_ct = diff_ct+2*np.pi
            if diff_ct>0:
                err_ct = abs(err_ct)
            else:
                err_ct = -abs(err_ct)
            yaw_diff_ct=np.arctan((k_e*err_ct)/(k_s+v))
            
            err_steering = diff_h + yaw_diff_ct

            steering_integral = self.vars.steering_integral_previous + (err_steering * ts)
            steering_deriv = (err_steering - self.vars.steering_err_previous) / ts

            steer_expect = (k_p_s*err_steering)+(k_i_s*steering_integral) + (k_d_s * steering_deriv)
            '''
            if steer_expect > np.pi:
                steer_expect = steer_expect-2 * np.pi
            if steer_expect < - np.pi:
                steer_expect = steer_expect+2 * np.pi
            '''
            steer_expect = np.min([1.22, steer_expect])
            steer_expect = np.max([-1.22, steer_expect])

            steer_output=steer_expect
            
            #LQR implementation
            '''
            steer_output    = self.calculate_steer(t,x,y,yaw,v)
            '''
            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
       self.vars.v_previous = v  # Store forward speed to be used in next step
       self.vars.time_previous = t
       self.vars.integral_previous = integral
       self.vars.throttle_previous = throttle
       self.vars.steering_err_previous = err_steering
       self.vars.err_previous = err
       self.vars.sttering_integral_previous = steering_integral
