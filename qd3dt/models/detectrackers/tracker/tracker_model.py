import numpy as np

import torch
from filterpy.kalman import KalmanFilter


class KalmanBox3DTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # coord3d - array of detections [x,y,z,theta,l,w,h]
        # X,Y,Z,theta, l, w, h, dX, dY, dZ
        self.kf = KalmanFilter(dim_x=10, dim_z=7)
        self.kf.F = np.array([
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
            [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        ])

        self.kf.H = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
            [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
        ])

        # state uncertainty, give high uncertainty to
        self.kf.P[7:, 7:] *= 1000.
        # the unobservable initial velocities, covariance matrix
        self.kf.P *= 10.

        # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
        self.kf.Q[7:, 7:] *= 0.01
        self.kf.x[:7] = bbox3D.reshape((7, 1))

        self.time_since_update = 0
        self.id = KalmanBox3DTracker.count
        KalmanBox3DTracker.count += 1
        self.nfr = 5
        self.history = []
        self.prev_ref = bbox3D
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first
        # detection
        self.age = 0
        self.info = info  # other info

    @property
    def obj_state(self):
        return self.kf.x.flatten()

    def _update_history(self, bbox3D):
        self.history = self.history[1:] + [bbox3D - self.prev_ref]

    def _init_history(self, bbox3D):
        self.history = [bbox3D - self.prev_ref] * self.nfr

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        self.time_since_update = 0

        # orientation correction
        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2

        new_theta = bbox3D[3]
        if new_theta >= np.pi:
            new_theta -= np.pi * 2  # make the theta still in the range
        if new_theta < -np.pi:
            new_theta += np.pi * 2
        bbox3D[3] = new_theta

        predicted_theta = self.kf.x[3]
        # if the angle of two theta is not acute angle
        if np.pi / 2.0 < abs(new_theta - predicted_theta) < np.pi * 3 / 2.0:
            self.kf.x[3] += np.pi
            if self.kf.x[3] > np.pi:
                self.kf.x[3] -= np.pi * 2  # make the theta still in the range
            if self.kf.x[3] < -np.pi:
                self.kf.x[3] += np.pi * 2

        # now the angle is acute: < 90 or > 270, convert the case of > 270 to
        # < 90
        if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
            if new_theta > 0:
                self.kf.x[3] += np.pi * 2
            else:
                self.kf.x[3] -= np.pi * 2

        # Update the bbox3D
        self.kf.update(bbox3D)

        if self.kf.x[3] >= np.pi:
            self.kf.x[3] -= np.pi * 2  # make the theta still in the range
        if self.kf.x[3] < -np.pi:
            self.kf.x[3] += np.pi * 2
        self.info = info
        self.prev_ref = self.kf.x.flatten()[:7]

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        self.kf.predict()
        if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
        if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.kf.x.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.kf.x.flatten()

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


class LSTM3DTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, device, lstm, bbox3D, info):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # coord3d - array of detections [x,y,z,theta,l,w,h]
        # X,Y,Z,theta, l, w, h, dX, dY, dZ

        self.device = device
        self.lstm = lstm
        self.loc_dim = self.lstm.loc_dim
        self.id = LSTM3DTracker.count
        LSTM3DTracker.count += 1
        self.nfr = 5
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.init_flag = True
        self.age = 0

        self.obj_state = np.hstack([bbox3D.reshape((7, )), np.zeros((3, ))])
        self.history = np.tile(
            np.zeros_like(bbox3D[:self.loc_dim]), (self.nfr, 1))
        self.ref_history = np.tile(bbox3D[:self.loc_dim], (self.nfr + 1, 1))
        self.avg_angle = bbox3D[3]
        self.avg_dim = np.array(bbox3D[4:])
        self.prev_obs = bbox3D.copy()
        self.prev_ref = bbox3D[:self.loc_dim].copy()
        self.info = info
        self.hidden_pred = self.lstm.init_hidden(self.device)
        self.hidden_ref = self.lstm.init_hidden(self.device)

    @staticmethod
    def fix_alpha(angle: float) -> float:
        return (angle + np.pi) % (2 * np.pi) - np.pi

    @staticmethod
    def update_array(origin_array: np.ndarray,
                     input_array: np.ndarray) -> np.ndarray:
        new_array = origin_array.copy()
        new_array[:-1] = origin_array[1:]
        new_array[-1:] = input_array
        return new_array

    def _update_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = self.update_array(
            self.history, self.ref_history[-1] - self.ref_history[-2])
        # align orientation history
        self.history[:, 3] = self.history[-1, 3]
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:,
                                                             3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def _init_history(self, bbox3D):
        self.ref_history = self.update_array(self.ref_history, bbox3D)
        self.history = np.tile([self.ref_history[-1] - self.ref_history[-2]],
                               (self.nfr, 1))
        self.prev_ref[:self.loc_dim] = self.obj_state[:self.loc_dim]
        if self.loc_dim > 3:
            self.avg_angle = self.fix_alpha(self.ref_history[:,
                                                             3]).mean(axis=0)
            self.avg_dim = self.ref_history.mean(axis=0)[4:]
        else:
            self.avg_angle = self.prev_obs[3]
            self.avg_dim = np.array(self.prev_obs[4:])

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        if self.age == 1:
            self.obj_state[:self.loc_dim] = bbox3D[:self.loc_dim].copy()

        if self.loc_dim > 3:
            # orientation correction
            self.obj_state[3] = self.fix_alpha(self.obj_state[3])
            bbox3D[3] = self.fix_alpha(bbox3D[3])

            # if the angle of two theta is not acute angle
            # make the theta still in the range
            curr_yaw = bbox3D[3]
            if np.pi / 2.0 < abs(curr_yaw -
                                 self.obj_state[3]) < np.pi * 3 / 2.0:
                self.obj_state[3] += np.pi
                if self.obj_state[3] > np.pi:
                    self.obj_state[3] -= np.pi * 2
                if self.obj_state[3] < -np.pi:
                    self.obj_state[3] += np.pi * 2

            # now the angle is acute: < 90 or > 270,
            # convert the case of > 270 to < 90
            if abs(curr_yaw - self.obj_state[3]) >= np.pi * 3 / 2.0:
                if curr_yaw > 0:
                    self.obj_state[3] += np.pi * 2
                else:
                    self.obj_state[3] -= np.pi * 2

        with torch.no_grad():
            refined_loc, self.hidden_ref = self.lstm.refine(
                torch.from_numpy(self.obj_state[:self.loc_dim]).view(
                    1, self.loc_dim).float().to(self.device),
                torch.from_numpy(bbox3D[:self.loc_dim]).view(
                    1, self.loc_dim).float().to(self.device),
                torch.from_numpy(self.prev_ref[:self.loc_dim]).view(
                    1, self.loc_dim).float().to(self.device),
                torch.from_numpy(info).view(1, 1).float().to(self.device),
                self.hidden_ref)

        refined_obj = refined_loc.cpu().numpy().flatten()
        if self.loc_dim > 3:
            refined_obj[3] = self.fix_alpha(refined_obj[3])

        self.obj_state[:self.loc_dim] = refined_obj
        self.prev_obs = bbox3D

        if np.pi / 2.0 < abs(bbox3D[3] - self.avg_angle) < np.pi * 3 / 2.0:
            for r_indx in range(len(self.ref_history)):
                self.ref_history[r_indx][3] = self.fix_alpha(
                    self.ref_history[r_indx][3] + np.pi)

        if self.init_flag:
            self._init_history(refined_obj)
            self.init_flag = False
        else:
            self._update_history(refined_obj)

        self.info = info

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        with torch.no_grad():
            pred_loc, hidden_pred = self.lstm.predict(
                torch.from_numpy(self.history[..., :self.loc_dim]).view(
                    self.nfr, -1, self.loc_dim).float().to(self.device),
                torch.from_numpy(self.obj_state[:self.loc_dim]).view(
                    -1, self.loc_dim).float().to(self.device),
                self.hidden_pred)

        pred_state = self.obj_state.copy()
        pred_state[:self.loc_dim] = pred_loc.cpu().numpy().flatten()
        pred_state[7:] = pred_state[:3] - self.prev_ref[:3]
        if self.loc_dim > 3:
            pred_state[3] = self.fix_alpha(pred_state[3])

        if update_state:
            self.hidden_pred = hidden_pred
            self.obj_state = pred_state

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return pred_state.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state.flatten()

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


class DummyTracker(object):
    """
    This class represents the internel state of individual tracked objects
    observed as bbox.
    """
    count = 0

    def __init__(self, bbox3D, info):
        """
        Initialises a tracker using initial bounding box.
        """
        # define constant velocity model
        # coord3d - array of detections [x,y,z,theta,l,w,h]
        self.nfr = 5
        self.hits = 1
        self.hit_streak = 0
        self.time_since_update = 0
        self.age = 0

        self.obj_state = np.hstack([bbox3D.reshape((7, )), np.zeros((3, ))])
        self.history = [np.zeros_like(bbox3D)] * self.nfr
        self.prev_ref = bbox3D
        self.motion_momentum = 0.9
        self.info = info

    def _update_history(self, bbox3D):
        self.history = self.history[1:] + [bbox3D - self.prev_ref]

    def _init_history(self, bbox3D):
        self.history = [bbox3D - self.prev_ref] * self.nfr

    def update(self, bbox3D, info):
        """
        Updates the state vector with observed bbox.
        """
        # Update the bbox3D
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        self.obj_state += self.motion_momentum * (
            np.hstack([bbox3D.reshape(
                (7, )), np.zeros((3, ))]) - self.obj_state)
        self.prev_ref = bbox3D
        self.info = info

    def predict(self, update_state: bool = True):
        """
        Advances the state vector and returns the predicted bounding box
        estimate.
        """
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        return self.obj_state.flatten()

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.obj_state.flatten()

    def get_history(self):
        """
        Returns the history of estimates.
        """
        return self.history


TRACKER_MODEL_ZOO = {
    'KalmanBox3DTracker': KalmanBox3DTracker,
    'LSTM3DTracker': LSTM3DTracker,
    'DummyTracker': DummyTracker,
}


def get_tracker(tracker_model_name) -> object:
    tracker_model = TRACKER_MODEL_ZOO.get(tracker_model_name, None)
    if tracker_model is None:
        raise NotImplementedError

    return tracker_model
