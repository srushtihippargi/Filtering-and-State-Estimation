import numpy as np
from scipy.linalg import block_diag
from copy import deepcopy, copy

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

class EKF:

    def __init__(self, system, init):
        # EKF Construct an instance of this class
        # Inputs:
        #   system: system and noise models
        #   init:   initial state mean and covariance
        self.gfun = system.gfun  # motion model
        self.hfun = system.hfun  # measurement model
        self.Gfun = init.Gfun  # Jocabian of motion model
        self.Vfun = init.Vfun  # Jocabian of motion model
        self.Hfun = init.Hfun  # Jocabian of measurement model
        self.M = system.M # motion noise covariance
        self.Q = system.Q # measurement noise covariance

        self.state_ = RobotState()

        # init state
        self.state_.setState(init.mu)
        self.state_.setCovariance(init.Sigma)


    ## Do prediction and set state in RobotState()
    def prediction(self, u, X, P, step):
        if step == 0:
            X = self.state_.getState()
            P = self.state_.getCovariance()
        else:
            X = X
            P = P
        ###############################################################################
        # TODO: Implement the prediction step for EKF                                 #
        # Hint: save your predicted state and cov as X_pred and P_pred                #
        ###############################################################################
        
        # Evaluate G with mean and input
        G = self.Gfun(X, u)
        # Evaluate V with mean and input
        V = self.Vfun(X, u)
        # Propoagate mean through non-linear dynamics
        X_pred = self.gfun(X, u)
        # Update covariance with G,V and M(u)
        P_pred = G @ P @ G.T + V @ self.M(u) @ V.T

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X_pred)
        self.state_.setCovariance(P_pred)
        return np.copy(X_pred), np.copy(P_pred)


    def correction(self, z, landmarks, X, P):
        # EKF correction step
        #
        # Inputs:
        #   z:  measurement
        X_predict = X
        P_predict = P
        
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))

        ###############################################################################
        # TODO: Implement the correction step for EKF                                 #
        # Hint: save your corrected state and cov as X and P                          #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################

        z_hat1 = self.hfun(landmark1.getPosition()[0],landmark1.getPosition()[1],X_predict)
        z_hat2 = self.hfun(landmark2.getPosition()[0],landmark2.getPosition()[1],X_predict)
        z_hat = np.hstack((z_hat1,z_hat2))

        # evaluate measurement Jacobian at current operating point
        H_1 = self.Hfun(landmark1.getPosition()[0],landmark1.getPosition()[1],X_predict,z_hat1)
        H_2 = self.Hfun(landmark2.getPosition()[0],landmark2.getPosition()[1],X_predict,z_hat2)
        H = np.vstack((H_1,H_2))
        
        # compute innovation statistics
        # We know here z[1] is an angle
        z_no_id = np.hstack((z[0:2],z[3:5]))
        v = z_no_id - z_hat  # innovation

        S = H @ P_predict @ H.T + block_diag(self.Q,self.Q) # innovation covariance
      
        K = P_predict @ H.T @ np.linalg.inv(S)  # Kalman (filter) gain

        # correct the predicted state statistics
        diff = [
                wrap2Pi(z[0] - z_hat1[0]),
                z[1] - z_hat1[1],
                wrap2Pi(z[3] - z_hat2[0]),
                z[4] - z_hat2[1]]
        X = X_predict + K @ diff
        X[2] = wrap2Pi(X[2])
        U = np.eye(np.shape(X)[0]) - K @ H
        P = U @ P_predict @ U.T + K @ block_diag(self.Q,self.Q) @ K.T
        
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        self.state_.setState(X)
        self.state_.setCovariance(P)
        return np.copy(X), np.copy(P)


    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state