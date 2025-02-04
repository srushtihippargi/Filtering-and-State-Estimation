
from mimetypes import init
from os import stat
from statistics import mean
from scipy.linalg import block_diag
from copy import deepcopy, copy
import numpy as np

from system.RobotState import RobotState
from utils.Landmark import LandmarkList
from utils.utils import wrap2Pi

# import InEKF lib
from scipy.linalg import logm, expm


class InEKF:
    # InEKF construct an instance of this class
    #
    # Inputs:
    #   system: system and noise models
    #   init:   initial state mean and covariance

    def __init__(self, system, init):

        self.gfun = system.gfun  # motion model
        # self.hfun = system.hfun  # measurement model
        # self.Gfun = init.Gfun  # Jocabian of motion model
        # self.Vfun = init.Vfun  
        # self.Hfun = init.Hfun  # Jocabian of measurement model
        self.W = system.W # motion noise covariance
        self.V = system.V # measurement noise covariance
        
        self.mu = init.mu
        self.Sigma = init.Sigma

        self.state_ = RobotState()
        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])
        self.state_.setState(X)
        self.state_.setCovariance(init.Sigma)

    
    def prediction(self, u, Sigma, mu, step):
        if step != 0 :
            self.Sigma = Sigma
            self.mu = mu
        state_vector = np.zeros(3)
        state_vector[0] = self.mu[0,2]
        state_vector[1] = self.mu[1,2]
        state_vector[2] = np.arctan2(self.mu[1,0], self.mu[0,0])
        H_prev = self.pose_mat(state_vector)
        state_pred = self.gfun(state_vector, u)
        H_pred = self.pose_mat(state_pred)

        u_se2 = logm(np.linalg.inv(H_prev) @ H_pred)

        ###############################################################################
        # TODO: Propagate mean and covairance (You need to compute adjoint AdjX)      #
        ###############################################################################
        adjX = self.adjoint(H_prev)
        self.mu_pred, self.sigma_pred = self.propagation(u_se2, adjX, self.mu, self.Sigma, self.W)

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################

        return np.copy(self.mu_pred), np.copy(self.sigma_pred)

    def propagation(self, u, adjX, mu, Sigma , W):
        self.mu = mu
        self.Sigma = Sigma
        self.W = W
        ###############################################################################
        # TODO: Complete propagation function                                         #
        # Hint: you can save predicted state and cov as self.X_pred and self.P_pred   #
        #       and use them in the correction function                               #
        ###############################################################################

        self.mu_pred = self.mu @ expm(u)
        self.sigma_pred = self.Sigma + adjX @ self.W @ adjX.T
        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        return np.copy(self.mu_pred), np.copy(self.sigma_pred)
        
    def correction(self, Y1, Y2, z, landmarks, mu_pred, sigma_pred):
        landmark1 = landmarks.getLandmark(z[2].astype(int))
        landmark2 = landmarks.getLandmark(z[5].astype(int))
        self.mu_pred = mu_pred
        self.sigma_pred = sigma_pred
        ###############################################################################
        # TODO: Implement the correction step for InEKF                               #
        # Hint: save your corrected state and cov as X and self.Sigma                 #
        # Hint: you can use landmark1.getPosition()[0] to get the x position of 1st   #
        #       landmark, and landmark1.getPosition()[1] to get its y position        #
        ###############################################################################
        G1 = np.zeros((3,3))
        G1[0,2] = 1
        G2 = np.zeros((3,3))
        G2[1,2] = 1
        G3 = np.zeros((3,3))
        G3[0,1] = -1
        G3[1,0] = 1

        b1 = np.array([landmark1.getPosition()[0], landmark1.getPosition()[1],1])
        b2 = np.array([landmark2.getPosition()[0], landmark2.getPosition()[1],1])
        H1 = np.array([[-1,0,landmark1.getPosition()[1]],\
                      [0,-1,-landmark1.getPosition()[0]]])
        H2 = np.array([[-1,0,landmark2.getPosition()[1]],\
                      [0,-1,-landmark2.getPosition()[0]]])
        H = np.vstack((H1,H2))

        R = self.mu_pred[:2,:2]
        R = block_diag(R,R)
        nu1 = self.mu_pred @ Y1.T - b1
        nu2 = self.mu_pred @ Y2.T - b2

        N = R @ block_diag(self.V, self.V) @ R.T
        S = H @ self.sigma_pred @ H.T + N
        K = self.sigma_pred @ H.T @ np.linalg.inv(S)

        temp = np.zeros((2,3))
        temp[:2,:2] = np.eye(2)
        delta1 = K[:3,:2] @ temp @ nu1
        delta2 = K[:3,2:4] @ temp @ nu2
        delta = delta1 + delta2
        self.mu = np.dot(expm(delta[0] * G1 + delta[1] * G2 + delta[2] * G3), self.mu_pred)
        self.Sigma = (np.eye(3) - K @ H) @ self.sigma_pred @ (np.eye(3) - K @ H).T + K @ N @ K.T

        X = np.array([self.mu[0,2], self.mu[1,2], np.arctan2(self.mu[1,0], self.mu[0,0])])

        ###############################################################################
        #                         END OF YOUR CODE                                    #
        ###############################################################################
        self.state_.setState(X)
        self.state_.setCovariance(self.Sigma)
        return np.copy(X), np.copy(self.Sigma), np.copy(self.mu)

    def getState(self):
        return deepcopy(self.state_)

    def setState(self, state):
        self.state_ = state

    def pose_mat(self, X):
        x = X[0]
        y = X[1]
        h = X[2]
        H = np.array([[np.cos(h),-np.sin(h),x],\
                      [np.sin(h),np.cos(h),y],\
                      [0,0,1]])
        return H
    
    def adjoint(self, X):
        out = np.eye(3)
        out[:2,:2] = X[:2,:2]
        out[0,2] = X[1,2]
        out[1,2] = -X[0,2]
        return out