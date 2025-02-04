import rospy
from filter.InEKF import InEKF 

class InEKF_ROS(InEKF):
    def __init__(self, system, init):
        super().__init__(system, init)

    def prediction(self, u, Sigma, mu, step):
        mu_pred, sigma_pred = super().prediction(u, Sigma, mu, step)
        return mu_pred, sigma_pred

    def propagation(self, u, adjX, mu, Sigma , W):
        mu_pred, sigma_pred = super().propagation(u, adjX, mu, Sigma, W)
        return mu_pred, sigma_pred

    def correction(self, Y1, Y2, z, landmarks, mu_pred, sigma_pred):
        X, Sigma, mu = super().correction(Y1, Y2, z, landmarks, mu_pred, sigma_pred)
        self.state_.setTime(rospy.Time.now())
        return X, Sigma, mu