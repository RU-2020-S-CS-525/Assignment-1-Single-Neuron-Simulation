import numpy as np
import matplotlib.pyplot as plt

class LIF(object):
    #leaky integrate-and-fire model
    #C_m \frac{dV}{dt} = I(t) - frac{V_m (t)}{R_m}
    def __init__(self, capitance, resistance, vThreshold, vRest, dt = 0.1, leaky = True):
        #float capitance: C_m
        #float resistance: R_m
        #float vThreshold: threshold votage V_t
        #float vRest: rest votage V_r
        #float dt: simulation step size, msec
        #bool leaky: True: leaky integrate-and-fire model; False: leaky-free integrate-and-fire model 
        super(LIF, self).__init__()
        self.capitance = capitance
        self.resistance = resistance
        self.vThreshold = vThreshold
        self.vRest = vRest
        self.dt = dt
        self.leakyFactor = 1 if leaky else 0
        return

    def _update(self, tempCurrent, tempVotage):
        #IN
        #np.ndarray tempCurrent, dtype = np.float64, shape = (1, n): n different input current
        #np.ndarray tempVotage, dtype = np.float64, shpape = (1, n): n different membrance potential
        #OUT
        #np.ndarray tempVotage, dtype = np.float64, shpape = (1, n): updated membrance potential
        #np.ndarray spike, dtype = np.bool, shpape = (1, n): True: fire; False: not fire
        dV = (tempCurrent - self.leakyFactor * tempVotage / self.resistance) * self.dt / self.capitance
        tempVotage = tempVotage + dV
        spike = tempVotage >= self.vThreshold
        tempVotage[spike] = self.vRest
        return tempVotage, spike

    def simulate(self, current):
        #IN
        #np.ndarray current, dtype = np.float64, shape = (k, n): input current, |t| = k * dt, n different currents
        #OUT
        #np.ndarray votage, dtype = np.float64, shape = (k, n): membrance potential
        #np.ndarray spike, dtype = np.bool, shape = (k, n): spiking behavior
        self.votage = np.empty_like(current, dtype = np.float64)
        self.spike = np.empty_like(current, dtype = np.bool)
        self.stepNum, self.simulationNum = current.shape
        
        tempVotage = np.full((1, self.simulationNum), self.vRest, dtype = np.float64)
        for i in range(self.stepNum):
            self.votage[i], self.spike[i] = self._update(current[i], tempVotage)
            tempVotage = self.votage[i]
        return self.votage, self.spike

    def plot(self):
        color = ['b', 'g', 'r', 'c', 'm', 'y']
        time = np.array(range(self.stepNum), dtype = np.float64) * self.dt
        for i in range(self.simulationNum):
            plt.plot(time, self.votage[:, i], c = color[i])
            plt.scatter(time[self.spike[:, i]], np.full(np.sum(self.spike[:, i]), self.vThreshold, dtype = np.float64), c = color[i], marker = 'o')
        plt.show()
        return

def Q1(currentList, timeWindow, capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True):
    #IN
    #list currentList: [float current]
    #float timeWindow: simulation time
    #float capitance: C_m
    #float resistance: R_m
    #float vThreshold: threshold votage V_t
    #float vRest: rest votage V_r
    #float dt: simulation step size, msec
    #bool leaky: True: leaky integrate-and-fire model; False: leaky-free integrate-and-fire model 
    stepNum = timeWindow / dt
    lif = LIF(capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True)
    current = np.full(stepNum, )

if __name__ == '__main__':
    capitance = 1
    resistance = 8
    vThreshold = 5
    vRest = 1
    lif = LIF(capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True)
    current = np.full((10000, 3), 1, dtype = np.float64)
    current[:, 1] = 0.8
    current[:, 2] = 0.5
    lif.simulate(current)
    lif.plot()