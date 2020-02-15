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

        #dV = (I(t) - frac{v_m (t)}{R_m}) * dt / C_m
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

    def getFiringNum(self):
        return np.sum(self.spike, axis = 0)

    def plot(self, currentList):
        #list currentList: [float current]
        color = ['b', 'g', 'r', 'c', 'm', 'y']
        if self.simulationNum > len(color):
            print('E: too many currents')
            exit(-1)

        time = np.array(range(self.stepNum), dtype = np.float64) * self.dt
        for i in range(self.simulationNum):
            line, = plt.plot(time, self.votage[:, i], c = color[i])
            point = plt.scatter(time[self.spike[:, i]], np.full(np.sum(self.spike[:, i]), self.vThreshold, dtype = np.float64), c = color[i], marker = 'o')
            line.set_label('I = ' + str(currentList[i]) + 'mA')
            point.set_label('spiking indicator')
        plt.xlabel('time (msec)')
        plt.ylabel('votage (mV)')
        plt.legend(loc = 2)
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

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = len(currentList)

    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init LIF model
    lif = LIF(capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True)

    #simulate and plot
    lif.simulate(current)
    lif.plot(currentList)
    return

def Q2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True):
    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = int(np.ceil((maxCurrent - minCurrent) / currentStepSize))
    currentList = np.array(range(simulationNum), dtype = np.float64) * currentStepSize + minCurrent
    
    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init LIF model
    lif = LIF(capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True)
    #simulate
    lif.simulate(current)
    rate = lif.getFiringNum() / timeWindow * 1000

    #plot
    plt.plot(currentList, rate)
    plt.xlabel('current (mA)')
    plt.ylabel('firing rate (Hz)')
    plt.show()

if __name__ == '__main__':
    currentList = [0.5, 0.8, 1]
    timeWindow = 1000
    capitance = 50
    resistance = 8
    vThreshold = 3
    vRest = 1
    Q1(currentList, timeWindow, capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True)

    minCurrent = 0.1
    maxCurrent = 1
    currentStepSize = 0.1
    Q2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vThreshold, vRest, dt = 0.01, leaky = True)