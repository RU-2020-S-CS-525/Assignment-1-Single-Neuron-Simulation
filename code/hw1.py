import numpy as np
import matplotlib.pyplot as plt

class LIF(object):
    #leaky integrate-and-fire model
    #C_m \frac{dV}{dt} = I(t) - frac{V_m (t)}{R_m}
    def __init__(self, capitance = 1, resistance = 20, vRest = -65, vThreshold = 30, dt = 0.01, leaky = True):
        #float capitance: C_m
        #float resistance: R_m
        #float vRest: rest votage V_r
        #float vThreshold: threshold votage V_t
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
        #np.ndarray tempVotage, dtype = np.float64, shape = (1, n): n different membrance potential
        #OUT
        #np.ndarray tempVotage, dtype = np.float64, shpape = (1, n): updated membrance potential
        #np.ndarray spike, dtype = np.bool, shpape = (1, n): True: fire; False: not fire

        #dV = (I(t) - frac{v_m (t)}{R_m}) * dt / C_m
        dV = (tempCurrent - self.leakyFactor * tempVotage / self.resistance) * self.dt / self.capitance
        tempVotage = tempVotage + dV

        #get spike and reset
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
        
        #init v
        tempVotage = np.full((1, self.simulationNum), self.vRest, dtype = np.float64)
        #loop
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

class Izhikevich(object):
    #Izhikevich model
    #frac{dv}{dt} = 0.04 v^2 + 5 v + 140 - u + I
    #frac{du}{dt} = a (b v - u)
    def __init__(self, a = 0.02, b = 0.2, c = -65, d = 2, vThreshold = 30, dt = 0.01):
        #float a: time scale of u
        #float b: sensitivity of u to v
        #float c: after-spike reset v
        #float d: after-spike reset u
        super(Izhikevich, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.vThreshold = vThreshold
        self.dt = dt
        self.halfDt = self.dt / 2 #used for update v stably

    def _update(self, tempCurrent, tempVotage, tempU):
        #IN
        #np.ndarray tempCurrent, dtype = np.float64, shape = (1, n): n different input currents
        #np.ndarray tempVotage, dtype = np.float64, shape = (1, n): n different membrance potentials
        #np.ndarray tempU, dtype = np.float64, shape = (1, n): n different recovary variables
        #OUT
        #np.ndarray tempVotage, dtype = np.float64, shpape = (1, n): updated membrance potential
        #np.ndarray tempU, dtype = np.float64, shape = (1, n): updated recovary variables
        #np.ndarray spike, dtype = np.bool, shpape = (1, n): True: fire; False: not fire

        #update V first half
        dV = (0.04 * np.square(tempVotage) + 5 * tempVotage + 140 - tempU + tempCurrent) * self.halfDt
        tempVotage = tempVotage + dV

        #update U
        dU = self.a * (self.b * tempVotage - tempU) * self.dt
        tempU = tempU + dU

        #update V second half
        dV = (0.04 * np.square(tempVotage) + 5 * tempVotage + 140 - tempU + tempCurrent) * self.halfDt
        tempVotage = tempVotage + dV

        #get spike and reset
        spike = tempVotage >= self.vThreshold
        tempVotage[spike] = self.c
        tempU[spike] = tempU[spike] + self.d
        return tempVotage, tempU, spike

    def simulate(self, current):
        #IN
        #np.ndarray current, dtype = np.float64, shape = (k, n): input current, |t| = k * dt, n different currents
        #OUT
        #np.ndarray votage, dtype = np.float64, shape = (k, n): membrance potential
        #np.ndarray spike, dtype = np.bool, shape = (k, n): spiking behavior
        self.votage = np.empty_like(current, dtype = np.float64)
        self.spike = np.empty_like(current, dtype = np.bool)
        self.stepNum, self.simulationNum = current.shape

        #init v, u
        tempVotage = np.full((1, self.simulationNum), self.c, dtype = np.float64)
        tempU = np.full((1, self.simulationNum), self.b * self.c, dtype = np.float64)
        #loop
        for i in range(self.stepNum):
            self.votage[i], tempU, self.spike[i] = self._update(current[i], tempVotage, tempU)
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
    lif = LIF(capitance, resistance, vThreshold, vRest, dt, leaky)
    #simulate
    lif.simulate(current)
    rate = lif.getFiringNum() / timeWindow * 1000

    #plot
    plt.plot(currentList, rate)
    plt.xlabel('current (mA)')
    plt.ylabel('firing rate (Hz)')
    plt.show()

def Q4(currentList, timeWindow, a = 0.02, b = 0.2, c = -65, d = 2, vThreshold = 30, dt = 0.01):
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
    izhikevich = Izhikevich(a, b, c, d, vThreshold, dt)

    #simulate and plot
    izhikevich.simulate(current)
    izhikevich.plot(currentList)
    return

if __name__ == '__main__':
    # currentList = [1.45, 1.55, 1.65]
    # timeWindow = 1000
    # capitance = 1
    # resistance = 20
    # vRest = -65
    # vThreshold = 30
    # Q1(currentList, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, leaky = True)

    # minCurrent = 0.1
    # maxCurrent = 1
    # currentStepSize = 0.1
    # Q2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, leaky = True)

    currentList = [4, 5, 6]
    timeWindow = 1000
    a = 0.02
    b = 0.2
    c = -65
    d = 2
    vThreshold = 30
    Q4(currentList, timeWindow, a, b, c, d, vThreshold, dt = 0.01)