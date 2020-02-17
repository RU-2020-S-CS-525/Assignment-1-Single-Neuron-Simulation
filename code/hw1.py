import numpy as np
import matplotlib.pyplot as plt

class LIF(object):
    #leaky integrate-and-fire model
    #C_m \frac{dV}{dt} = I(t) - frac{V_m (t)}{R_m}
    def __init__(self, capitance = 1, resistance = 20, vRest = -65, vThreshold = 5, dt = 0.01, leaky = True):
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
        return

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
        
class HodgkinHuxley(object):
    #Hodgkin-Huxley model
    #refer to Hodgkin, A.L. and Huxley, A.F., 1952. A quantitative description of membrane current and its application to conduction and excitation in nerve. The Journal of physiology, 117(4), pp.500-544.
    #C \times \frac{dv}{dt} = I - g_{Na} m^3 h (V - V_{Na}) - g_{K} n^4 (V - V_{K}) - g_{L} (V - V_{L})
    #\frac{dm}{dt} = a_m(V) (1 - m) - b_m(V) m
    #\frac{dh}{dt} = a_h(V) (1 - n) - b_h(V) h
    #\frac{dn}{dt} = a_n(V) (1 - n) - b_n(V) n
    #a_m(V) = \frac{0.1 (25 - V)}{\exp(\frac{25 - V}{10}) - 1}
    #b_m(V) = 4 \exp(\frac{-V}{18})
    #a_h(V) = 0.07 \exp(\frac{-V}{20})
    #b_h(V) = \frac{1}{\exp(\frac{30 - V}{10} + 1)}
    #a_n(V) = \frac{0.01 (10 - V)}{\exp(10 - V) - 1}
    #b_n(V) = 0.125 \exp(\frac{-V}{80})
    def __init__(self, capitance = 1, gK = 36, gNa = 120, gL = 0.3, VK = -12, VNa = 115, VL = 10.6, dt = 0.01, TTX = False, pronase = False):
        super(HodgkinHuxley, self).__init__()
        self.capitance = capitance
        self.gK = gK
        self.gNa = gNa
        self.gL = gL
        self.VK = VK
        self.VNa = VNa
        self.VL = VL
        self.dt = dt
        self.TTX = TTX
        self.pronase = pronase

        self.vRest = 0
        self.mInit = 0.05
        self.hInit = 0.60
        self.nInit = 0.32
        return

    def _update(self, tempCurrent, tempVotage, tempM, tempH, tempN):
        #IN
        #np.ndarray tempCurrent, dtype = np.float64, shape = (1, n): n different input currents
        #np.ndarray tempVotage, dtype = np.float64, shape = (1, n): n different membrance potentials
        #np.ndarray tempM, dtype = np.float64, shape = (1, n): n different m values
        #np.ndarray tempH, dtype = np.float64, shape = (1, n): n different h values
        #np.ndarray tempN, dtype = np.float64, shape = (1, n): n different n values
        #OUT
        #np.ndarray tempVotage, dtype = np.float64, shpape = (1, n): updated membrance potential
        #np.ndarray tempM, dtype = np.float64, shape = (1, n): updated m values
        #np.ndarray tempH, dtype = np.float64, shape = (1, n): updated h values
        #np.ndarray tempN, dtype = np.float64, shape = (1, n): updated n values
        #np.ndarray spike, dtype = np.bool, shpape = (1, n): True: fire; False: not fire

        aM = 0.1 * (25 - tempVotage) / (np.exp((25 - tempVotage) / 10) - 1)
        bM = 4 * np.exp(-1 * tempVotage / 18)
        aH = 0.07 * np.exp(-1 * tempVotage / 20)
        bH = 1 / (np.exp((30 - tempVotage) / 10) + 1)
        aN = 0.01 * (10 - tempVotage) / (np.exp(10 - tempVotage) / 10 - 1)
        bN = 0.125 * np.exp(-1 * tempVotage / 80)

        dV = (tempCurrent - self.gK * np.power(tempN, 4) * (tempVotage - self.VK) - self.gNa * np.power(tempM, 3) * tempH * (tempVotage - self.VNa) - self.gL * (tempVotage - self.VL)) / self.capitance * self.dt
        dM = (aM * (1 - tempM) - bM * tempM) * self.dt
        dH = (aH * (1 - tempH) - bH * tempH) * self.dt
        dN = (aN * (1 - tempN) - bN * tempN) * self.dt

        tempVotage = tempVotage + dV
        tempM = tempM + dM
        tempH = tempH + dH
        tempN = tempN + dN
        return tempVotage, tempM, tempH, tempN

    def simulate(self, current):
        #IN
        #np.ndarray current, dtype = np.float64, shape = (k, n): input current, |t| = k * dt, n different currents
        #OUT
        #np.ndarray votage, dtype = np.float64, shape = (k, n): membrance potential
        self.votage = np.empty_like(current, dtype = np.float64)
        self.stepNum, self.simulationNum = current.shape

        #init v, u
        tempVotage = np.full((1, self.simulationNum), self.vRest, dtype = np.float64)
        tempM = np.full((1, self.simulationNum), self.mInit, dtype = np.float64)
        tempH = np.full((1, self.simulationNum), self.hInit, dtype = np.float64)
        tempN = np.full((1, self.simulationNum), self.nInit, dtype = np.float64)
        #loop
        for i in range(self.stepNum):
            self.votage[i], tempM, tempH, tempN = self._update(current[i], tempVotage, tempM, tempH, tempN)
            tempVotage = self.votage[i]
        return self.votage

    def plot(self, currentList):
        #list currentList: [float current]
        color = ['b', 'g', 'r', 'c', 'm', 'y']
        if self.simulationNum > len(color):
            print('E: too many currents')
            exit(-1)

        time = np.array(range(self.stepNum), dtype = np.float64) * self.dt
        for i in range(self.simulationNum):
            line, = plt.plot(time, self.votage[:, i], c = color[i])
            line.set_label('I = ' + str(currentList[i]) + 'mA')
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

def Q5(currentList, timeWindow, capitance = 1, gK = 36, gNa = 120, gL = 0.3, VK = -12, VNa = 115, VL = 10.6, dt = 0.01, TTX = False, pronase = False):
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
    HH = HodgkinHuxley(capitance, gK, gNa, gL, VK, VNa, VL, dt, TTX, pronase)

    #simulate and plot
    HH.simulate(current)
    HH.plot(currentList)
    return

if __name__ == '__main__':
    # currentList = [0.3, 0.4, 0.5]
    # timeWindow = 1000
    # capitance = 1
    # resistance = 20
    # vRest = -65
    # vThreshold = 5
    # Q1(currentList, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, leaky = True)

    # minCurrent = 0.1
    # maxCurrent = 1
    # currentStepSize = 0.1
    # Q2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, leaky = True)

    # currentList = [4, 5, 6]
    # timeWindow = 1000
    # a = 0.02
    # b = 0.2
    # c = -65
    # d = 2
    # vThreshold = 30
    # Q4(currentList, timeWindow, a, b, c, d, vThreshold, dt = 0.01)

    currentList = [30, 50, 80]
    timeWindow = 25
    capitance = 1
    gK = 36
    gNa = 120
    gL = 0.3
    VK = -12
    VNa = 115
    VL = 10.6
    Q5(currentList, timeWindow, capitance, gK, gNa, gL, VK, VNa, VL, dt = 0.01, TTX = False, pronase = False)