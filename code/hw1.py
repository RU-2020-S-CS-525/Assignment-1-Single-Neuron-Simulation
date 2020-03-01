import numpy as np
import matplotlib.pyplot as plt

class LIF(object):
    #leaky integrate-and-fire model
    #C_m \frac{dV}{dt} = I(t) - frac{V_m (t)}{R_m}
    def __init__(self, capitance = 1, resistance = 20, vRest = -65, vThreshold = 5, 
                 dt = 0.01, leaky = True, tRef = 0):
        #float capitance: C_m in μF
        #float resistance: R_m in kΩ
        #float vRest: rest voltage V_r in mV
        #float vThreshold: threshold voltage V_t in mV
        #float dt: simulation step size in msec
        #bool leaky: True: leaky integrate-and-fire model; False: leaky-free integrate-and-fire model
        #float tRef: refractory time
        super(LIF, self).__init__()
        self.capitance = capitance
        self.resistance = resistance
        self.vThreshold = vThreshold
        self.vRest = vRest
        self.dt = dt
        self.leakyFactor = 1 if leaky else 0
        self.tRef = tRef
        self.refStep = int(self.tRef / self.dt)
        return

    def _update(self, tempCurrent, tempVoltage, ref_mask):
        #IN
        #np.ndarray tempCurrent, dtype = np.float64, shape = (1, n): n different input current in μA
        #np.ndarray tempVoltage, dtype = np.float64, shape = (1, n): n different membrance potential in mV
        #np.ndarray ref_mask, dtype = np.float64, shape = (1, n): n different indicator; True: not in refractory time; False: in refractory time
        #OUT
        #np.ndarray tempVoltage, dtype = np.float64, shape = (1, n): updated membrance potential in mV
        #np.ndarray spike, dtype = np.bool, shape = (1, n): True: fire; False: not fire

        #dV = (I(t) - frac{v_m (t)}{R_m}) * dt / C_m
        dV = (tempCurrent - self.leakyFactor * tempVoltage / self.resistance) * self.dt / self.capitance
        tempVoltage[ref_mask] = tempVoltage[ref_mask] + dV[ref_mask]

        #get spike and reset
        spike = tempVoltage >= self.vThreshold
        tempVoltage[spike] = self.vRest
        return tempVoltage, spike

    def simulate(self, current):
        #IN
        #np.ndarray current, dtype = np.float64, shape = (k, n): input current, |t| = k * dt, n different currents in μA
        #OUT
        #np.ndarray voltage, dtype = np.float64, shape = (k, n): membrance potential
        #np.ndarray spike, dtype = np.bool, shape = (k, n): spiking behavior
        self.voltage = np.empty_like(current, dtype = np.float64)
        self.spike = np.empty_like(current, dtype = np.bool)
        self.stepNum, self.simulationNum = current.shape
        
        #init v
        tempVoltage = np.full((1, self.simulationNum), self.vRest, dtype = np.float64)
        ref_mask = np.ones((1, self.simulationNum), dtype = np.bool)
        #loop
        for i in range(self.stepNum):
            if self.tRef > 0:
                spiked = np.any(self.spike[max(0, i - self.refStep): i], axis = 0, keepdims = True)
                ref_mask = ~spiked
            self.voltage[i], self.spike[i] = self._update(current[i], tempVoltage, ref_mask)
            tempVoltage = self.voltage[i].reshape((1, self.simulationNum))

        return self.voltage, self.spike

    def getFiringNum(self):
        #OUT
        #np.ndarray fireNum, dtype = np.float64, shape = (k, 1): the number of spikes
        return np.sum(self.spike, axis = 0, dtype = np.float64)

    def plot(self, currentList, fn_save = None):
        #IN
        #list currentList: [float current] in μA
        #str fn_save: file name; None: not save
        color = ['b', 'g', 'r', 'c', 'm', 'y']
        if self.simulationNum > len(color):
            print('E: too many currents')
            exit(-1)

        time = np.array(range(self.stepNum), dtype = np.float64) * self.dt
        for i in range(self.simulationNum):
            line, = plt.plot(time, self.voltage[:, i], c = color[i])
            point = plt.scatter(time[self.spike[:, i]], np.full(np.sum(self.spike[:, i]), self.vThreshold, dtype = np.float64), c = color[i], marker = 'o')
            line.set_label('I = ' + str(currentList[i]) + ' μA')
            point.set_label('spiking indicator')
        plt.xlabel('time (msec)')
        plt.ylabel('voltage (mV)')
        plt.legend(loc = 5)
        if self.tRef > 0:
            plt.title('membrane potential and spiking behavior with refractory time = ' + str(self.tRef) + ' msec')
        else:
            plt.title('membrane potential and spiking behavior')
        if fn_save is not None:
            plt.savefig('../docs/plots/' + fn_save)
        plt.show()
        return

class Izhikevich(object):
    #Izhikevich model
    #frac{dv}{dt} = 0.04 v^2 + 5 v + 140 - u + I
    #frac{du}{dt} = a (b v - u)
    def __init__(self, a = 0.02, b = 0.2, c = -65, d = 8, vThreshold = 30, dt = 0.01):
        #float a: time scale of u
        #float b: sensitivity of u to v
        #float c: after-spike reset v
        #float d: after-spike reset u
        #float vThreshold: threshold voltage V_t in mV
        #float dt: simulation step size in msec
        super(Izhikevich, self).__init__()
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.vThreshold = vThreshold
        self.dt = dt
        self.halfDt = self.dt / 2 #used for update v stably
        return

    def _update(self, tempCurrent, tempVoltage, tempU):
        #IN
        #np.ndarray tempCurrent, dtype = np.float64, shape = (1, n): n different input currents in μA
        #np.ndarray tempVoltage, dtype = np.float64, shape = (1, n): n different membrance potentials in mV
        #np.ndarray tempU, dtype = np.float64, shape = (1, n): n different recovary variables
        #OUT
        #np.ndarray tempVoltage, dtype = np.float64, shape = (1, n): updated membrance potential in mV
        #np.ndarray tempU, dtype = np.float64, shape = (1, n): updated recovary variables
        #np.ndarray spike, dtype = np.bool, shape = (1, n): True: fire; False: not fire

        #update V first half
        dV = (0.04 * np.square(tempVoltage) + 5 * tempVoltage + 140 - tempU + tempCurrent) * self.halfDt
        tempVoltage = tempVoltage + dV

        #update U
        dU = self.a * (self.b * tempVoltage - tempU) * self.dt
        tempU = tempU + dU

        #update V second half
        dV = (0.04 * np.square(tempVoltage) + 5 * tempVoltage + 140 - tempU + tempCurrent) * self.halfDt
        tempVoltage = tempVoltage + dV

        #get spike and reset
        spike = tempVoltage >= self.vThreshold
        tempVoltage[spike] = self.c
        tempU[spike] = tempU[spike] + self.d
        return tempVoltage, tempU, spike

    def simulate(self, current):
        #IN
        #np.ndarray current, dtype = np.float64, shape = (k, n): input current, |t| = k * dt, n different currents in μA
        #OUT
        #np.ndarray voltage, dtype = np.float64, shape = (k, n): membrance potential in mV
        #np.ndarray spike, dtype = np.bool, shape = (k, n): spiking behavior
        self.voltage = np.empty_like(current, dtype = np.float64)
        self.spike = np.empty_like(current, dtype = np.bool)
        self.stepNum, self.simulationNum = current.shape

        #init v, u
        tempVoltage = np.full((1, self.simulationNum), self.c, dtype = np.float64)
        tempU = np.full((1, self.simulationNum), self.b * self.c, dtype = np.float64)
        #loop
        for i in range(self.stepNum):
            self.voltage[i], tempU, self.spike[i] = self._update(current[i], tempVoltage, tempU)
            tempVoltage = self.voltage[i].reshape((1, self.simulationNum))
        return self.voltage, self.spike

    def getFiringNum(self):
        #OUT
        #np.ndarray fireNum, dtype = np.float64, shape = (k, 1): the number of spikes
        return np.sum(self.spike, axis = 0, dtype = np.float64)

    def plot(self, currentList, fn_save = None):
        #IN
        #list currentList: [float current] in μA
        #str fn_save: file name; None: not save
        color = ['b', 'g', 'r', 'c', 'm', 'y']
        if self.simulationNum > len(color):
            print('E: too many currents')
            exit(-1)

        time = np.array(range(self.stepNum), dtype = np.float64) * self.dt
        for i in range(self.simulationNum):
            line, = plt.plot(time, self.voltage[:, i], c = color[i])
            point = plt.scatter(time[self.spike[:, i]], np.full(np.sum(self.spike[:, i]), self.vThreshold, dtype = np.float64), c = color[i], marker = 'o')
            line.set_label('I = ' + str(currentList[i]) + ' μA')
            point.set_label('spiking indicator')
        plt.xlabel('time (msec)')
        plt.ylabel('voltage (mV)')
        plt.legend(loc = 5)
        plt.title('membrane potential and spiking behavior')
        if fn_save is not None:
            plt.savefig('../docs/plots/' + fn_save)
        plt.show()
        return
        
class HodgkinHuxley(object):
    #Hodgkin-Huxley model
    #refer to Handout
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
    def __init__(self, capitance = 1, gK = 36, gNa = 120, gL = 0.3, VK = -12, VNa = 115, VL = 10.6, dt = 0.01, TTX = False, pronase = False, VBase = -65):
        #float capitance: C_m in μF
        #float gK: maximum conductance for K in mS
        #float gNa: maximum conductance for Na in mS
        #float gL: maximum conductance for other linear ions in mS
        #float VK: equilibrium potential for K in mV
        #flaot VNa: equilibrium potential for Na in mV
        #float VL: equilibrium potential for other linear ions in mV
        #float dt: simulation step size in msec
        #bool TTX: True: use drug TTX; False: not use
        #bool pronase: True: use drug pronase; False: not use
        #float VBase: baseline voltage in mV
        super(HodgkinHuxley, self).__init__()
        self.capitance = capitance
        self.gK = gK
        self.gNa = gNa
        self.gL = gL
        self.VK = VK + VBase
        self.VNa = VNa + VBase
        self.VL = VL + VBase
        self.dt = dt
        self.TTX = TTX
        self.pronase = pronase
        self.VBase = VBase

        #define channel parameters
        aMParameter = (0.1, 25, 10, 1)
        bMParameter = (4, 0, 18)
        aHParameter = (0.07, 0, 20)
        bHParameter = (1, 30, 10, -1)
        aNParameter = (0.01, 10, 10, 1)
        bNParameter = (0.125, 0, 80)
        self.aMPA = aMParameter[0]
        self.aMPB = aMParameter[1] + self.VBase
        self.aMPC = aMParameter[2]
        self.aMPD = aMParameter[3]
        self.bMPA = bMParameter[0]
        self.bMPB = bMParameter[1] + self.VBase #I believe there is a typo in handout
        self.bMPC = bMParameter[2]
        self.aHPA = aHParameter[0]
        self.aHPB = aHParameter[1] + self.VBase
        self.aHPC = aHParameter[2]
        self.bHPA = bHParameter[0]
        self.bHPB = bHParameter[1] + self.VBase
        self.bHPC = bHParameter[2]
        self.bHPD = bHParameter[3]
        self.aNPA = aNParameter[0]
        self.aNPB = aNParameter[1] + self.VBase
        self.aNPC = aNParameter[2]
        self.aNPD = aNParameter[3]
        self.bNPA = bNParameter[0]
        self.bNPB = bNParameter[1] + self.VBase
        self.bNPC = bNParameter[2]

        #init cell
        #float vRest: rest voltage V_r in mV
        #float mInit: gating variable for Na activation gate
        #float hInit: gating variable for Na inactivation gate
        #float nInit: gating variable for K activation gate
        self.vRest = 0 + VBase
        aM = self.aMPA * (self.aMPB - self.vRest) / (np.exp((self.aMPB - self.vRest) / self.aMPC) - self.aMPD)
        bM = self.bMPA * np.exp((self.bMPB - self.vRest) / self.bMPC)
        aH = self.aHPA * np.exp((self.aHPB - self.vRest) / self.aHPC)
        bH = self.bHPA / (np.exp((self.bHPB - self.vRest) / self.bHPC) - self.bHPD)
        aN = self.aNPA * (self.aNPB - self.vRest) / (np.exp((self.aNPB - self.vRest) / self.aNPC) - self.aNPD)
        bN = self.bNPA * np.exp((self.bNPB - self.vRest) / self.bNPC)
        self.mInit = aM / (aM + bM)
        self.hInit = aH / (aH + bH)
        self.nInit = aN / (aN + bN)
        return

    def _update(self, tempCurrent, tempVoltage, tempM, tempH, tempN):
        #IN
        #np.ndarray tempCurrent, dtype = np.float64, shape = (1, n): n different input currents in μA
        #np.ndarray tempVoltage, dtype = np.float64, shape = (1, n): n different membrance potentials in mV
        #np.ndarray tempM, dtype = np.float64, shape = (1, n): n different m values
        #np.ndarray tempH, dtype = np.float64, shape = (1, n): n different h values
        #np.ndarray tempN, dtype = np.float64, shape = (1, n): n different n values
        #OUT
        #np.ndarray tempVoltage, dtype = np.float64, shape = (1, n): updated membrance potential in mV
        #np.ndarray tempM, dtype = np.float64, shape = (1, n): updated m values
        #np.ndarray tempH, dtype = np.float64, shape = (1, n): updated h values
        #np.ndarray tempN, dtype = np.float64, shape = (1, n): updated n values
        #np.ndarray spike, dtype = np.bool, shape = (1, n): True: fire; False: not fire


        #compute a, b
        aM = self.aMPA * (self.aMPB - tempVoltage) / (np.exp((self.aMPB - tempVoltage) / self.aMPC) - self.aMPD)
        bM = self.bMPA * np.exp((self.bMPB - tempVoltage) / self.bMPC)
        if self.pronase:
            tempH = 1
        else:
            aH = self.aHPA * np.exp((self.aHPB - tempVoltage) / self.aHPC)
            bH = self.bHPA / (np.exp((self.bHPB - tempVoltage) / self.bHPC) - self.bHPD)
        aN = self.aNPA * (self.aNPB - tempVoltage) / (np.exp((self.aNPB - tempVoltage) / self.aNPC) - self.aNPD)
        bN = self.bNPA * np.exp((self.bNPB - tempVoltage) / self.bNPC)

        #update V, m, h, n
        IK = self.gK * np.power(tempN, 4) * (tempVoltage - self.VK)
        if self.TTX:
            INa = 0
        else:
            INa = self.gNa * np.power(tempM, 3) * tempH * (tempVoltage - self.VNa)
        IL = self.gL * (tempVoltage - self.VL)
        dV = (tempCurrent - IK - INa - IL) / self.capitance * self.dt
        dM = (aM * (1 - tempM) - bM * tempM) * self.dt
        if self.pronase:
            pass
        else:
            dH = (aH * (1 - tempH) - bH * tempH) * self.dt
        dN = (aN * (1 - tempN) - bN * tempN) * self.dt
        tempVoltage = tempVoltage + dV
        tempM = tempM + dM
        if self.pronase:
            tempH = 1
        else:
            tempH = tempH + dH
        tempN = tempN + dN

        #avoid overflow
        mask0 = tempM < 0
        mask1 = tempM > 1
        if mask0.any() or mask1.any():
            print('W: overflow for m')
            tempM[mask0] = 0
            tempM[mask1] = 1
        if self.pronase:
            pass
        else:
            mask0 = tempH < 0
            mask1 = tempH > 1
            if mask0.any() or mask1.any():
                print('W: overflow for h')
                tempH[mask0] = 0
                tempH[mask1] = 1
        mask0 = tempN < 0
        mask1 = tempN > 1
        if mask0.any() or mask1.any():
            print('W: overflow for n')
            tempN[mask0] = 0
            tempN[mask1] = 1
        return tempVoltage, tempM, tempH, tempN

    def simulate(self, current):
        #IN
        #np.ndarray current, dtype = np.float64, shape = (k, n): input current, |t| = k * dt, n different currents in μA
        #OUT
        #np.ndarray voltage, dtype = np.float64, shape = (k, n): membrance potential in mV
        self.voltage = np.empty_like(current, dtype = np.float64)
        self.parameterM = np.empty_like(current, dtype = np.float64)
        self.parameterH = np.empty_like(current, dtype = np.float64)
        self.parameterN = np.empty_like(current, dtype = np.float64)
        self.stepNum, self.simulationNum = current.shape

        #init v, u
        tempVoltage = np.full((1, self.simulationNum), self.vRest, dtype = np.float64)
        tempM = np.full((1, self.simulationNum), self.mInit, dtype = np.float64)
        tempH = np.full((1, self.simulationNum), self.hInit, dtype = np.float64)
        tempN = np.full((1, self.simulationNum), self.nInit, dtype = np.float64)
        #loop
        for i in range(self.stepNum):
            self.voltage[i], self.parameterM[i], self.parameterH[i], self.parameterN[i] = self._update(current[i], tempVoltage, tempM, tempH, tempN)
            tempVoltage = self.voltage[i].reshape((1, self.simulationNum))
            tempM = self.parameterM[i].reshape((1, self.simulationNum))
            tempH = self.parameterH[i].reshape((1, self.simulationNum))
            tempN = self.parameterN[i].reshape((1, self.simulationNum))
        return self.voltage

    def plot(self, currentList, halfInputFlag = False, plotMHNFlag = False, fn_save = None):
        #IN
        #list currentList: [float current] in μA
        #bool halfInputFlag: change the title; True: plot for EX4, False: general plot
        #bool plotMHNFlag: True: plot parameter m h n; False: not plot
        #str fn_save: file name; None: not save
        color = ['b', 'g', 'r', 'c', 'm', 'y']
        if self.simulationNum > len(color):
            print('E: too many currents')
            exit(-1)

        time = np.array(range(self.stepNum), dtype = np.float64) * self.dt
        for i in range(self.simulationNum):
            line, = plt.plot(time, self.voltage[:, i], c = color[i])
            line.set_label('I = ' + str(currentList[i]) + ' μA')
        plt.xlabel('time (msec)')
        plt.ylabel('voltage (mV)')
        plt.legend(loc = 5)
        if halfInputFlag:
            plt.title('membrane potential when input currents last for first ' + str(self.stepNum / 2 * self.dt) + ' msecs')
        else:
            plt.title('membrane potential')
        if fn_save is not None:
            plt.savefig('../docs/plots/' + fn_save)
        plt.show()

        if plotMHNFlag:
            for i in range(self.simulationNum):
                line, = plt.plot(time, self.parameterM[:, i], c = color[i])
                line.set_label('I = ' + str(currentList[i]) + ' μA')
            plt.xlabel('time (msec)')
            plt.ylabel('voltage (mV)')
            plt.legend(loc = 5)
            if halfInputFlag:
                plt.title('parameter m when input currents last for first ' + str(self.stepNum / 2 * self.dt) + ' msecs')
            else:
                plt.title('parameter m')
            plt.show()

            for i in range(self.simulationNum):
                line, = plt.plot(time, self.parameterH[:, i], c = color[i])
                line.set_label('I = ' + str(currentList[i]) + ' μA')
            plt.xlabel('time (msec)')
            plt.ylabel('voltage (mV)')
            plt.legend(loc = 5)
            if halfInputFlag:
                plt.title('parameter h when input currents last for first ' + str(self.stepNum / 2 * self.dt) + ' msecs')
            else:
                plt.title('parameter h')
            plt.show()

            for i in range(self.simulationNum):
                line, = plt.plot(time, self.parameterN[:, i], c = color[i])
                line.set_label('I = ' + str(currentList[i]) + ' μA')
            plt.xlabel('time (msec)')
            plt.ylabel('voltage (mV)')
            plt.legend(loc = 5)
            if halfInputFlag:
                plt.title('parameter n when input currents last for first ' + str(self.stepNum / 2 * self.dt) + ' msecs')
            else:
                plt.title('parameter n')
            plt.show()
        return    


def Q1(currentList, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, leaky = True, tRef = 0, fn_save = None):
    #IN
    #list currentList: [float current] in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float resistance: R_m in kΩ
    #float vRest: rest voltage V_r in mV
    #float vThreshold: threshold voltage V_t in mV
    #float dt: simulation step size in msec
    #bool leaky: True: leaky integrate-and-fire model; False: leaky-free integrate-and-fire model
    #float tRef: refractory time in msec
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = len(currentList)

    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init LIF model
    lif = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = leaky, tRef = tRef)

    #simulate and plot
    lif.simulate(current)
    lif.plot(currentList, fn_save = fn_save)
    return

def Q2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance,
       resistance, vRest, vThreshold, dt = 0.01, leaky = True, tRef = 0, fn_save = None):
    #IN
    #float minCurrent: minimum input current in μA
    #float maxCurrent: maximum input current in μA
    #float currentStepSize: increasing step for input current in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float resistance: R_m in kΩ
    #float vRest: rest voltage V_r in mV
    #float vThreshold: threshold voltage V_t in mV
    #float dt: simulation step size in msec
    #bool leaky: True: leaky integrate-and-fire model; False: leaky-free integrate-and-fire model
    #float tRef: refractory time in msec
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = int(np.ceil((maxCurrent - minCurrent) / currentStepSize))
    currentList = np.array(range(simulationNum), dtype = np.float64) * currentStepSize + minCurrent
    
    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init LIF model
    lif = LIF(capitance, resistance, vRest, vThreshold, dt, leaky, tRef)
    #simulate
    lif.simulate(current)
    rate = lif.getFiringNum() / timeWindow * 1000

    #plot
    plt.plot(currentList, rate)
    plt.xlabel('current (μA)')
    plt.ylabel('firing rate (Hz)')
    plt.title('firing rate vs. input current')
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save)
    plt.show()
    return

def Q4(currentList, timeWindow, a = 0.02, b = 0.2, c = -65, d = 8, vThreshold = 30, dt = 0.01, fn_save = None):
    #IN
    #list currentList: [float current] in μA
    #float timeWindow: simulation time in msec
    #float a: time scale of u
    #float b: sensitivity of u to v
    #float c: after-spike reset v
    #float d: after-spike reset u
    #float vThreshold: threshold voltage V_t in mV
    #float dt: simulation step size in msec
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = len(currentList)

    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init Izhikevich model
    izhikevich = Izhikevich(a, b, c, d, vThreshold, dt)

    #simulate and plot
    izhikevich.simulate(current)
    izhikevich.plot(currentList, fn_save = fn_save)
    return

def Q5(currentList, timeWindow, capitance = 1, gK = 36, gNa = 120, gL = 0.3, VK = -12, VNa = 115, VL = 10.6, 
       dt = 0.01, TTX = False, pronase = False, VBase = -65, plotMHNFlag = False, fn_save = None):
    #IN
    #list currentList: [float current] in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float gK: maximum conductance for K in mS
    #float gNa: maximum conductance for Na in mS
    #float gL: maximum conductance for other linear ions in mS
    #float VK: equilibrium potential for K in mV
    #flaot VNa: equilibrium potential for Na in mV
    #float VL: equilibrium potential for other linear ions in mV
    #float dt: simulation step size in msec
    #bool TTX: True: use drug TTX; False: not use
    #bool pronase: True: use drug pronase; False: not use
    #float VBase: baseline voltage in mV
    #bool plotMHNFlag: True: plot parameter m h n; False: not plot
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = len(currentList)

    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init HH model
    HH = HodgkinHuxley(capitance, gK, gNa, gL, VK, VNa, VL, dt, TTX, pronase, VBase = VBase)

    #simulate and plot
    HH.simulate(current)
    HH.plot(currentList, plotMHNFlag, fn_save = fn_save)
    return

def Q6(initCurrent, timeWindow, capitance = 1, gK = 36, gNa = 120, gL = 0.3, VK = -12, VNa = 115, VL = 10.6, dt = 0.01, VBase = -65, plotMHNFlag = False, fn_save = None):
    #IN
    #float initCurrent: input current in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float gK: maximum conductance for K in mS
    #float gNa: maximum conductance for Na in mS
    #float gL: maximum conductance for other linear ions in mS
    #float VK: equilibrium potential for K in mV
    #flaot VNa: equilibrium potential for Na in mV
    #float VL: equilibrium potential for other linear ions in mV
    #float dt: simulation step size in msec
    #float VBase: baseline voltage in mV
    #bool plotMHNFlag: True: plot parameter m h n; False: not plot
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    current = np.full((stepNum, 1), initCurrent, dtype = np.float64)

    #init HH model
    HH0 = HodgkinHuxley(capitance, gK, gNa, gL, VK, VNa, VL, dt, TTX = False, pronase = False, VBase = VBase)
    HH1 = HodgkinHuxley(capitance, gK, gNa, gL, VK, VNa, VL, dt, TTX = True, pronase = False, VBase = VBase)
    HH2 = HodgkinHuxley(capitance, gK, gNa, gL, VK, VNa, VL, dt, TTX = False, pronase = True, VBase = VBase)

    #simulate
    v0 = HH0.simulate(current)
    v1 = HH1.simulate(current)
    v2 = HH2.simulate(current)

    time = np.array(range(stepNum), dtype = np.float64) * dt

    line, = plt.plot(time, v0[:, 0], c = 'b')
    line.set_label('no drug')
    line, = plt.plot(time, v1[:, 0], c = 'g')
    line.set_label('TTX')
    line, = plt.plot(time, v2[:, 0], c = 'r')
    line.set_label('pronase')
    plt.xlabel('time (msec)')
    plt.ylabel('voltage (mV)')
    plt.legend(loc = 5)
    plt.title('membrane potential when I = ' + str(initCurrent) + ' μA')
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save)
    plt.show()

    if plotMHNFlag:
        line, = plt.plot(time, HH0.parameterM[:, 0], c = 'b')
        line.set_label('no drug')
        line, = plt.plot(time, HH1.parameterM[:, 0], c = 'g')
        line.set_label('TTX')
        line, = plt.plot(time, HH2.parameterM[:, 0], c = 'r')
        line.set_label('pronase')
        plt.xlabel('time (msec)')
        plt.ylabel('voltage (mV)')
        plt.legend(loc = 5)
        plt.title('parameter m when I = ' + str(initCurrent) + ' μA')
        plt.show()

        line, = plt.plot(time, HH0.parameterH[:, 0], c = 'b')
        line.set_label('no drug')
        line, = plt.plot(time, HH1.parameterH[:, 0], c = 'g')
        line.set_label('TTX')
        line, = plt.plot(time, HH2.parameterH[:, 0], c = 'r')
        line.set_label('pronase')
        plt.xlabel('time (msec)')
        plt.ylabel('voltage (mV)')
        plt.legend(loc = 5)
        plt.title('parameter h when I = ' + str(initCurrent) + ' μA')
        plt.show()

        line, = plt.plot(time, HH0.parameterN[:, 0], c = 'b')
        line.set_label('no drug')
        line, = plt.plot(time, HH1.parameterN[:, 0], c = 'g')
        line.set_label('TTX')
        line, = plt.plot(time, HH2.parameterN[:, 0], c = 'r')
        line.set_label('pronase')
        plt.xlabel('time (msec)')
        plt.ylabel('voltage (mV)')
        plt.legend(loc = 5)
        plt.title('parameter n when I = ' + str(initCurrent) + ' μA')
        plt.show()
    return


def EX1(initCurrent, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, tRef = 0, fn_save = None):
    #for comparing membrane potential of IF and LIF
    #IN
    #float initCurrent: input current in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float resistance: R_m in kΩ
    #float vRest: rest voltage V_r in mV
    #float vThreshold: threshold voltage V_t in mV
    #float dt: simulation step size in msec
    #float tRef: refractory time in msec
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    current = np.full((stepNum, 1), initCurrent, dtype = np.float64)

    #init LIF and IF model
    lif1 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = True, tRef = tRef)
    lif2 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = False, tRef = tRef)

    #simulate and plot
    v1, s1 = lif1.simulate(current)
    v2, s2 = lif2.simulate(current)

    #plot
    time = np.array(range(stepNum), dtype = np.float64) * dt

    line, = plt.plot(time, v1, c = 'b')
    point = plt.scatter(time[s1[:, 0]], np.full(np.sum(s1[:, 0]), vThreshold, dtype = np.float64), c = 'b', marker = 'o')
    line.set_label('LIF')
    point.set_label('LIF')
    line, = plt.plot(time, v2, c = 'g')
    point = plt.scatter(time[s2[:, 0]], np.full(np.sum(s2[:, 0]), vThreshold, dtype = np.float64), c = 'g', marker = 'o')
    line.set_label('IF')
    point.set_label('IF')
    plt.xlabel('time (msec)')
    plt.ylabel('voltage (mV)')
    plt.title('membrane potential and spiking behavior when I = ' + str(initCurrent) + ' μA')
    plt.legend(loc = 5)
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save)
    plt.show()
    return

def EX2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, tRef = 0, fn_save = None):
    #for comparing firing rate of IF and LIF
    #IN
    #float minCurrent: minimum input current in μA
    #float maxCurrent: maximum input current in μA
    #float currentStepSize: increasing step for input current in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float resistance: R_m in kΩ
    #float vRest: rest voltage V_r in mV
    #float vThreshold: threshold voltage V_t in mV
    #float dt: simulation step size in msec
    #float tRef: refractory time in msec
    #str fn_save: file name; None: not save


    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = int(np.ceil((maxCurrent - minCurrent) / currentStepSize))
    currentList = np.array(range(simulationNum), dtype = np.float64) * currentStepSize + minCurrent
    
    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init LIF and IF model
    lif1 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = True, tRef = tRef)
    lif2 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = False, tRef = tRef)

    #simulate
    lif1.simulate(current)
    rate1 = lif1.getFiringNum() / timeWindow * 1000
    lif2.simulate(current)
    rate2 = lif2.getFiringNum() / timeWindow * 1000

    #plot
    line, = plt.plot(currentList, rate1)
    line.set_label('LIF')
    line, = plt.plot(currentList, rate2)
    line.set_label('IF')
    plt.xlabel('current (μA)')
    plt.ylabel('firing rate (Hz)')
    plt.legend(loc = 5)
    plt.title('firing rate vs. input current')
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save)
    plt.show()
    return

def EX3(initCurrent, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, tRef = 0, fn_save = None):
    #for finding dominant term
    #IN
    #float initCurrent: input current in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float resistance: R_m in mΩ
    #float vRest: rest voltage V_r in mV
    #float vThreshold: threshold voltage V_t in mV
    #float dt: simulation step size in msec
    #float tRef: refractory time in msec
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    current = np.full((stepNum, 1), initCurrent, dtype = np.float64)

    #init LIF
    lif = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = True, tRef = tRef)

    #simulate and plot
    v1, s1 = lif.simulate(current)

    #plot
    time = np.array(range(stepNum), dtype = np.float64) * dt

    line, = plt.plot(time, v1, c = 'b')
    line.set_label('V_m')
    line, = plt.plot(time, resistance * current, c = 'g')
    line.set_label('R_m * I')
    plt.xlabel('time (msec)')
    plt.ylabel('voltage (mV)')
    plt.title('terms in LIF model when I = ' + str(initCurrent) + ' μA')
    plt.legend(loc = 5)
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save)
    plt.show()
    return

def EX4(currentList, timeWindow, capitance = 1, gK = 36, gNa = 120, gL = 0.3, VK = -12, VNa = 115, VL = 10.6, dt = 0.01,
        TTX = False, pronase = False, VBase = -65, plotMHNFlag = False, fn_save = None):
    #for illustrating rebound spiking
    #IN
    #list currentList: [float current] in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float gK: maximum conductance for K in mS
    #float gNa: maximum conductance for Na in mS
    #float gL: maximum conductance for other linear ions in mS
    #float VK: equilibrium potential for K in mV
    #flaot VNa: equilibrium potential for Na in mV
    #float VL: equilibrium potential for other linear ions in mV
    #float dt: simulation step size in msec
    #bool TTX: True: use drug TTX; False: not use
    #bool pronase: True: use drug pronase; False: not use
    #float VBase: baseline voltage
    #bool plotMHNFlag: True: plot parameter m h n; False: not plot
    #str fn_save: file name; None: not save

    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = len(currentList)

    #init input current
    current = np.zeros((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[: (stepNum // 2), i] = currentList[i]

    #init HH model
    HH = HodgkinHuxley(capitance, gK, gNa, gL, VK, VNa, VL, dt, TTX, pronase, VBase = VBase)

    #simulate and plot
    HH.simulate(current)
    HH.plot(currentList, halfInputFlag = True, plotMHNFlag = plotMHNFlag, fn_save = fn_save)
    return

def EX5(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, tRef = 0, fn_save = None):
    #for comparing firing rate of LIF with and without refractory time
    #IN
    #float minCurrent: minimum input current in μA
    #float maxCurrent: maximum input current in μA
    #float currentStepSize: increasing step for input current in μA
    #float timeWindow: simulation time in msec
    #float capitance: C_m in μF
    #float resistance: R_m in kΩ
    #float vRest: rest voltage V_r in mV
    #float vThreshold: threshold voltage V_t in mV
    #float dt: simulation step size in msec
    #float tRef: refractory time in msec
    #str fn_save: file name; None: not save


    #preprocessing
    stepNum = int(np.ceil(timeWindow / dt))
    simulationNum = int(np.ceil((maxCurrent - minCurrent) / currentStepSize))
    currentList = np.array(range(simulationNum), dtype = np.float64) * currentStepSize + minCurrent
    
    #init input current
    current = np.empty((stepNum, simulationNum), dtype = np.float64)
    for i in range(simulationNum):
        current[:, i] = currentList[i]

    #init LIF and IF model
    lif1 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = True, tRef = 0)
    lif2 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = False, tRef = 0)
    lif3 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = True, tRef = tRef)
    lif4 = LIF(capitance, resistance, vRest, vThreshold, dt = dt, leaky = False, tRef = tRef)

    #simulate
    lif1.simulate(current)
    rate1 = lif1.getFiringNum() / timeWindow * 1000
    lif2.simulate(current)
    rate2 = lif2.getFiringNum() / timeWindow * 1000
    lif3.simulate(current)
    rate3 = lif3.getFiringNum() / timeWindow * 1000
    lif4.simulate(current)
    rate4 = lif4.getFiringNum() / timeWindow * 1000

    #plot
    line, = plt.plot(currentList, rate1)
    line.set_label('LIF without refractory time')
    line, = plt.plot(currentList, rate2)
    line.set_label('IF without refractory time')
    line, = plt.plot(currentList, rate3)
    line.set_label('LIF with refractory time')
    line, = plt.plot(currentList, rate4)
    line.set_label('IF with refractory time')
    plt.xlabel('current (μA)')
    plt.ylabel('firing rate (Hz)')
    plt.legend(loc = 4)
    plt.title('firing rate vs. input current')
    if fn_save is not None:
        plt.savefig('../docs/plots/' + fn_save)
    plt.show()
    return

if __name__ == '__main__':
    currentList = [0.3, 0.4, 0.5]
    timeWindow = 250
    capitance = 1
    resistance = 20
    vRest = -65
    vThreshold = 5
    fn_save = 'plot_programming_1.png'
    Q1(currentList, timeWindow, capitance, resistance, vRest, vThreshold,
        dt = 0.01, leaky = True, fn_save = fn_save)

    minCurrent = 0.01
    maxCurrent = 3
    currentStepSize = 0.01
    timeWindow = 10000
    capitance = 1
    resistance = 20
    vRest = -65
    vThreshold = 5
    fn_save = 'plot_programming_2.png'
    Q2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance,
       resistance, vRest, vThreshold, dt = 0.01, leaky = True, tRef = 0,
       fn_save = fn_save)

    minCurrent = 0.01
    maxCurrent = 100
    currentStepSize = 1
    timeWindow = 1000
    tRef = 5
    capitance = 1
    resistance = 20
    vRest = -65
    vThreshold = 5
    fn_save = 'plot_programming_2_2.png'
    Q2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance,
       resistance, vRest, vThreshold, dt = 0.01, leaky = True, tRef = tRef,
       fn_save = fn_save)

    currentList = [4, 5, 6]
    timeWindow = 500
    a = 0.02
    b = 0.2
    c = -65
    d = 8
    vThreshold = 30
    fn_save = None
    Q4(currentList, timeWindow, a, b, c, d, vThreshold, dt = 0.01, fn_save = fn_save)

    currentList = [-10, 2, 5, 9]
    timeWindow = 50
    capitance = 1
    gK = 36
    gNa = 120
    gL = 0.3
    VK = -12
    VNa = 115
    VL = 10.6
    fn_save = None
    Q5(currentList, timeWindow, capitance, gK, gNa, gL, VK, VNa, VL, dt = 0.01, TTX = False, pronase = False, fn_save = fn_save)

    current = 5
    timeWindow = 25
    capitance = 1
    gK = 36
    gNa = 120
    gL = 0.3
    VK = -12
    VNa = 115
    VL = 10.6
    fn_save = None
    Q6(current, timeWindow, capitance, gK, gNa, gL, VK, VNa, VL, dt = 0.01, plotMHNFlag = True, fn_save = fn_save)

    current = 0.27
    timeWindow = 1000
    capitance = 1
    resistance = 20
    vRest = -65
    vThreshold = 5
    fn_save = None
    EX1(current, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, fn_save = fn_save)
    
    minCurrent = 0.01
    maxCurrent = 3
    currentStepSize = 0.01
    timeWindow = 10000
    capitance = 1
    resistance = 20
    vRest = -65
    vThreshold = 5
    fn_save = None
    EX2(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, fn_save = fn_save)
    
    current = 0.26
    timeWindow = 1000
    capitance = 1
    resistance = 20
    vRest = -65
    vThreshold = 5
    fn_save = None
    EX3(current, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, fn_save = fn_save)

    currentList = [-9, 5, 9]
    timeWindow = 50
    capitance = 1
    gK = 36
    gNa = 120
    gL = 0.3
    VK = -12
    VNa = 115
    VL = 10.6
    fn_save = None
    EX4(currentList, timeWindow, capitance, gK, gNa, gL, VK, VNa, VL, dt = 0.01, TTX = False, pronase = False, plotMHNFlag = True, fn_save = fn_save)

    minCurrent = 0.05
    maxCurrent = 5
    currentStepSize = 0.05
    timeWindow = 3000
    capitance = 1
    resistance = 20
    vRest = -65
    vThreshold = 5
    tRef = 5
    fn_save = None
    EX5(minCurrent, maxCurrent, currentStepSize, timeWindow, capitance, resistance, vRest, vThreshold, dt = 0.01, tRef = tRef, fn_save = fn_save)