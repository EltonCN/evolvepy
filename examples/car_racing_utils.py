import numpy as np
import cv2 as cv
import gym
from gym.wrappers import Monitor
from matplotlib import pyplot as plt

from evolvepy.evaluator import ProcessFitnessFunction

class PID:

    def __init__(self, kp:float=1.0, ki:float=0.0, kd:float=0.0):
        self._kp = kp
        self._ki = ki
        self._kd = kd

        self._cumm = 0.0
        self._error_prev = 0.0

    def compute(self, error, dt):
        u = self._kp*error

        self._cumm += (self._error_prev+error)*dt/2.0
        u += self._ki * self._cumm

        u += self._kd * (error-self._error_prev)

        self._error_prev = error

        return u

def compute_long_error(env, pos_prev=None, setpoint_velocity=1.0):
    # Get's the car position
    pos = np.array(env.car.hull.position)
    
    if pos_prev is None:
        pos_prev = pos
    
    # Calculates the difference between the two positions to get the speed.
    vel = np.linalg.norm(pos-pos_prev)

    error = setpoint_velocity - vel

    return error, pos

def compute_lat_error(obs):
    # Position of the car on the image
    centroid = np.array([48, 72], int)

    # Get's only the track (and the car)
    hsv = cv.cvtColor(obs, cv.COLOR_RGB2HSV)
    min_gray = np.array([0,0,0],np.uint8)
    max_gray = np.array([10,255,255],np.uint8)
    mask_gray = cv.inRange(hsv, min_gray, max_gray)

    # Take a spot just above the car
    centroid2 = centroid.copy()
    centroid2[1] -= 5

    # Take a line of the image from that point
    line = mask_gray[int(centroid2[1]), :]

    # Check the position of the edges around the car
    border_rigth = np.argmax(line[int(centroid2[0]):] < 1) + int(centroid2[0])
    border_left = int(centroid2[0]) - np.argmax(np.flip(line[:int(centroid2[0])]) < 1)

    # Computes the error
    error = abs(border_rigth - centroid[0])/abs(border_rigth-border_left)
    error -= 0.5
    
    return error

class CarRacingEvaluator(ProcessFitnessFunction):
    def __init__(self, show=False, save=False) -> None:
        super().__init__(reset=save)

        self._env = None
        self._show = show
        self._save = save
        self._count = 0
    
    def setup(self) -> None:
        '''
            Initializes the environment.
        '''
        self._env = gym.make("CarRacing-v0", verbose=0)

        if self._save:
            self._env = Monitor(self._env, "./video"+str(self._count), force=True)
            self._count += 1

        

    def evaluate(self, individuals:np.ndarray) -> np.ndarray:
        
        # Transfer weights to PID controllers
        individual = individuals[0]["chr0"]
        kp = individual[0]
        ki = individual[1]
        kd = individual[2]
        pid_long = PID(kp, ki, kd)

        kp = individual[3]
        ki = individual[4]
        kd = individual[5]
        pid_lat = PID(kp, ki, kd)

        env = self._env

        obs = env.reset()
        total_reward = 0.0
        
        # Discards the first iterations the camera is being adjusted
        for _ in range(50):
            obs, _, _, _ = env.step([0,0,0])
        
        # Drive the car through the environment, accumulating the reward
        pos_prev = None
        done = False
        while not done:
            error, pos = compute_long_error(env, pos_prev)
            accel = pid_long.compute(error, 1.0/50.0)
            accel = np.clip(accel, -1.0, 1.0)
            if np.isnan(accel):
                accel = 0.0

            error = compute_lat_error(obs)
            steering = pid_lat.compute(error, 1.0/50.0)
            steering = np.clip(steering, -1.0, 1.0)
            if np.isnan(steering):
                steering = 0.0

            if accel > 0:
                action = [steering, accel, 0.0]
            else:
                action = [steering, 0.0, accel]
            
            obs, rew, done, info = env.step(action)
            
            total_reward += rew

            if self._show:
                if isnotebook():
                    show_state(env)
                else:
                    env.render()


            pos_prev = pos

        if self._save:
            self._env.close()

        return total_reward

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter    

def show_state(env):
    from IPython import display
    plt.figure(3)
    plt.clf()
    plt.imshow(env.render(mode='rgb_array'))
    plt.axis('off')

    display.clear_output(wait=True)
    display.display(plt.gcf())