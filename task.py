import numpy as np
from physics_sim import PhysicsSim

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None, 
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime) 
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 1 # All four propellant with same thrust

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.]) 

    def get_reward(self, lv,av):
        """Uses current pose, v and angular_v of sim to return reward."""
        reward = 0
        """current pose of sim to calculate reward."""
        reward += (10.-.2*(abs(self.sim.pose[:3] - self.target_pos)).sum())
        """current linear velocity of sim to calculate reward."""
        reward += 5 -.3*(abs(self.sim.v- lv)).sum()
        """current angular velocity of sim to calculate reward."""
        #reward += -1*(abs(self.sim.angular_v - av)).sum()
        
        return reward

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        prev_v = []
        prev_ang_v = []
        for _ in range(self.action_repeat):
            prev_v  = self.sim.v
            prev_ang_v = self.sim.angular_v
            # update the sim pose and velocities
            done = self.sim.next_timestep(rotor_speeds) 
            reward += self.get_reward(prev_v, prev_ang_v) 
            pose_all.append(self.sim.pose[:-3])
            pose_all.append(self.sim.angular_v)#pose_all.append(self.sim.v)  # Added velocities to the states
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose[:-3],self.sim.angular_v] * self.action_repeat) # Added velocities to the states
        return state