import numpy as np
from .State import State
from .Action import Action

'''
Do you want to continue with the before defined measurements or do I want to modify them

#Five level of an observation in a system ( can be what?)

Includes blood glucose level proxy for diabetes: 0-3
    (lo2, lo1, normal, hi1, hi2); Any other than normal is "abnormal"

#Intilial distribtuion of cpu usage

Initial distribution:
    [.05, .15, .6, .15, .05] for non-diabetics and [.01, .05, .15, .6, .19] for diabetics

# Effect of the more process inside

Effect of vasopressors on if diabetic:
    raise blood pressure: normal -> hi w.p. .9, lo -> normal w.p. .5, lo -> hi w.p. .4
    raise blood glucose by 1 w.p. .5

# Effect of the overlay node activation
Effect of vasopressors off if diabetic:
    blood pressure falls by 1 w.p. .05 instead of .1
    glucose does not fall - apply fluctuations below instead

# Flauctuation in system observations based on an event in the server.

Fluctuation in blood glucose levels (IV/insulin therapy are not possible actions):
    fluctuate w.p. .3 if diabetic
    fluctuate w.p. .1 if non-diabetic

Ref: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4530321/

# do we need additinal flauctuations?

Additional fluctuation regardless of other changes

what are these?

This order is applied:
    antibiotics, ventilation, vasopressors, fluctuations
'''

class MDP(object):

    def __init__(self, init_state_idx=None, init_state_idx_type='obs',
            policy_array=None, policy_idx_type='obs', p_busy_server=0.2):
        '''
        initialize the simulator
        '''
        assert p_busy_server >= 0 and p_busy_server <= 1, \
                "Invalid p_busy_server: {}".format(p_busy_server)
        assert policy_idx_type in ['obs', 'full', 'proj_obs']

        # Check the policy dimensions (states x actions)
        if policy_array is not None:
            assert policy_array.shape[1] == Action.NUM_ACTIONS_TOTAL
            if policy_idx_type == 'obs':
                assert policy_array.shape[0] == State.NUM_OBS_STATES
            elif policy_idx_type == 'full':
                assert policy_array.shape[0] == \
                        State.NUM_HID_STATES * State.NUM_OBS_STATES
            elif policy_idx_type == 'proj_obs':
                assert policy_array.shape[0] == State.NUM_PROJ_OBS_STATES

        # p_busy_server is used to generate random state if init_state is None
        self.p_busy_server = p_busy_server
        self.state = None

        # Only need to use init_state_idx_type if you are providing a state_idx!
        self.state = self.get_new_state(init_state_idx, init_state_idx_type)

        self.policy_array = policy_array
        self.policy_idx_type = policy_idx_type  # Used for mapping the policy to actions

    def get_new_state(self, state_idx = None, idx_type = 'obs', busy_server_idx = None):
        '''
        use to start MDP over.  A few options:

        Full specification:
        1. Provide state_idx with idx_type = 'obs' + busy_server_idx
        2. Provide state_idx with idx_type = 'full', busy_server_idx is ignored
        3. Provide state_idx with idx_type = 'proj_obs' + busy_server_idx*

        * This option will set glucose to a normal level

        Random specification
        4. State_idx, no busy_server_idx: Latter will be generated
        5. No state_idx, no busy_server_idx:  Completely random
        6. No state_idx, busy_server_idx given:  Random conditional on diabetes
        '''
        assert idx_type in ['obs', 'full', 'proj_obs']
        option = None
        if state_idx is not None:
            if idx_type == 'obs' and busy_server_idx is not None:
                option = 'spec_obs'
            elif idx_type == 'obs' and busy_server_idx is None:
                option = 'spec_obs_no_diab'
                busy_server_idx = np.random.binomial(1, self.p_busy_server)
            elif idx_type == 'full':
                option = 'spec_full'
            elif idx_type == 'proj_obs' and busy_server_idx is not None:
                option = 'spec_proj_obs'
        elif state_idx is None and busy_server_idx is None:
            option = 'random'
        elif state_idx is None and busy_server_idx is not None:
            option = 'random_cond_busy'

        assert option is not None, "Invalid specification of new state"

        if option in ['random', 'random_cond_busy'] :
            init_state = self.generate_random_state(busy_server_idx)
            # Do not start in failed or recovered state
            while init_state.check_absorbing_state():
                init_state = self.generate_random_state(busy_server_idx)
        else:
            # Note that busy_server_idx will be ignored if idx_type = 'full'
            init_state = State(
                    state_idx=state_idx, idx_type=idx_type,
                    busy_server_idx=busy_server_idx)

        return init_state

    def generate_random_state(self, busy_server_idx=None):
        # Note that we will condition on diabetic idx if provided
	    # what the modification needed here?
        if busy_server_idx is None:
            busy_server_idx = np.random.binomial(1, self.p_busy_server)

	    # starting probiblities in particular observations
        # cpu_usage and gpu_usage w.p. [.25, .5, .25]
        cpu_usage_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        gpu_usage_state = np.random.choice(np.arange(3), p=np.array([.25, .5, .25]))
        
        # memory w.p. [.2, .8]
        memory_usage_state = np.random.choice(np.arange(2), p=np.array([.2, .8]))

	    # Start the 
        # network usage has 5 different values(glaucose)
        if busy_server_idx == 0:
            network_usage_state  = np.random.choice(np.arange(5), \
                p=np.array([.05, .15, .6, .15, .05]))
        else:
            network_usage_state  = np.random.choice(np.arange(5), \
                p=np.array([.01, .05, .15, .6, .19]))
        # all the actions has the value of '0'
        more_process_state = 0
        pause_for_maintenance_state = 0
        activate_overlay_node = 0
        
        state_categs = [cpu_usage_state, gpu_usage_state, memory_usage_state,
                network_usage_state , more_process_state, pause_for_maintenance_state, activate_overlay_node]

        return State(state_categs=state_categs, busy_server_idx=busy_server_idx)

    def transition_more_users_on(self):
        '''
        let more process state on
        cpu_usage, gpu_usage: medium -> high w.p. 0.5
        '''
        self.state.more_process_state = 1
        if self.state.cpu_usage_state == 1 and np.random.uniform(0,1) < 0.5:
            self.state.cpu_usage_state = 2
        if self.state.gpu_usage_state == 1 and np.random.uniform(0,1) < 0.5:
            self.state.gpu_usage_state = 2

    def transition_more_users_off(self):
        '''
        more users state off
        if antibiotics was on: cpu usage, gpu usage: hi -> normal w.p. .1
        '''
        if self.state.more_process_state == 1:
            if self.state.cpu_usage_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.cpu_usage_state = 2
            if self.state.gpu_usage_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.gpu_usage_state = 2
            self.state.more_process_state = 0

    def transition_overlay_node_on(self):
        '''
        transition overlay node state on
        ( how it can change the beahvior of the system)
        
        percent memory usage: low -> normal w.p. .7
        '''
        self.state.activate_overlay_node = 1
        if self.state.memory_usage_state == 2 and np.random.uniform(0,1) < 0.7:
            self.state.memory_usage_state = 1

    def transition_overlay_node_off(self):
        '''
        transition_overlay_node state off
        if ventilation was on: percent oxygen: normal -> lo w.p. .1
        '''
        if self.state.activate_overlay_node == 1:
            if self.state.memory_usage_state == 1 and np.random.uniform(0,1) < 0.1:
                self.state.memory_usage_state = 0
            self.state.activate_overlay_node = 0

    def transition_pause_for_maintenace_on(self):
        '''
        vasopressor state on
        for non-diabetic:
            sys bp: low -> normal, normal -> hi w.p. .7
        for diabetic:
            raise blood pressure: normal -> hi w.p. .9,
                lo -> normal w.p. .5, lo -> hi w.p. .4
            raise blood glucose by 1 w.p. .5
        '''
        self.state.pause_for_maintenance_state = 1
        if self.state.busy_server_idx == 0:
            if np.random.uniform(0,1) < 0.7:
                if self.state.gpu_usage_state == 0:
                    self.state.gpu_usage_state = 1
                elif self.state.gpu_usage_state == 1:
                    self.state.gpu_usage_state = 2
        else:
            if self.state.gpu_usage_state == 1:
                if np.random.uniform(0,1) < 0.9:
                    self.state.gpu_usage_state = 2
            elif self.state.gpu_usage_state == 0:
                up_prob = np.random.uniform(0,1)
                if up_prob < 0.5:
                    self.state.gpu_usage_state = 1
                elif up_prob < 0.9:
                    self.state.gpu_usage_state = 2
            if np.random.uniform(0,1) < 0.5:
                self.state.network_usage_state  = min(4, self.state.network_usage_state  + 1)

    def transition_pause_for_maintenance_off(self):
        '''
        vasopressor state off
        if vasopressor was on:
            for non-diabetics, sys bp: normal -> low, hi -> normal w.p. .1
            for diabetics, blood pressure falls by 1 w.p. .05 instead of .1
        '''
        if self.state.pause_for_maintenance_state == 1:
            if self.state.busy_server_idx == 0:
                if np.random.uniform(0,1) < 0.1:
                    self.state.gpu_usage_state = max(0, self.state.gpu_usage_state - 1)
            else:
                if np.random.uniform(0,1) < 0.05:
                    self.state.gpu_usage_state = max(0, self.state.gpu_usage_state - 1)
            self.state.pause_for_maintenance_state = 0
    
    # is it neccesiry to have this flauctuations?
    def transition_fluctuate(self, cpu_usage_fluctuate, gpu_usage_fluctuate, memory_usage_fluctuate, \
        network_usage_fluctuate):
        '''
        all (non-treatment) states fluctuate +/- 1 w.p. 0.1
        exception: glucose flucuates +/- 1 w.p. 0.3 if diabetic
        '''
        if cpu_usage_fluctuate:
            cpu_usage_prob = np.random.uniform(0,1)
            if cpu_usage_prob < 0.1:
                self.state.cpu_usage_state = max(0, self.state.cpu_usage_state - 1)
            elif cpu_usage_prob < 0.2:
                self.state.cpu_usage_state = min(2, self.state.cpu_usage_state + 1)
        if gpu_usage_fluctuate:
            gpu_usage_prob = np.random.uniform(0,1)
            if gpu_usage_prob < 0.1:
                self.state.gpu_usage_state = max(0, self.state.gpu_usage_state - 1)
            elif gpu_usage_prob < 0.2:
                self.state.gpu_usage_state = min(2, self.state.gpu_usage_state + 1)
        if memory_usage_fluctuate:
            memory_usage_prob = np.random.uniform(0,1)
            if memory_usage_prob < 0.1:
                self.state.memory_usage_state = max(0, self.state.memory_usage_state - 1)
            elif memory_usage_prob < 0.2:
                self.state.memory_usage_state = min(1, self.state.memory_usage_state + 1)
        if network_usage_fluctuate:
            network_usage = np.random.uniform(0,1)
            if self.state.busy_server_idx == 0:
                if network_usage < 0.1:
                    self.state.network_usage_state  = max(0, self.state.network_usage_state  - 1)
                elif network_usage < 0.2:
                    self.state.network_usage_state  = min(1, self.state.network_usage_state  + 1)
            else:
                if network_usage < 0.3:
                    self.state.network_usage_state  = max(0, self.state.network_usage_state  - 1)
                elif network_usage < 0.6:
                    self.state.network_usage_state  = min(4, self.state.network_usage_state  + 1)

    def calculateReward(self):
        num_abnormal = self.state.get_num_abnormal()
        if num_abnormal >= 3:
            return -1
        elif num_abnormal == 0 and not self.state.on_treatment():
            return 1
        return 0

    def transition(self, action):
        self.state = self.state.copy_state()

        if action.more_users == 1:
            self.transition_more_users_on()
            cpu_usage_fluctuate = False
            gpu_usage_fluctuate = False
        elif self.state.more_process_state == 1:
            self.transition_more_users_off()
            cpu_usage_fluctuate = False
            gpu_usage_fluctuate = False
        else:
            cpu_usage_fluctuate = True
            gpu_usage_fluctuate = True

        if action.overlay_node == 1:
            self.transition_overlay_node_on()
            memory_usage_fluctuate = False
        elif self.state.activate_overlay_node == 1:
            self.transition_overlay_node_off()
            memory_usage_fluctuate = False
        else:
            memory_usage_fluctuate = True

        network_usage_fluctuate = True

        if action.pause_for_maintenance == 1:
            self.transition_pause_for_maintenace_on()
            gpu_usage_fluctuate = False
            network_usage_fluctuate = False
        elif self.state.pause_for_maintenance_state == 1:
            self.transition_pause_for_maintenance_off()
            gpu_usage_fluctuate = False

        self.transition_fluctuate(cpu_usage_fluctuate, gpu_usage_fluctuate, memory_usage_fluctuate, \
            network_usage_fluctuate)

        return self.calculateReward()

    def select_actions(self):
        assert self.policy_array is not None
        probs = self.policy_array[self.state.get_state_idx(self.policy_idx_type)]
        aev_idx = np.random.choice(np.arange(Action.NUM_ACTIONS_TOTAL), p=probs)
        return Action(action_idx = aev_idx)
