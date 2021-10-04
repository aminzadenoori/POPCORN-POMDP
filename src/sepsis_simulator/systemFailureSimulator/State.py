import numpy as np

'''
Includes Gpu usage resource : 0-3
    (lo2 - counts as abnormal, lo1, normal, hi1, hi2 - counts as abnormal)

is it correct to have these ranges?

Initial distribution:
    [.05, .15, .6, .15, .05] for non-congested server and [.01, .05, .15, .6, .19] for congested server
'''

class State(object):
    # is it correct to have these states?
    NUM_OBS_STATES = 720
    # Is it correct to have this hidden state?
    NUM_HID_STATES = 2  # Binary value of a faluating variable
    # What is the number of proj OBJ?
    NUM_PROJ_OBS_STATES = int(720 / 5)  # Marginalizing over glucose
    # Number of full states.
    NUM_FULL_STATES = int(NUM_OBS_STATES * NUM_HID_STATES)

    def __init__(self,
            state_idx = None, idx_type = 'obs',
            busy_server_idx = None, state_categs = None):

        assert state_idx is not None or state_categs is not None
        assert ((busy_server_idx is not None and busy_server_idx in [0, 1]) or
                (state_idx is not None and idx_type == 'full'))

        assert idx_type in ['obs', 'full', 'proj_obs']

        if state_idx is not None:
            self.set_state_by_idx(
                    state_idx, idx_type=idx_type, busy_server_idx=busy_server_idx)
        elif state_categs is not None:
            assert len(state_categs) == 7, "must specify 7 state variables"
            self.cpu_usage_state = state_categs[0]
            self.gpu_usage_state = state_categs[1]
            self.memory_usage_state = state_categs[2]
            self.network_usage_state = state_categs[3]
            self.more_processes_state = state_categs[4]
            self.pause_for_maintenance_state = state_categs[5]
            self.activate_overlay_node_state = state_categs[6]
            self.busy_server_idx = busy_server_idx

    
    # absorbing state
    # checking abnormality and checking on the treatment
    
    def check_absorbing_state(self):
        num_abnormal = self.get_num_abnormal()
        if num_abnormal >= 3:
            return True
        elif num_abnormal == 0 and not self.on_action_taken():
            return True
        return False
   
    # Set state what is this?
    def set_state_by_idx(self, state_idx, idx_type, busy_server_idx=None):
        """set_state_by_idx

        The state index is determined by using "bit" arithmetic, with the
        complication that not every state is binary

        :param state_idx: Given index
        :param idx_type: Index type, either observed (720), projected (144) or
        full (1440)
        :param busy_server_idx: If full state index not given, this is required
        """
	# What is this part is for?
        if idx_type == 'obs':
            term_base = State.NUM_OBS_STATES/3 # Starts with heart rate
        elif idx_type == 'proj_obs':
            term_base = State.NUM_PROJ_OBS_STATES/3
        elif idx_type == 'full':
            term_base = State.NUM_FULL_STATES/2 # Starts with diab

        # Start with the given state index( What is this mod_idx?)
        mod_idx = state_idx

        if idx_type == 'full':
            self.busy_server_idx = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 3 # This is for heart rate, the next item( what is this next item?)
        else:
            assert busy_server_idx is not None
            self.busy_server_idx = busy_server_idx

        self.cpu_usage_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 3
        self.gpu_usage_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.memory_usage_state = np.floor(mod_idx/term_base).astype(int)

        if idx_type == 'proj_obs':
            self.network_usage_state = 2
        else:
            mod_idx %= term_base
            term_base /= 5
            self.network_usage_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.more_processes_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.pause_for_maintenance_state = np.floor(mod_idx/term_base).astype(int)

        mod_idx %= term_base
        term_base /= 2
        self.activate_overlay_node_state = np.floor(mod_idx/term_base).astype(int)
    # What it returns?
    def get_state_idx(self, idx_type='obs'):
        '''
        returns integer index of state: significance order as in categorical array
        '''
        if idx_type == 'obs':
            categ_num = np.array([3,3,2,5,2,2,2])
            state_categs = [
                    self.cpu_usage_state ,
                    self.gpu_usage_state,
                    self.memory_usage_state,
                    self.network_usage_state,
                    self.more_processes_state,
                    self.pause_for_maintenance_state,
                    self.activate_overlay_node_state]

	# Without gloucose
        elif idx_type == 'proj_obs':
            categ_num = np.array([3,3,2,2,2,2])
            state_categs = [
                    self.cpu_usage_state ,
                    self.gpu_usage_state,
                    self.memory_usage_state,
                    self.more_processes_state,
                    self.pause_for_maintenance_state,
                    self.activate_overlay_node_state]

        elif idx_type == 'full':
	    # I need to change this part.
	    
            categ_num = np.array([2,3,3,2,5,2,2,2])
            state_categs = [
                    self.busy_server_idx,
                    self.cpu_usage_state ,
                    self.gpu_usage_state,
                    self.memory_usage_state,
                    self.network_usage_state,
                    self.more_processes_state,
                    self.pause_for_maintenance_state,
                    self.activate_overlay_node_state]

        sum_idx = 0
        prev_base = 1
        for i in range(len(state_categs)):
            idx = len(state_categs) - 1 - i
            sum_idx += prev_base*state_categs[idx]
            prev_base *= categ_num[idx]
        return sum_idx

    def __eq__(self, other):
        '''
        override equals: two states equal if all internal states same
        '''
        return isinstance(other, self.__class__) and \
            self.cpu_usage_state == other.cpu_usage_state and \
            self.gpu_usage_state == other.gpu_usage_state and \
            self.memory_usage_state == other.memory_usage_state and \
            self.network_usage_state == other.network_usage_state and \
            self.more_processes_state == other.more_processes_state and \
            self.pause_for_maintenance_state == other.pause_for_maintenance_state and \
            self.activate_overlay_node_state == other.activate_overlay_node_state

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return self.get_state_idx()

    # get num abnormal
    def get_num_abnormal(self):
        '''
        returns number of abnormal conditions
        '''
        num_abnormal = 0
        if self.cpu_usage_state != 1:
            num_abnormal += 1
        if self.gpu_usage_state != 1:
            num_abnormal += 1
        if self.memory_usage_state != 1:
            num_abnormal += 1
        if self.network_usage_state != 2:
            num_abnormal += 1
        return num_abnormal
    
    # check if an action is applied
    def on_action_taken(self):
        '''
        returns True iff any of 3 treatments active
        '''
        if self.more_processes_state == 0 and \
            self.pause_for_maintenance_state == 0 and self.activate_overlay_node_state == 0:
            return False
        return True

    # Check if more process inside is activated
    def on_more_processess(self):
        '''
        returns True iff more_processess active
        '''
        return self.more_processes_state == 1
    # Check if the overlay node activation is activated
    def on_pause_for_maintenancepressors(self):
        '''
        returns True iff pause_for_maintenancepressors active
        '''
        return self.pause_for_maintenance_state == 1

    # Check if another action is activated
    def on_activate_overlay_node(self):
        '''
        returns True iff activate_overlay_nodeilation active
        '''
        return self.activate_overlay_node_state == 1

    def copy_state(self):
        return State(state_categs = [
            self.cpu_usage_state ,
            self.gpu_usage_state,
            self.memory_usage_state,
            self.network_usage_state,
            self.more_processes_state,
            self.pause_for_maintenance_state,
            self.activate_overlay_node_state],
            busy_server_idx=self.busy_server_idx)

    def get_state_vector(self):
        return np.array([self.cpu_usage_state ,
            self.gpu_usage_state,
            self.memory_usage_state,
            self.network_usage_state,
            self.more_processes_state,
            self.pause_for_maintenance_state,
            self.activate_overlay_node_state]).astype(int)
