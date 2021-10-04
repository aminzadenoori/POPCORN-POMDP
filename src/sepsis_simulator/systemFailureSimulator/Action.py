import numpy as np

class Action(object):

    NUM_ACTIONS_TOTAL = 8
    MORE_USERS_STRING = "more_users"
    ACTIVATE_OVERLAY_NODE = "activate_overlay_node"
    PAUSE_FOR_MAINTENANCE = "pause_for_maintenance"
    ACTION_VEC_SIZE = 3

    def __init__(self, selected_actions = None, action_idx = None):
        assert (selected_actions is not None and action_idx is None) \
            or (selected_actions is None and action_idx is not None), \
            "must specify either set of action strings or action index"
        if selected_actions is not None:
            if Action.MORE_USERS_STRING in selected_actions:
                self.more_users = 1
            else:
                self.more_users = 0
            if Action.ACTIVATE_OVERLAY_NODE in selected_actions:
                self.activate_overlay_node = 1
            else:
                self.activate_overlay_node = 0
            if Action.PAUSE_FOR_MAINTENANCE in selected_actions:
                self.pause_for_maintenance = 1
            else:
                self.pause_for_maintenance = 0
        else:
            mod_idx = action_idx
            term_base = Action.NUM_ACTIONS_TOTAL/2
            self.more_users = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.activate_overlay_node = np.floor(mod_idx/term_base).astype(int)
            mod_idx %= term_base
            term_base /= 2
            self.pause_for_maintenance = np.floor(mod_idx/term_base).astype(int)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and \
            self.more_users == other.more_users and \
            self.activate_overlay_node == other.activate_overlay_node and \
            self.pause_for_maintenance == other.pause_for_maintenance

    def __ne__(self, other):
        return not self.__eq__(other)

    def get_action_idx(self):
        assert self.more_users in (0, 1)
        assert self.activate_overlay_node in (0, 1)
        assert self.pause_for_maintenance in (0, 1)
        return 4*self.more_users + 2*self.activate_overlay_node + self.pause_for_maintenance

    def __hash__(self):
        return self.get_action_idx()

    def get_selected_actions(self):
        selected_actions = set()
        if self.more_users == 1:
            selected_actions.add(Action.MORE_USERS_STRING)
        if self.activate_overlay_node == 1:
            selected_actions.add(Action.ACTIVATE_OVERLAY_NODE)
        if self.pause_for_maintenance == 1:
            selected_actions.add(Action.PAUSE_FOR_MAINTENANCE)
        return selected_actions

    def get_abbrev_string(self):
        '''
        AEV: more_users, activate_overlay_node, pause_for_maintenance
        '''
        output_str = ''
        if self.more_users == 1:
            output_str += 'A'
        if self.activate_overlay_node == 1:
            output_str += 'E'
        if self.pause_for_maintenance == 1:
            output_str += 'V'
        return output_str

    def get_action_vec(self):
        return np.array([[self.more_users], [self.activate_overlay_node], [self.pause_for_maintenance]])
