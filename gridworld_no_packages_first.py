import numpy as np
import matplotlib.pyplot as plt

class Gworld():


    learning_rate = 0.01
    discount_rate = 0.9
    ratio_of_exploitation = 0

    def __init__(self, height, width, startingtuple):

        self.height = height
        self.width = width
        self.i = startingtuple[0]
        self.j = startingtuple[1]
        self.move_list = []
        self.terminals_dict = None
        self.actions_dict = None
        self.rounds = 0
        self.counter_win = 0
        self.win_move_list = []

    def set_env(self, rewards):

        """this takes in a list of tuples, each of three numbers specifying the location of each reward and the corresponding reward, eg if the top left cell in the grid has a reward of +5, the tuple would be (0,0,5), and there would be one entry in the list for every such location & reward"""

        # first, a dictionary of terminal states/rewards is built:

        terminals_dict = {}
        for tupleobj in rewards:

            key1 = tuple((tupleobj[0], tupleobj[1]))
            terminals_dict[key1] = tupleobj[2]

        self.terminals_dict = terminals_dict

        # next we build an actions and assc values dictionary. The keys will be the states and the values will be dictionaries with each key being an action and each value being the
        # q value for that action

        # we build a list of all the states based on the dimensions of the grid

        states_all_list = []
        height = self.height
        width = self.width

        for i in range(height):
            for j in range(width):
                tup_ob = tuple((i,j))
                states_all_list.append(tup_ob)

        # we take out all the states which are terminal, then, make a list/dictionary of all the possible actions in that state, then each of those actions
        # are the keys to a nested dictionary. It looks as follows for one parent dictinary entry (position top left, only two possible states): (0,0): {'d':0, 'r': 0}, here
        # we can see there is one position on the grid as key, its possibel actions are in a nested dictionary, each of those actions is initialised with present q values

        dict_all_spaces = dict(zip(states_all_list, [None] * len(states_all_list)))
        for key in list(self.terminals_dict.keys()):
            dict_all_spaces.pop(key)

        for key in list(dict_all_spaces.keys()):

            if key == (0,0):
                subkeys = ('d', 'r')

            elif key == (0,width-1):
                subkeys = ('d','l')

            elif key  == (height-1, width-1):
                subkeys = ('u','l')

            elif key == (height-1, 0):
                subkeys = ('u', 'r')

            elif key[0] == 0:
                subkeys = ('d', 'l', 'r')

            elif key[0] == height-1:
                subkeys = ('u', 'l', 'r')

            elif key[1]  == 0:
                subkeys = ('u', 'd','r')

            elif key[1] == width-1:
                subkeys = ('u', 'd', 'l')

            else:
                subkeys = ('u', 'd', 'l', 'r')

            subkeys_dict = dict(zip(subkeys, [0]*len(subkeys)))
            dict_all_spaces[key] = subkeys_dict

        self.actions_dict = dict_all_spaces
        print(self.terminals_dict)

    @staticmethod
    def set_exp():

        """this sets the change of exploiting at every step, then runs a little numpy test"""

        chance_exploit = Gworld.ratio_of_exploitation
        chance_explore = 1 - Gworld.ratio_of_exploitation
        decision_exp = np.random.choice(['explore','exploit'], p=[chance_explore, chance_exploit])
        return decision_exp

    def terminal_checker(self):

        present_position = (self.i, self.j)
        if present_position in self.terminals_dict:

            if self.terminals_dict[self.i, self.j] == 1:
                self.counter_win += 1
                self.win_move_list = self.move_list

            self.rounds += 1
            reward = self.terminals_dict[self.i, self.j]
            for index, move in enumerate(reversed(self.move_list)):

                self.actions_dict[move[0], move[1]][move[2]] = reward*\
                                                               (Gworld.discount_rate**(index+1))*\
                                                               Gworld.learning_rate

            self.i = 0
            self.j = 0
            self.move_list = []
            return True

        else:

            return False



    def step(self):

        present_position = (self.i, self.j)
        options = list(self.actions_dict[tuple(present_position)].keys())
        the_exp_decision = Gworld.set_exp()

        # if we decide to explore, then we choose randomly between the options of actions

        if the_exp_decision == 'explore':

            next_move = np.random.choice(options)
            this_action_list = [self.i, self.j, next_move]
            self.move_list.append(this_action_list)

            if next_move == 'u':
                self.i-=1

            elif next_move == 'd':
                self.i+=1

            elif next_move == 'l':
                self.j-=1

            elif next_move == 'r':
                self.j+=1

        # if we decide to exploit, we make a list of all the actions with the equal highest q value
        # then randomly choose one. We do it as so because initially, there will be
        # of zero q values

        elif the_exp_decision == 'exploit':

            max_option = max(self.actions_dict[tuple(present_position)].values())
            max_list = [k for k,v in self.actions_dict[tuple(present_position)].items() if v == max_option]

            next_move = np.random.choice(max_list)
            this_action_list = [self.i, self.j, next_move]
            self.move_list.append(this_action_list)

            if next_move == 'u':
                self.i-=1

            elif next_move == 'd':
                self.i+=1

            elif next_move == 'l':
                self.j-=1

            elif next_move == 'r':
                self.j+=1

    def start_image(self):

        img = np.zeros((self.height, self.width))
        for key, val in self.terminals_dict.items():
            if val == -1:
                img[key] = -0.2

            elif val == 1:
                img[key] = 0.5

        fig, (ax0) = plt.subplots(1, 1)

        c = ax0.pcolor(img)
        ax0.set_title('Starting grid')
        plt.show()

    def end_image(self):

        img = np.zeros((self.height, self.width))
        for key, val in self.terminals_dict.items():
            if val == -1:
                img[key] = -0.5

            elif val == 1:
                img[key] = 0.25

        for move in self.win_move_list:

            newli = move[0:2]
            img[newli[0], newli[1]] = 0.5


        fig, (ax0) = plt.subplots(1, 1)

        c = ax0.pcolor(img)
        ax0.set_title('Ending path grid')

        # fig.tight_layout()
        plt.show()




# def gworld_learner(height, width, start_loc, steps_entry, env_list):

height = 6
width = 6
start_loc = (0,0)
steps_entry = 1000000
env_list = [(1,4,-1), (2,4,-1), (3,4,-1), (2,5,1)]



nelson_learner = Gworld(height,width,start_loc)
nelson_learner.set_env(env_list)
loops = steps_entry
print('exp rate', nelson_learner.ratio_of_exploitation)
nelson_learner.start_image()

for i in range(loops):


    nelson_learner.step()
    nelson_learner.terminal_checker()
    if i%(loops*0.1) == 0:
        nelson_learner.ratio_of_exploitation+=0.099

print(nelson_learner.ratio_of_exploitation)
nelson_learner.end_image()
