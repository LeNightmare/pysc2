import random
import math
import os.path

import numpy as np
import pandas as pd

from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features



# no action
_NO_OP = actions.FUNCTIONS.no_op.id


# select point
_SELECT_POINT = actions.FUNCTIONS.select_point.id


# build something
_BUILD_SUPPLY_DEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_COMMANDCENTER = actions.FUNCTIONS.Build_CommandCenter_screen.id
_BUILD_REFINERY = actions.FUNCTIONS.Build_Refinery_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_BUILD_FACTORY = actions.FUNCTIONS.Build_Reactor_Factory_screen.id
_BUILD_STARPORT = actions.FUNCTIONS.Build_Reactor_Starport_screen.id
_BUILD_ENGINEERINGBAY = actions.FUNCTIONS.Build_EngineeringBay_screen.id
_BUILD_ARMORY = actions.FUNCTIONS.Build_Armory_screen.id


# train
_TRAIN_SCV = actions.FUNCTIONS.Train_SCV_quick.id



_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_TRAIN_MARAUDER = actions.FUNCTIONS.Train_Marauder_quick.id
_TRAIN_REAPER = actions.FUNCTIONS.Train_Reaper_quick.id
_TRAIN_GHOST = actions.FUNCTIONS.Train_Ghost_quick.id



_TRAIN_BANSHEE = actions.FUNCTIONS.Train_Banshee_quick.id
_TRAIN_LIBERATOR = actions.FUNCTIONS.Train_Liberator_quick.id
_TRAIN_TAVEN = actions.FUNCTIONS.Train_Raven_quick.id
_TRAIN_MEDIVAC = actions.FUNCTIONS.Train_Medivac_quick.id
_TRAIN_VIKINGFIGHTER = actions.FUNCTIONS.Train_VikingFighter_quick.id



_TRAIN_HELLION = actions.FUNCTIONS.Train_Hellion_quick.id
_TRAIN_SIEGETANK = actions.FUNCTIONS.Train_SiegeTank_quick.id
_TRAIN_THOR = actions.FUNCTIONS.Train_Thor_quick.id
_TRAIN_WIDOWMINE = actions.FUNCTIONS.Train_WidowMine_quick.id
_TRAIN_CYCLONE = actions.FUNCTIONS.Train_Cyclone_quick.id



# select army
_SELECT_ARMY = actions.FUNCTIONS.select_army.id


# attack minimap
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id




_HARVEST_GATHER = actions.FUNCTIONS.Harvest_Gather_screen.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

# select idle worker
# max worker capacity

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 4
_ARMY_SUPPLY = 5



# building
_TERRAN_COMMANDCENTER = 18
_TERRAN_COMMANDCENTERFLYING = 36
_TERRAN_ORBITALCOMMAND = 132
_TERRAN_ORBITALCOMMANDFLYING = 134
_TERRAN_PLANETARYFORTRESS = 130


_TERRAN_REFINERY = 20

_TERRAN_SUPPLY_DEPOT = 19
_TERRAN_SUPPLYDEPOTLOWERED = 47


_TERRAN_BARRACKS = 21
_TERRAN_BARRACKSFLYING = 46
_TERRAN_BARRACKSREACTOR = 38
_TERRAN_BARRACKSTECHLAB = 37

_TERRAN_FACTORY = 27
_TERRAN_FACTORYFLYING = 43
_TERRAN_FACTORYREACTOR = 40
_TERRAN_FACTORYTECHLAB = 39


_TERRAN_STARPORT = 28
_TERRAN_STARPORTFLYING = 44
_TERRAN_STARPORTREACTOR = 42
_TERRAN_STARPORTTECHLAB = 41

_TERRAN_ENGINEERINGBAY = 22
_TERRAN_GHOSTACADEMY = 26
_TERRAN_ARMORY = 29


# unit
_TERRAN_SCV = 45
_TERRAN_MULE = 268

## barracks
_TERRAN_GHOST = 50
_TERRAN_REAPER = 49
_TERRAN_MARAUDER = 51
_TERRAN_MARINE = 48



## starport
_TERRAN_BANSHEE = 55
_TERRAN_LIBERATOR = 689
_TERRAN_LIBERATORAG = 734
_TERRAN_MEDIVAC = 54
_TERRAN_RAVEN = 56
_TERRAN_VIKINGASSAULT = 34
_TERRAN_VIKINGFIGHTER = 35

## factory
_TERRAN_HELLION = 53
_TERRAN_SIEGETANK = 33
_TERRAN_SIEGETANKSIEGED = 32
_TERRAN_THOR = 52
_TERRAN_THORAP = 691
_TERRAN_WIDOWMINE = 498
_TERRAN_WIDOWMINEBURROWED = 500
_TERRAN_CYCLONE = 692

_NEUTRAL_MINERAL_FIELD = 341

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

DATA_FILE = 'sparse_agent_data'
DATA_FILE_NEW = 'my_q_learning_data'



ACTION_DO_NOTHING = 'donothing'
ACTION_BUILD_SUPPLY_DEPOT = 'buildsupplydepot'
ACTION_BUILD_BARRACKS = 'buildbarracks'
ACTION_BUILD_MARINE = 'buildmarine'
ACTION_ATTACK = 'attack'

ACTION_BUILD_REFINERY = 'buildrefinery'
ACTION_TRAIN_SCV = 'trainSCV'
ACTION_BUILD_FACTORY = 'buildfactory'
ACTION_TRAIN_HELLION = 'trainhellion'
ACTION_BUILD_ENGINEERINGBAY = 'buildengineerbay'



smart_actions = [
    ACTION_DO_NOTHING,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_MARINE,
]


afk = [
    ACTION_DO_NOTHING
]

disrupt = [


]

sneak = [



]

manage = [
    ACTION_TRAIN_SCV,
    ACTION_BUILD_SUPPLY_DEPOT,
    ACTION_BUILD_REFINERY,
]

attack = [

    ACTION_ATTACK
]
defend = [


]
explore = [


]
build = [

    ACTION_BUILD_BARRACKS,
    ACTION_BUILD_ENGINEERINGBAY,
    #ACTION_BUILD_FACTORY


]
produce = [

    ACTION_BUILD_MARINE,
    #ACTION_TRAIN_HELLION,

]
research = [


]
generalActions = [
    afk,
    # disrupt,
    # sneak,
    manage,
    attack,
    # defend,
    # explore,
    build,
    produce,
    # research,
]
# reward: disrupt, income, valueTrade, lost, explore, combatPower
# disrupt =
# income = mineral + gas
# valueTrade = 4 * destroy buildings + kill units
# lost = 2 * lose buildings + lose units
# explore = points explored
# combatPower =







for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))



class QLearningAgent:
    def __init__(self, actionSet, alpha, epsilon, gamma, num):
        """
        :param  actions, action list
        :param alpha:  learning rate
        :param epsilon: exploration rate
        :param gamma:  discount factor
        :param num: number of training episodes

        """
        self.actionSet = actionSet
        self.alpha = float (alpha)
        self.epsilon = float(epsilon)
        self.discount = float(gamma)
        self.numTraining = int(num)

        self.dic = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def getQValue(self, state, action):
        if (state, action) not in self.q:
            self.q[(state, action)] = 0.0
        return self.q[(state, action)]

    def computeValueFromQValues(self, state):
        actions = self.getLegalActions(state)
        if not actions:
            return 0.0
        return max(map(lambda a: self.getQValue(state, a), self.getLegalActions(state)))

    def computeActionFromQValues(self, state):

        actions = self.getLegalActions(state)
        if not self.getLegalActions(state):
            return None

        value = map(lambda a: self.getQValue(state, a), actions)
        best = max(value)
        index = [index for index in range(len(value)) if value[index] == best]
        choice = random.choice(index)  # Pick randomly among the best
        return actions[choice]


    def getAction(self, state):
        legalActions = self.getLegalActions(state)
        if not legalActions:
            return None
        elif self.flipCoin(self.epsilon):
            return random.choice(legalActions)
        else:
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        self.q[(state, action)] = (1 - self.alpha) * self.getQValue(state, action) + self.alpha * (
                reward + self.discount * self.computeValueFromQValues(nextState))

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


    def getLegalActions(self, state):
        return self.actionSet

    def flipCoin(self, p):
        r = random.random()
        return r < p

# Stolen from https://github.com/MorvanZhou/Reinforcement-learning-with-tensorflow
class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions  # a list
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.uniform() < self.epsilon:
            # choose best action
            state_action = self.q_table.ix[observation, :]

            # some actions have the same value
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            action = state_action.idxmax()
        else:
            # choose random action
            action = np.random.choice(self.actions)

        return action

    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        self.check_state_exist(s)

        q_predict = self.q_table.ix[s, a]

        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.ix[s_, :].max()
        else:
            q_target = r  # next state is terminal

        # update
        self.q_table.ix[s, a] += self.lr * (q_target - q_predict)

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            # append new state to q table
            self.q_table = self.q_table.append(
                pd.Series([0] * len(self.actions), index=self.q_table.columns, name=state))


class SparseAgent(base_agent.BaseAgent):


    def __init__(self):
        super(SparseAgent, self).__init__()

        # self.qlearn = QLearningTable(actions=list(range(len(smart_actions))))
        self.actions = []
        for a in generalActions:
            self.actions += a
        # self.actions = map(lambda a : list(range(len(a))) , generalActions)
        print self.actions
        self.qlearn = QLearningTable(actions= list(range(len(self.actions))))
        # self.myQlearning = QLearningAgent(actions = list(range(len(smart_actions))))

        self.previous_action = None
        self.previous_state = None

        self.cc_y = None
        self.cc_x = None

        self.move_number = 0

        if os.path.isfile(DATA_FILE + '.gz'):
            self.qlearn.q_table = pd.read_pickle(DATA_FILE + '.gz', compression='gzip')
            # self.myQlearning.dic = pd.read_pickle(DATA_FILE_NEW + '.gz', compression='gzip')


    def transformDistance(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]

    def transformLocation(self, x, y):
        if not self.base_top_left:
            return [64 - x, 64 - y]

        return [x, y]

    def splitAction(self, action_id):
        # smart_action = smart_actions[action_id]
        smart_action = self.actions[action_id]

        x = 0
        y = 0
        if '_' in smart_action:
            smart_action, x, y = smart_action.split('_')

        return (smart_action, x, y)

    def step(self, obs):
        super(SparseAgent, self).step(obs)

        if obs.last():
            reward = obs.reward

            self.qlearn.learn(str(self.previous_state), self.previous_action, reward, 'terminal')
            # self.myQlearning.
            self.qlearn.q_table.to_pickle(DATA_FILE + '.gz', 'gzip')

            self.previous_action = None
            self.previous_state = None

            self.move_number = 0

            return actions.FunctionCall(_NO_OP, [])

        unit_type = obs.observation['screen'][_UNIT_TYPE]

        if obs.first():
            player_y, player_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = 1 if player_y.any() and player_y.mean() <= 31 else 0

            self.cc_y, self.cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()

        cc_y, cc_x = (unit_type == _TERRAN_COMMANDCENTER).nonzero()
        cc_count = 1 if cc_y.any() else 0

        depot_y, depot_x = (unit_type == _TERRAN_SUPPLY_DEPOT).nonzero()
        supply_depot_count = int(round(len(depot_y) / 69))

        ref_y, ref_x = (unit_type == _TERRAN_REFINERY).nonzero()
        refinery_count = int(round(len(ref_y) / 69))

        barracks_y, barracks_x = (unit_type == _TERRAN_BARRACKS).nonzero()
        barracks_count = int(round(len(barracks_y) / 137))

        eb_y, eb_x = (unit_type == _TERRAN_ENGINEERINGBAY).nonzero()
        eb_count = int(round(len(eb_y))/ 137)


        factory_y, factory_x = (unit_type == _TERRAN_FACTORY).nonzero()
        factory_count = int(round(len(factory_y))/ 137)


        if self.move_number == 0:
            self.move_number += 1

            current_state = np.zeros(14)
            current_state[0] = cc_count
            current_state[1] = supply_depot_count
            current_state[2] = barracks_count
            current_state[3] = obs.observation['player'][_ARMY_SUPPLY]
            current_state[4] = eb_count
            current_state[5] = factory_count
            current_state[6] = refinery_count

            hot_squares = np.zeros(7)
            enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
            for i in range(0, len(enemy_y)):
                y = int(math.ceil((enemy_y[i] + 1) / 32))
                x = int(math.ceil((enemy_x[i] + 1) / 32))

                hot_squares[((y - 1) * 2) + (x - 1)] = 1

            if not self.base_top_left:
                hot_squares = hot_squares[::-1]

            for i in range(0, 7):
                current_state[i + 7] = hot_squares[i]

            if self.previous_action is not None:
                self.qlearn.learn(str(self.previous_state), self.previous_action, 0, str(current_state))

            rl_action = self.qlearn.choose_action(str(current_state))

            self.previous_state = current_state
            self.previous_action = rl_action

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:# or smart_action == ACTION_BUILD_ENGINEERINGBAY
                    # or smart_action == ACTION_BUILD_FACTORY\
                unit_y, unit_x = (unit_type == _TERRAN_SCV).nonzero()

                if unit_y.any():
                    i = random.randint(0, len(unit_y) - 1)
                    target = [unit_x[i], unit_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if barracks_y.any():
                    i = random.randint(0, len(barracks_y) - 1)
                    target = [barracks_x[i], barracks_y[i]]

                    return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])
            # elif smart_action == ACTION_TRAIN_HELLION:
            #     if factory_y.any():
            #         i = random.randint(0, len(factory_y) - 1)
            #         target = [factory_x[i], factory_y[i]]
            #
            #         return actions.FunctionCall(_SELECT_POINT, [_SELECT_ALL, target])

            elif smart_action == ACTION_ATTACK:
                if _SELECT_ARMY in obs.observation['available_actions']:
                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])

        elif self.move_number == 1:
            self.move_number += 1

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if supply_depot_count < 4 and _BUILD_SUPPLY_DEPOT in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if supply_depot_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), -35, round(self.cc_y.mean()), 0)
                        elif supply_depot_count == 1:
                            target = self.transformDistance(round(self.cc_x.mean()), -25, round(self.cc_y.mean()), -25)
                        elif supply_depot_count == 2:
                            target = self.transformDistance(round(self.cc_x.mean()), -20, round(self.cc_y.mean()), -25)
                        elif supply_depot_count == 3:
                            target = self.transformDistance(round(self.cc_x.mean()), -15, round(self.cc_y.mean()), -25)

                        return actions.FunctionCall(_BUILD_SUPPLY_DEPOT, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_BARRACKS:
                if barracks_count < 3 and _BUILD_BARRACKS in obs.observation['available_actions']:
                    if self.cc_y.any():
                        if barracks_count == 0:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), -9)
                        elif barracks_count == 2:
                            target = self.transformDistance(round(self.cc_x.mean()), 15, round(self.cc_y.mean()), 12)
                        elif barracks_count == 3:
                            target = self.transformDistance(round(self.cc_x.mean()), 12, round(self.cc_y.mean()), 15)

                        return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])

            elif smart_action == ACTION_BUILD_MARINE:
                if _TRAIN_MARINE in obs.observation['available_actions']:
                    return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])

            elif smart_action == ACTION_ATTACK:
                do_it = True

                if len(obs.observation['single_select']) > 0 and obs.observation['single_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if len(obs.observation['multi_select']) > 0 and obs.observation['multi_select'][0][0] == _TERRAN_SCV:
                    do_it = False

                if do_it and _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    x_offset = random.randint(-1, 1)
                    y_offset = random.randint(-1, 1)

                    print 'position:', int(x), int(y)
                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED,
                                                                  self.transformLocation(int(x) + (x_offset * 8),
                                                                                         int(y) + (y_offset * 8))])

        elif self.move_number == 2:
            self.move_number = 0

            smart_action, x, y = self.splitAction(self.previous_action)

            if smart_action == ACTION_BUILD_BARRACKS or smart_action == ACTION_BUILD_SUPPLY_DEPOT:
                if _HARVEST_GATHER in obs.observation['available_actions']:
                    unit_y, unit_x = (unit_type == _NEUTRAL_MINERAL_FIELD).nonzero()

                    if unit_y.any():
                        i = random.randint(0, len(unit_y) - 1)

                        m_x = unit_x[i]
                        m_y = unit_y[i]

                        target = [int(m_x), int(m_y)]

                        return actions.FunctionCall(_HARVEST_GATHER, [_QUEUED, target])

        return actions.FunctionCall(_NO_OP, [])