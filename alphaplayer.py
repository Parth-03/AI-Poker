import os
import sys
sys.path.append(os.environ["BASE_DIR"] + "/alpha-poker")

from pypokerengine.players import BasePokerPlayer
import numpy as np
from poker.pytorch.NNet import NNetWrapper as nnet
from poker.PokerLogic import PokerState
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.engine.card import Card

# TODO: change so everything only happens in declare_action and its stateless 
class AlphaTrainPlayer(BasePokerPlayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.opponent_uuid = None
        self.nn = nnet(None)
        self.hole = []
        self.community = []
        self.EHS = 0.0
        self.name = "alpha"
        self.stack = 0
    
    def declare_action(self, valid_actions, hole_card, round_state):
        # Set poker state
        state = PokerState()
        opponent_moves = self.__class__.get_last_n_actions(round_state["action_histories"], self.uuid, 5)
        state.set_opponent_moves(opponent_moves)
        state.set_hole(hole_card)
        state.set_community(round_state["community_card"])
        state.set_pot(round_state["pot"]["main"]["amount"])
        state.set_street(0)
        state["round_count"] = float(round_state["round_count"])

        # If for some reason this function was called at unexpected times due to parallelization, recalculate EHS
        if self.hole != hole_card or self.community != round_state["community_card"]:
            win_rate = estimate_hole_card_win_rate(30, 2, [Card.from_str(card) for card in hole_card], [Card.from_str(card) for card in round_state["community_card"]])
            state["EHS"] = win_rate
            self.EHS = win_rate
            self.hole = hole_card
            self.community = round_state["community_card"]
        else:
            state["EHS"] = self.EHS
        
        # TODO: run through MCTS instead of just plain predict
        pi, v = self.nn.predict(state.to_vector())
        return self.__class__.get_action(pi, valid_actions)
        

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        # Only calculate EHS when communnity cards change
        self.hole = hole_card
        pass
    
    def receive_street_start_message(self, street, round_state):
        # update the current EHS
        # print("Calculating EHS... with", self.hole, "and", round_state["community_card"])
        if self.hole and street != "showdown":
            self.EHS = estimate_hole_card_win_rate(30, 2, [Card.from_str(card) for card in self.hole], [Card.from_str(card) for card in round_state["community_card"]])
            self.community = round_state["community_card"]
        pass
    
    def receive_game_start_message(self, game_info):
        pass
    
    def receive_game_update_message(self, action, round_state):
        pass
    
    @staticmethod
    def get_last_n_actions(action_history, player_uuid, n):
        """
        Returns the last n actions (as strings) taken by the opponent to player_uuid, ordered oldest to newest.
        action_history: dict of street -> list of action dicts (see example above)
        player_uuid: string
        n: int
        """
        actions = []
        
        # Flatten all actions in order of streets
        street_order = ['preflop', 'flop', 'turn', 'river']
        all_actions = []
        for street in street_order:
            if street in action_history:
                all_actions.extend(action_history[street])
        
        # Filter for actions by the other player
        for act in all_actions:
            if act.get('uuid') != player_uuid:
                action = act["action"].lower()
                if action == "bigblind" or action == "smallblind":
                    continue
                elif action == "call" and act["amount"] == 0:
                    action = "check"
                actions.append(action)
        
        # Return the last n, oldest to newest (left oldest, right newest)
        return actions[-n:] if n > 0 else []
    
    @staticmethod
    def get_action(pi, valid_actions):
        # [raise, call, check, fold]
        mask = np.zeros(4, dtype=int)
        for action in valid_actions:
            action = action["action"].lower()
            if action == "call":
                mask[1] = 1
                mask[2] = 1
            elif action == "raise":
                mask[0] = 1
            elif action == "fold":
                mask[3] = 1

        pi = pi * mask
        pi = pi / np.sum(pi)
        result = np.random.choice(np.arange(4, dtype=int), p=pi)
        if result == 0:
            return "raise"
        elif result == 1 or result == 2:
            return "call"
        else:
            return "fold"

        