from pypokerengine.players import BasePokerPlayer
import random as rand
import pprint
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate
from pypokerengine.utils.card_utils import gen_cards

import coloredlogs
import logging

log = logging.getLogger(__name__)

coloredlogs.install(level='INFO')  # Change this to DEBUG to see more info.

def get_last_n_actions(action_history, player_uuid, n):
        """
        Returns the last n actions (as strings) taken by player_uuid, ordered oldest to newest.
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
        
        # Filter for actions by this player
        for act in all_actions:
            if act.get('uuid') == player_uuid:
                action = act["action"].lower()
                if action == "bigblind" or action == "smallblind":
                    continue
                elif action == "call" and act["amount"] == 0:
                    action = "check"
                actions.append(action)
        
        # Return the last n, oldest to newest (left oldest, right newest)
        return actions[-n:] if n > 0 else []


# test player so I can see what's happening
class TestPlayer(BasePokerPlayer):

  def __init__(self):
    super().__init__()
    self.opponent_uuid = None
  
  
  def declare_action(self, valid_actions, hole_card, round_state):
    # log.info("declare_action called\n")
    # log.info("Valid actions: %s", valid_actions)
    # log.info("Hole card: %s", hole_card)
    # log.info("Round state: %s", round_state)
    log.info("OPPONENT LAST N ACTIONS:" + str(get_last_n_actions(round_state["action_histories"], self.opponent_uuid, 5)))

    return valid_actions[1]["action"]
    

  def receive_game_start_message(self, game_info):
    log.info("Game start message received\n")
    log.info("Game info: %s", game_info)
    for seat in game_info["seats"]:
      if seat["uuid"] != self.uuid:
         self.opponent_uuid = seat["uuid"]

  def receive_round_start_message(self, round_count, hole_card, seats):
    # log.info("Round start message received\n")
    # log.info("Round count: %s", round_count)
    # log.info("Hole card: %s", hole_card)
    # log.info("Seats: %s", seats)
    pass


  def receive_street_start_message(self, street, round_state):
    log.info("Street start message received\n")
    log.info("Street: %s", street)
    log.info("Round state: %s", round_state)
    pass

  def receive_game_update_message(self, action, round_state):
    log.info("Game update message received\n")
    log.info("Action: %s", action)
    log.info("Round state: %s", round_state)

  def receive_round_result_message(self, winners, hand_info, round_state):
    log.info("Round result message received\n")
    log.info("Winners: %s", winners)
    log.info("Hand info: %s", hand_info)
    log.info("Round state: %s", round_state)
