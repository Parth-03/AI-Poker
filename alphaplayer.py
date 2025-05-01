import os, sys
sys.path.append(os.environ["BASE_DIR"] + "/alpha-zero-general")

from pypokerengine.players import BasePokerPlayer
from MCTS import MCTS
from poker.pytorch.NNet import NNetWrapper as nn
from poker.PokerGame import PokerGame
from poker.PokerLogic import PokerBoard, PokerState
from utils import dotdict
import os
import numpy as np
from pypokerengine.engine.table import Table
from pypokerengine.utils.game_state_utils import restore_game_state
from pypokerengine.utils.card_utils import gen_deck
from pypokerengine.utils.card_utils import estimate_hole_card_win_rate, gen_cards
from poker.boardtexture import classify_board_texture, board_texture_to_onehot # Added import

class AlphaPlayer(BasePokerPlayer):

  def __init__(self, model=None, dim=None, **kwargs):
    super().__init__(**kwargs)
    self.hole_card = None
    self.opponent_hole_card = None
    self.num_raises_this_street = 0
    self.dim = dim
    self.model = model
  
  def declare_action(self, valid_actions, hole_card, round_state):
    hole_card = [str(card) for card in hole_card]
    self.hole_card = hole_card
    
    for seat in round_state['seats']:
        if seat['uuid'] == self.uuid:
            my_stack = seat['stack']
        else:
            opponent_stack = seat['stack']
            other_uuid = seat['uuid']
    
    players_info = {
        "player1": {"stack": my_stack, "name": "Alice"},
        "player2": {"stack": opponent_stack, "name": "Bob"}
    }
    
    board = PokerBoard(players_info)
    board.emulator_state = restore_game_state(round_state)
    board.seed = 64
    # Add my hole card and random opponent cards for MCTS simulations
    for player in board.emulator_state["table"].seats.players:
        if player.uuid == self.uuid:
            player.add_holecard(gen_cards(hole_card))
            player.uuid = "player1"
            # set action histories to match uuids
        else:
            player.add_holecard(gen_cards(self.opponent_hole_card))
            player.uuid = "player2"
    
    for player in board.player_states:
        # Set state for each
        cur_player = board.player_states[player]
        # Set round count
        cur_player.set_round_count(round_state["round_count"])
        # Set pot
        cur_player.set_pot(round_state["pot"]["main"]["amount"])
        # Set street
        streets = ["preflop", "flop", "turn", "river", "showdown"]
        cur_player.set_street(streets.index(round_state["street"].lower()))
        # Set pot odds
        if board.emulator_state["street"] < 2:
            price = 20
        else:
            price = 40
        cur_player.set_pot_odds(round_state["pot"]["main"]["amount"], price)
        # Set hole cards and ehs
        if player == "player1":
            cur_player.set_hole(hole_card)
            cur_player["EHS"] = estimate_hole_card_win_rate(50, 2, gen_cards(hole_card), gen_cards(round_state["community_card"]))
            cur_player.set_stack(my_stack, opponent_stack)
        else:
            cur_player.set_hole(self.opponent_hole_card)
            cur_player["EHS"] = estimate_hole_card_win_rate(50, 2, gen_cards(self.opponent_hole_card), gen_cards(round_state["community_card"]))
            cur_player.set_stack(opponent_stack, my_stack)

        # Set community cards
        cur_player.set_community(round_state["community_card"])
        cur_player.set_num_raises_this_street(self.num_raises_this_street)

        # Set board texture
        if len(round_state["community_card"]) >= 3:
            texture_dict = classify_board_texture(board.emulator_state["street"], (), round_state["community_card"])
            texture_onehot = board_texture_to_onehot(texture_dict)
            cur_player.set_board_texture(texture_onehot)

    board = self.game.getCanonicalForm(board, self.currPlayer)
    action = np.argmax(self.mcts.getActionProb(board, temp=0))
    # pi, v = self.nnet.predict(board)
    # action = np.argmax(pi)
    # actions = ["raise", "call", "check", "fold"]
    if action == 0:
       return "raise"
    elif action == 1 or action == 2:
       return "call"
    else:
       return "fold"
        

  def receive_game_start_message(self, game_info):
    args = dotdict({
        'lr': 0.0005,
        'dropout': 0.3,
        'epochs': 10,
        'batch_size': 64,
        'cuda': False,
        "use_wandb": False, # Set to True to use wandb
        "dim": self.dim
        # 'run_name': 'poker_run_' + str(int(time.time())) # Remove run_name, wandb initialized externally
    })
    for player in game_info["seats"]:
        if player['uuid'] != self.uuid:
           other_uuid = player['uuid']
    self.game = PokerGame(max_round=10, uuids=(self.uuid, other_uuid))
    nnet = nn(self.game, a=args)
    nnet.load_checkpoint("/Users/samonuallain/AI-Poker/alpha-zero-general/pretrained_data", self.model)
    self.nnet = nnet
    
    args = dotdict({'numMCTSSims': 10, 'cpuct':1.0})
    self.mcts = MCTS(self.game, nnet, args)

  def receive_round_start_message(self, round_count, hole_card, seats):
        deck = gen_deck(exclude_cards=hole_card)
        deck.shuffle()
        self.opponent_hole_card = deck.draw_cards(2)
        self.opponent_hole_card = [str(card) for card in self.opponent_hole_card]

        if seats[1]["uuid"] == self.uuid:
            self.currPlayer = 1
        else:
           self.currPlayer = -1

  def receive_street_start_message(self, street, round_state):
    if self.opponent_hole_card[0] in round_state["community_card"] or self.opponent_hole_card[1] in round_state["community_card"]:
        deck = gen_deck(exclude_cards= self.hole_card + round_state['community_card'])
        deck.shuffle()
        self.opponent_hole_cards = deck.draw_cards(2)
        self.opponent_hole_card = [str(card) for card in self.opponent_hole_cards]
    self.num_raises_this_street = 0

  def receive_game_update_message(self, action, round_state):
    if list(round_state["action_histories"].values())[-1][-1]["action"] == "RAISE":
        self.num_raises_this_street += 1

  def receive_round_result_message(self, winners, hand_info, round_state):
    pass
