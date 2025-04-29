
from pypokerengine.players import BasePokerPlayer
from pypokerengine.utils.card_utils import gen_cards, estimate_hole_card_win_rate

class CustomAgent(BasePokerPlayer):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def evaluate_state(self, state_features):
        return sum(w * f for w, f in zip(self.weights, state_features))

    def declare_action(self, valid_actions, hole_card, round_state):
        features = extract_features(hole_card, round_state)
        score = self.evaluate_state(features)
        if score > 0.5:
            return "raise" if any(a['action'] == 'raise' for a in valid_actions) else "call"
        elif score > 0:
            return "call"
        else:
            return "fold"

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass

def extract_features(hole_card, round_state):
    strength = estimate_hand_strength(hole_card, round_state.get('community_card', []))
    pot_amount = round_state['pot']['main']['amount']
    my_stack = round_state['seats'][round_state['next_player']]['stack']
    to_call = pot_amount
    pot_odds = pot_amount / (pot_amount + to_call + 1e-6)
    stack_ratio = my_stack / 1000
    board_texture = calculate_board_texture(round_state.get('community_card', []))
    street_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
    street_number = street_mapping.get(round_state['street'], 0)
    actions = round_state['action_histories'].get(round_state['street'], [])
    num_raises = sum(1 for action in actions if action['action'] == 'raise')
    opponent_aggression = calculate_opponent_aggression(round_state)
    payout = strength * pot_amount

    return [
        strength,
        pot_odds,
        stack_ratio,
        board_texture,
        street_number,
        num_raises,
        opponent_aggression,
        payout
    ]

def estimate_hand_strength(hole_card, community_card):
    nb_simulation = 100
    nb_player = 2
    win_rate = estimate_hole_card_win_rate(nb_simulation, nb_player, gen_cards(hole_card), gen_cards(community_card))
    return win_rate

def calculate_board_texture(community_card):
    suits = [card[-1] for card in community_card]
    suit_counts = {s: suits.count(s) for s in set(suits)}
    max_suit_count = max(suit_counts.values()) if suit_counts else 0
    return max_suit_count / 5

def calculate_opponent_aggression(round_state):
    aggression = 0
    for street, actions in round_state.get('action_histories', {}).items():
        for action in actions:
            if action['action'] == 'raise':
                aggression += 1
    total_actions = sum(len(actions) for actions in round_state.get('action_histories', {}).values())
    return aggression / (total_actions + 1e-6)
