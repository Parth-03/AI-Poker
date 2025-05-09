from pypokerengine.api.game import setup_config, start_poker
from raise_player import RaisedPlayer
from naiveplayer import NaivePlayer
from alphaplayer import AlphaPlayer
from randomplayer import RandomPlayer
from tqdm import tqdm

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)

# config.register_player(name="AlphaPoker10sims", algorithm=AlphaPlayer(model="alpha-zero-general/temp-lr_decay-board_texture-80-redo/best.pth.tar", dim=80, numSims=10))
# config.register_player(name="AlphaPoker0sims", algorithm=AlphaPlayer(model="alpha-zero-general/temp-lr_decay-board_texture-80-redo/best.pth.tar", dim=80, numSims=0))
config.register_player(name="AlphaPoker", algorithm=AlphaPlayer(model="alpha-zero-general/pretrained_data/both_ofem.pth.tar", dim=100, numSims=0))
# config.register_player(name="AlphaNaive", algorithm=AlphaPlayer(model="alpha-zero-general/pretrained_data/pretrain_data_50round_500games_naivplayer100_8epoch.pth.tar", dim=100, numSims=0))





# config.register_player(name="AlphaPoker80", algorithm=AlphaPlayer(model="temp-lr_decay-board_texture-80-redo2.0/best.pth.tar", dim=80))
config.register_player(name="Naive", algorithm=NaivePlayer())
# config.register_player(name="Random", algorithm=RandomPlayer())
# config.register_player(name="RaisedPlayer", algorithm=RaisedPlayer())
# config.register_player(name="100", algorithm=AlphaPlayer(model="naive_pretrained_model_100.pth.tar", dim=100))

num_games = 10
winner_count = {}
for i in tqdm(range(num_games)):
    game_result = start_poker(config, verbose=False)
    winner = None
    max_stack = 0
    for player in game_result['players']:
        if player['stack'] > max_stack:
            max_stack = player['stack']
            winner = player['name']
    if winner not in winner_count:
        winner_count[winner] = 0
    winner_count[winner] += 1
    print(f"Winner: {winner}, Stack: {max_stack}")

print("Winner count:", winner_count)
