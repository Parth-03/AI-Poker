from pypokerengine.api.game import setup_config, start_poker
from raise_player import RaisedPlayer
from naiveplayer import NaivePlayer
from alphaplayer import AlphaPlayer
from randomplayer import RandomPlayer

#TODO:config the config as our wish
config = setup_config(max_round=10, initial_stack=1000, small_blind_amount=10)

config.register_player(name="80", algorithm=AlphaPlayer(model="naive_pretrained_model.pth.tar", dim=80))
config.register_player(name="Naive", algorithm=NaivePlayer())
# config.register_player(name="100", algorithm=AlphaPlayer(model="naive_pretrained_model_100.pth.tar", dim=100))


game_result = start_poker(config, verbose=1)
