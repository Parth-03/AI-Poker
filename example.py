from pypokerengine.api.game import setup_config, start_poker
from raise_player import RaisedPlayer
from testplayer import TestPlayer
from alphaplayer import AlphaTrainPlayer

#TODO:config the config as our wish
config = setup_config(max_round=5, initial_stack=1000, small_blind_amount=10)



config.register_player(name="f1", algorithm=RaisedPlayer())
config.register_player(name="FT2", algorithm=AlphaTrainPlayer())


game_result = start_poker(config, verbose=1)
