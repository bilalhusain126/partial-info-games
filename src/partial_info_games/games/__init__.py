"""The games, each with its exact solution where one is known."""

from .base import Game, TradingGame
from .two_traders import TwoTraders, LinearStrategies
from .markov_traders import MarkovTraders
from .broker_trader import BrokerTrader
from .major_minor import MajorMinor
