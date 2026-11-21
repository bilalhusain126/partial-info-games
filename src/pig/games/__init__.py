"""The games, each with its exact or reference solution."""

from .two_traders import TwoTraders, exact_lam0 as two_traders_exact_lam0
from .linear_strategies import LinearStrategies
from .markov_traders import MarkovTraders, exact_lam0 as markov_traders_exact_lam0, linear_equilibrium
from .broker_trader import BrokerTrader, exact as broker_trader_exact
from .major_minor import MajorMinor, exact as major_minor_exact
