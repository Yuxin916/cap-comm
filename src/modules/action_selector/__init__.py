from .EpsilonGreedyActionSelector import EpsilonGreedyActionSelector
from .MultinomialActionSelector import MultinomialActionSelector
from .SoftPoliciesSelector import SoftPoliciesSelector

REGISTRY = {}

REGISTRY["multinomial"] = MultinomialActionSelector

REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

REGISTRY["soft_policies"] = SoftPoliciesSelector
