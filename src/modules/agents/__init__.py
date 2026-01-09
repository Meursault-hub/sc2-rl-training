REGISTRY = {}

from .rnn_agent import RNNAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .rnn_agent_new import RNNAgent_New

REGISTRY["rnn"] = RNNAgent
REGISTRY["ff"] = FFAgent
REGISTRY["rnn_new"] = RNNAgent_New
REGISTRY["central_rnn"] = CentralRNNAgent
