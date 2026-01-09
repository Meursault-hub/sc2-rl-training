REGISTRY = {}

from .basic_controller import BasicMAC
from .basic_controller_policy import BasicMAC as PolicyMAC
from .central_basic_controller import CentralBasicMAC
from .hierarchical_controller import HierarchicalMAC

REGISTRY["basic_mac"] = BasicMAC
REGISTRY["policy"] = PolicyMAC
REGISTRY["basic_central_mac"] = CentralBasicMAC
REGISTRY["hierarchical_mac"] = HierarchicalMAC