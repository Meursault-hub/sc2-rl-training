from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env, SC2TacticsEnv
from smac.env.sc2_tactics.sc2_tactics_env import SC2TacticsEnv_NEW
from smac.env.sc2_tactics.star36env_sdjx import SC2TacticsSDJXEnv
from smac.env.sc2_tactics.star36env_dhls import SC2TacticsDHLSEnv
from smac.env.sc2_tactics.star36env_wzsy import SC2TacticsWZSYEnv
from smac.env.sc2_tactics.star36env_wwjz import SC2TacticsWWJZEnv
from smac.env.sc2_tactics.star36env_adcc import SC2TacticsADCCEnv
from smac.env.sc2_tactics.star36env_tlhz import SC2TacticsTLHZEnv
from smac.env.sc2_tactics.star36env_yqgz import SC2TacticsYQGZEnv
from smac.env.sc2_tactics.star36env_jctq import SC2TacticsJCTQEnv
from smac.env.sc2_tactics.star36env_swct import SC2TacticsSWCTEnv
from smac.env.sc2_tactics.star36env_jdsr import SC2TacticsJDSREnv
from smac.env.sc2_tactics.star36env_fkwz import SC2TacticsFKWZEnv
from smac.env.sc2_tactics.star36env_gmzz import SC2TacticsGMZZEnv
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

def env_te(env, **kwargs) -> MultiAgentEnv:
    # === 【新增】 拦截并移除 render 参数 ===
    # 环境类的构造函数不认识 'render'，所以在这里移除它。
    # 这只会影响当前函数的 kwargs，不会影响 episode_runner 里读取配置。
    kwargs.pop("render", None) 
    # ====================================
    map_name = kwargs["map_name"]
    if map_name.startswith("sdjx"):
        return SC2TacticsSDJXEnv(**kwargs)
    elif map_name.startswith("dhls"):
        return SC2TacticsDHLSEnv(**kwargs)
    elif map_name.startswith("wzsy"):
        return SC2TacticsWZSYEnv(**kwargs)
    elif map_name.startswith("wwjz"):
        return SC2TacticsWWJZEnv(**kwargs)
    elif map_name.startswith("adcc"):
        return SC2TacticsADCCEnv(**kwargs)
    elif map_name.startswith("tlhz"):
        return SC2TacticsTLHZEnv(**kwargs)
    elif map_name.startswith("yqgz"):
        return SC2TacticsYQGZEnv(**kwargs)
    elif map_name.startswith("jctq"):
        return SC2TacticsJCTQEnv(**kwargs)
    elif map_name.startswith("swct"):
        return SC2TacticsSWCTEnv(**kwargs)
    elif map_name.startswith("jdsr"):
        return SC2TacticsJDSREnv(**kwargs)
    elif map_name.startswith("fkwz"):
        return SC2TacticsFKWZEnv(**kwargs)
    elif map_name.startswith("gmzz"):
        return SC2TacticsGMZZEnv(**kwargs)
    else:
        return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["sc2_tactics"] = partial(env_te, env=SC2TacticsEnv)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
