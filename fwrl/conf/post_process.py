from pathlib import Path
from umcog.confutils import Conf, MultiConfGen, dict_update_recursive
from ..alg.qlearning import post_process as qlearning_post_process
from ..alg.floyd_warshall_grid import post_process as fw_post_process
from .default import CommonPlayConf

class QLearningPostProcess(CommonPlayConf):
    def defaults(self):
        defaults = dict_update_recursive(
            super().defaults(),
            dict(_call_func = qlearning_post_process))
        return defaults


class FloydWarshalPostProcess(CommonPlayConf):
    def defaults(self):
        defaults = dict_update_recursive(
            super().defaults(),
            dict(_call_func = fw_post_process))
        return defaults

if __name__ == '__main__':
    import sys
    conf = Conf.parse_all_args("conf.post_process:FloydWarshalPostProcess",
                               sys.argv[1:], glbls=globals())
    conf()

    
