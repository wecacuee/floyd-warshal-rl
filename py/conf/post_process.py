from pathlib import Path
from cog.confutils import Conf, MultiConfGen, dict_update_recursive
from alg.qlearning import post_process as qlearning_post_process
from alg.floyd_warshall_grid import post_process as fw_post_process
from conf.default import CommonPlayConf


class CommonPostProcess(CommonPlayConf):
    @property
    def cellsize(self):
        return self.visualizer_conf.cellsize

    @property
    def image_file_fmt(self):
        Path(self.log_file_dir).mkdir(parents=True, exist_ok=True)
        return str(Path(self.log_file_dir) / self.image_file_fmt_basename)

    def defaults(self):
        return dict_update_recursive(
            super().defaults(),
            dict(func = lambda log_file: print("I did nothing"),
                 image_file_fmt_basename = "action_value_{episode}_{step}.png",
                 _log_file = None))
        
    
class QLearningPostProcess(CommonPostProcess):
    def defaults(self):
        defaults = super().defaults()
        defaults = dict_update_recursive(
            defaults,
            dict(func = qlearning_post_process,
                 _log_file = None))
        return defaults


class FloydWarshalPostProcess(CommonPostProcess):
    def defaults(self):
        defaults = super().defaults()
        defaults = dict_update_recursive(
            defaults,
            dict(func = fw_post_process))
        return defaults


MultiPostProcessConf = MultiConfGen(
    "MultiPostProcessConf",
    dict(fw = FloydWarshalPostProcess(), ql = QLearningPostProcess()))


if __name__ == '__main__':
    import sys
    conf = Conf.parse_all_args("conf.post_process:MultiPostProcessConf",
                               sys.argv[1:], glbls=globals())
    conf.apply_func()

    
