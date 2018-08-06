import os
import sys
from conf.default import Conf

if __name__ == '__main__':
    conf = Conf.parse_all_args(
        "conf.default:FloydWarshallPlayConf", sys.argv[1:])
    if not os.path.exists(conf.log_file):
        raise ValueError("Need a better log file")
    conf.compute_metrics_from_replay.on_play_end()
