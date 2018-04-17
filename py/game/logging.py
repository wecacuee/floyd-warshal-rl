import os
import time
from pathlib import Path
import json

import logging
import logging.config

from cog.misc import DictEncoder, NumpyEncoder, ChainedEncoders
from cog.confutils import extended_kwprop, KWProp as prop, xargs
from cog.memoize import MEMOIZE_METHOD
from cog.misc import ensuredirs, git_revision


class NPJSONEncDec(object):
    def dumps(self, dct):
        return DictEncoder(default=NumpyEncoder().default).encode(dct)

    def loads(self, str_):
        return json.loads(
            str_,
            object_hook = ChainedEncoders(
                encoders = [NumpyEncoder(), DictEncoder()]).loads_hook)

class JSONLoggingFormatter(logging.Formatter):
    def __init__(self, enc, sep = "\t", **kwargs) :
        self.enc  = enc
        self.sep  = sep
        super().__init__(**kwargs)

    def format(self, record):
        str_ = super().format(record)
        record_str = self.enc.dumps(getattr(record, "data", {}))
        str_ = self.sep.join((str_, getattr(record, "tag", ""),
                              str(len(record_str)), record_str))
        return str_



def logging_dictConfig(log_file, logging_encdec):
    return dict(
        version = 1,
        formatters = dict(
            json_fmt = {
                '()' : JSONLoggingFormatter,
                'enc' : logging_encdec,
                'sep' : "\t",
                'format' : "%(asctime)s %(name)-15s %(message)s",
                'datefmt' : "%d %H:%M:%S"
            }
        ),
        handlers = dict(
            file = {
                'class' : "logging.FileHandler",
                'filename' : log_file,
                'formatter' : "json_fmt",
                'level' : "DEBUG",
            },
            console = {
                'class' : "logging.StreamHandler",
                'level' : "INFO"
            }
        ),
        root = dict(
            level = 'DEBUG',
            handlers = "console file".split()
        )
    )


def find_latest_file(dir_):
    if not Path(dir_).exists():
        return None
    p_stats = [(p, p.stat()) for p in Path(dir_).iterdir() if p.is_file()]
    return max(p_stats, key = lambda p_stat: p_stat[1].st_mtime)[0]


def setLoggerConfig(confname, log_file, logging_encdec):
    print("Setting dict config from {confname}".format(confname=confname))
    logging.root = logging.RootLogger(logging.WARNING)
    logging.Logger.root = logging.root
    logging.Logger.manager = logging.Manager(logging.Logger.root)
    logging.config.dictConfig(
        logging_dictConfig(log_file,
                           logging_encdec))


def logger_factory(s):
    _ = s.set_logger_conf
    return (lambda name : logging.getLogger(name))
    

def run_full_time(self):
    return time.strftime(self.run_full_time_format)
    
@extended_kwprop
def LogFileConf(
        log_file_conf         = prop(lambda self: self),
        log_file              = prop(lambda self: ensuredirs(self.log_file_template.format(self=self))),
        log_file_dir          = prop(lambda self: ensuredirs(self.log_file_dir_template.format(self=self))),
        data_dir              = prop(lambda self: os.environ["MID_DIR"]), exp_name     = prop(lambda self: self.exp_name_template.format(self=self)),
        gitrev                = prop(lambda self: git_revision(Path(__file__).parent)),
        run_month             = prop(lambda self: self.run_full_time[:6]),
        run_time              = prop(lambda self: self.run_full_time[6:]),
        run_full_time         = prop(MEMOIZE_METHOD(run_full_time)),
        set_logger_conf       = prop(lambda s: setLoggerConfig(s.confname, s.log_file, s.logging_encdec)),
        logger_factory        = prop(MEMOIZE_METHOD(logger_factory)),
        logging_encdec        = NPJSONEncDec(),
        exp_name_template     = "{self.run_month}_{self.gitrev}_{self.confname}",
        log_file_dir_template = "{self.data_dir}/{self.project_name}/{self.exp_name}",
        log_file_template     = "{self.log_file_dir}/{self.run_time}.log",
        run_full_time_format  = "%Y%m%d-%H%M%S",
    ):
    return log_file_conf
