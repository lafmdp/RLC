import atexit
import json
import os
import pprint
import sys
from datetime import datetime
from os.path import join
from typing import Any, Dict, List

from loguru import _defaults
from loguru._logger import Core as _Core
from loguru._logger import Logger as _Logger
from tensorboardX import SummaryWriter


class IntegratedLogger(_Logger, SummaryWriter):
    def __init__(
        self, record_param: List[str] = None, log_root: str = "logs", args: Dict = None
    ):
        """
        :param record_param: Used for name the experiment results dir
        :param log_root: The root path for all logs
        """
        _Logger.__init__(
            self,
            core=_Core(),
            exception=None,
            depth=0,
            record=False,
            lazy=False,
            colors=False,
            raw=False,
            capture=True,
            patcher=None,
            extra={},
        )

        if _defaults.LOGURU_AUTOINIT and sys.stderr:
            self.add(sys.stderr)

        atexit.register(self.remove)

        self.log_root = log_root
        self.args = args
        self.record_param_dict = self._parse_record_param(record_param)

        self._create_print_logger()
        self._save_args()

        SummaryWriter.__init__(self, logdir=self.exp_dir)

    def _create_print_logger(self):
        self.exp_dir = join(self.log_root, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if self.record_param_dict is not None:
            for key, value in self.record_param_dict.items():
                self.exp_dir = self.exp_dir + f"&{key}={value}"
        self.add(join(self.exp_dir, "log.log"), format="{time} -- {level} -- {message}")


    def _parse_record_param(self, record_param: List[str]) -> Dict[str, Any]:
        if self.args is None or record_param is None:
            return None
        else:
            record_param_dict = dict()
            for param in record_param:
                param = param.split(".")
                value = self.args
                for p in param:
                    value = value[p]
                record_param_dict["-".join(param)] = value
            return record_param_dict

    def _save_args(self):
        if self.args is None:
            return
        else:
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(self.args)
            with open(join(self.exp_dir, "parameter.json"), "w") as f:
                jd = json.dumps(self.args, indent=4)
                print(jd, file=f)

    def save_data(self, data, data_file_name):
        with open(join(self.exp_dir, data_file_name+".json"), "w") as f:
            jd = json.dumps(data)
            print(jd, file=f)

    def add_dict(self, info: Dict[str, float], t: int):
        for key, value in info.items():
            self.add_scalar(key, value, t)