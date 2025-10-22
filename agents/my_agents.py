from abc import ABC, abstractmethod
import json
import logging
import sys


class Agent(ABC):

    def __init__(self, name, client, model):
        self.name = name
        self.client = client
        self.model = model
        self.__init_logger()

    @abstractmethod
    async def run(self, **kwargs):
        ...

    def log(self, msg: str, level: int = logging.DEBUG):
        self.logger.log(level=level, msg=msg)

    def __init_logger(self):
        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        logger_handler = logging.StreamHandler(sys.stdout)
        logger_handler.setLevel(logging.DEBUG)
        logger_formatter = logging.Formatter(
            '[%(asctime)s](%(levelname)s) %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        logger_handler.setFormatter(logger_formatter)
        self.logger.addHandler(logger_handler)
