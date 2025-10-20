from abc import ABC, abstractmethod


class Agent(ABC):
    def __init__(self, name, client, model):
        self.name = name
        self.client = client
        self.model = model

    @abstractmethod
    def run(self, **kwargs):
        ...
