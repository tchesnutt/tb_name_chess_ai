import tensorflow as tf


class Train:
    def __init__(self, model, data, config):
        self.model = model
        self.data = data
        self.config = config

    def train(self):
        