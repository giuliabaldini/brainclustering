from .base_options import BaseOptions


class TrainOptions(BaseOptions):

    def initialize(self, parser):
        self.phase = 1
        self.prefix = "train"
        parser = BaseOptions.initialize(self, parser)

        return parser
