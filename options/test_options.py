from .base_options import BaseOptions


class TestOptions(BaseOptions):

    def initialize(self, parser):
        self.phase = 2
        self.prefix = "test"
        parser = BaseOptions.initialize(self, parser)
        # Query image
        parser.add_argument('--query_filename', type=str, default=None,
                            help='the optional query image to be computed instead of computing an entire folder')

        parser.add_argument('--model_phase', type=str, default="train",
                            help='decide which model you want to load, the models are always saved in *model_phase*_model')

        parser.add_argument('--model_index', type=int, default=-1,
                            help='decide which model to take for testing. Default is the last computed model. '
                                 'Allowed values are from 0 to training images - 1.')

        return self.add_common_test_search(parser)
