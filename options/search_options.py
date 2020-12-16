from .base_options import BaseOptions


class SearchOptions(BaseOptions):

    def initialize(self, parser):
        self.phase = 0
        self.prefix = "search"
        parser = BaseOptions.initialize(self, parser)

        # Query image
        parser.add_argument('--query_filename', type=str, required=True, help='the query image to be computed')

        # How many images it should be trained on
        parser.add_argument('--n_images', type=int, default=5,
                            help='the number of images with closest MSE')

        return self.add_common_test_search(parser)
