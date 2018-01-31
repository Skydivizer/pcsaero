import argparse

import code.models as models
import code.obstacles as obstacles

# To not overflow the help interface of arugment parser only the main obstacle
# names are # shown as possible options
onames = [l[0] for l in obstacles.names_lists]
onames_map = {n: l[0] for l in obstacles.names_lists for n in l}

# Defines the type for argument parser
def onames_type(string):
    string = str.lower(string)
    try:
        return onames_map[string]
    except KeyError:
        raise TypeError(string, 'not a known obstacle name')


class ModelArgParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(ModelArgParser, self).__init__(*args, **kwargs)
        ### Uncomment your own risk
        # self.add_argument(
        #     "-m",
        #     "--model",
        #     help=
        #     "The model to use. All models use lattice boltzmann methods, the "
        #     "difference lies in the implementation. [srt] single relaxation time. "
        #     "[mrt] multiple relaxation times [pylbm] mrt model that wraps pyLBM",
        #     type=str.lower,
        #     choices=[
        #         'srt',
        #         'mrt',
        #         'pylbm'],
        #     default='mrt')

        self.add_argument(
            "-r",
            "--resolution",
            help='Number of cells per 1 characteristic length',
            type=int,
            default=256)

        self.add_argument(
            "-R",
            "--reynolds",
            help='Reynolds number',
            type=float,
            default=220)

        self.add_argument(
            "-o",
            "--obstacle",
            help='The obstacle to calculate the drag coefficient of.',
            type=onames_type,
            choices=onames,
            default=onames[0])

        self.add_argument(
            "-t",
            "--theta",
            type=float,
            help="Angle of approach in degrees",
            default=0)

        self.add_argument(
            '-u',
            '--stream_velocity',
            type=float,
            help="Base stream lattice velocity, reasonable values are in the "
            "interval [0.01, 0.15]",
            default=0.1)

        self.add_argument(
            '-s',
            '--size',
            type=float,
            help='Reference size (height) of obstacle in characteristic units.',
            default=1 / 8)



        self.add_argument(
            '-T',
            '--time',
            type=float,
            help="Minimum time the simulation has to be ran.",
            default=0
            )

        self.add_argument(
            '-l',
            '--load',
            type=str,
            help='Load model from file using pickle. This overrides all other '
            'settings.'
        )

    def parse_args(self, *args, **kwargs):
        args = super(ModelArgParser, self).parse_args(*args, **kwargs)

        if args.load:
            import pickle
            with open(args.load, 'rb') as f:
                model = pickle.load(f)

        else:
            # Create the model
            model = {
                "srt": models.SRT,
                "mrt": models.MRT,
                "pylbm": models.PyLBM,
            }['mrt'](
            # }[args.model](
                Re=args.reynolds,
                resolution=args.resolution,
                obstacle=args.obstacle,
                theta=args.theta,
                Uin=args.stream_velocity,
                size=args.size)

        while model.time < args.time:
            model.step()

        return args, model