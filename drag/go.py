import collections
import subprocess

Experiment = collections.namedtuple('Experiment', ['shape', 'thetas'])
Setting = collections.namedtuple('Setting',
                                 ['resolution', 'reynolds', 'time', 'repeats'])

experiments = [
    # Experiment('circle', [0]),
    # Experiment('rect', range(0, 45 + 1, 5)),
    # Experiment('halfcircle', range(0, 180 + 1, 5)),
    # Experiment('triangle', range(0, 180 + 1, 5)),
    Experiment('moon', range(0, 180 + 1, 5)),
    Experiment('bullet', range(0, 180 + 1, 5)),
]

settings = [
    Setting(92, 98, 60, 1),
    Setting(92, 99, 60, 1),
    Setting(92, 100, 60, 1),
    Setting(92, 101, 60, 1),
    Setting(92, 102, 60, 1),
]

for res, rey, time, repeats in settings:
    out_name = f'r{res}R{rey}T{time}.txt'

    for shape, thetas in experiments:
        print(shape)
        for theta in thetas:
            print(theta)
            for i in range(repeats):
                with open(out_name, 'a') as out:
                    out.write(f"{shape} {theta} ")
                    out.flush()
                    subprocess.call(
                        [
                            "./drag.py", "-mmrt", f"-r{res}", f"-R{rey}",
                            f"-T{time}", f"-o{shape}", f"-t{theta}"
                        ],
                        stdout=out)
