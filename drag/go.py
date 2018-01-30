import itertools
import subprocess

import numpy as np


class ExperimentGroup():
    def __init__(self, name, var):
        self.name = name

        self.experiments = []

        self.ids = [key for key in var.keys() if len(var[key]) > 1]

        id_string = ""
        for i in self.ids:
            id_string += f"{{{i}}} "

        for values in itertools.product(*var.values()):
            args = dict(zip(var.keys(), values))
            self.experiments.append(Experiment(**args, id_format=id_string))


class Experiment():
    def __init__(self,
                 shape='circle',
                 theta=0,
                 r=92,
                 Re=100,
                 time=60,
                 Uin=0.1,
                 size=1 / 8,
                 id_format="{shape} {theta} {r} {Re} {time} {Uin} {size}"):
        self.shape = shape
        self.r = r
        self.Re = Re
        self.time = time
        self.theta = theta
        self.Uin = Uin
        self.size = size

        self.id = id_format.format(
            shape=self.shape,
            r=self.r,
            Re=self.Re,
            time=self.time,
            theta=self.theta,
            Uin=self.Uin,
            size=self.size)

    def generate_args(self):
        return [
            "./drag.py",
            "-mmrt",
            f"-r{self.r}",
            f"-R{self.Re}",
            f"-T{self.time}",
            f"-o{self.shape}",
            f"-t{self.theta}",
            f"-u{self.Uin}",
            f"-s{self.size}",
        ]


if __name__ == "__main__":
    experiments = [
        # ExperimentGroup('results/reynolds_circle.txt', {
        #     'shape': ['circle'],
        #     'Re': np.linspace(10, 1000, 100)
        # }),
        # ExperimentGroup('results/velocity.txt', {
        #     'shape': ['circle', 'rect'],
        #     'Uin': np.arange(1, 16) / 100,
        #     'time': [240],
        #     'Re': [??],
        # }),
        # ExperimentGroup('results/size.txt', {
        # 'shape': ['circle', 'rect', 'halfcircle', 'triangle', 'moon', 'bullet'],
        #     'size': [1/32, 1/24, 1/16, 1/12, 1/8, 1/6, 1/4, 1/3],
        #     'Re': [??],
        # }),
    ]

    print('{:10}{}'.format('name', 'progress'))
    for eg in experiments:
        print(eg.name)
        print('-' * len(eg.experiments))
        out_name = eg.name
        with open(out_name, 'a') as out:
            out.write(' '.join(eg.ids + ['Cd', 'Cl']) + '\n')
            out.flush()

        for e in eg.experiments:
            with open(eg.name, 'a') as out:
                out.write(e.id)
                out.flush()
                subprocess.call(e.generate_args(), stdout=out)
            print('|', end='', flush=True)
        print(' Done', flush=True)
