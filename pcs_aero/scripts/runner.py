#!/usr/bin/env python3

import argparse
import subprocess

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Brute force experiment runner.')

    parser.add_argument('script', help='Script to use.')
    parser.add_argument('config', help='Config file to use')
    
    args = parser.parse_args()    

    program = args.script

    # This works for now...
    exec(open(args.config).read())


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
                subprocess.call([program] + e.generate_args(), stdout=out)
            print('|', end='', flush=True)
        print(' Done', flush=True)
