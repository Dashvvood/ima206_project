import os
import motti
motti.append_parent_dir(__file__)
thisfile = os.path.basename(__file__).split(".")[0]
o_d = motti.o_d()

import argparse

argparse.ArgumentParser()

parser.add_argument(
    "--model", type=src, default="barlow_twins"
)
