import os
import random

num_train = 432
num_test = 144


root = "data"
source = "raw"
dest = "training"
root_relative_to_dest = ".."


def link_files(files, dest):
    os.makedirs(f"{root}/{dest}", exist_ok=True)
    for f in files:
        os.symlink(f"{root_relative_to_dest}/{source}/{f}", f"{root}/{dest}/{f}")


random.seed(0)
files = os.listdir(f"{root}/{source}/")
random.shuffle(files)

link_files(files[:num_train], "training")
link_files(files[num_train:][:num_test], "validation")
