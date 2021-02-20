import os
import shutil

build_dir = "readme_links"


def link_readmes(root_dir, dest_dir):
    for root, dirs, files in os.walk(root_dir):
        if "README.md" in files:
            readme_parent = os.path.split(root)[-1]
            shutil.copyfile(
                os.path.join(root, "README.md"),
                os.path.join(dest_dir, f"{readme_parent}_readme.md"),
            )


link_readmes(os.path.abspath("../workflows"), build_dir)
link_readmes(os.path.abspath("../external"), build_dir)
