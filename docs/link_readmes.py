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
            # with open(os.path.join(dest_dir, f"{readme_parent}_readme.rst"), "w") as f:
            #     readme_relative_path = os.path.relpath(os.path.join(root, "README.md"), dest_dir)
            # f.write(f".. _{readme_parent.replace('_', '-')}-readme:\n")
            # f.write(f".. mdinclude:: {readme_relative_path}")


link_readmes(os.path.abspath("../workflows"), build_dir)
link_readmes(os.path.abspath("../external"), build_dir)


# def find_readmes(workflow_root):
#     workflows = []
#     for root, dirs, files in os.walk(workflow_root):
#         for file in files:
#             if file == "README.md":
#                 if root == workflow_root:
#                     workflow = "readme"
#                 else:
#                     workflow = os.path.split(root)[-1]
#                 readme = os.path.relpath(os.path.join(root, file), build_dir)
#                 workflows.append((workflow, readme))
#     return workflows


# links = []
# for workflow, readme in workflows:
#     link = os.path.join(build_dir, f"{workflow}_link")
#     links.append(link)
#     with open(link + ".rst", "w") as f:
#         f.write(f".. mdinclude:: {readme}")
