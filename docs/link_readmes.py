import os

build_dir = "readme_links"
cwd = os.getcwd()
workflow_root = os.path.abspath("../workflows")


workflows = []
for root, dirs, files in os.walk(workflow_root):
    for file in files:
        if file == "README.md":
            if root == workflow_root:
                workflow = "readme"
            else:
                workflow = os.path.split(root)[-1]
            readme = os.path.relpath(os.path.join(root, file), build_dir)
            workflows.append((workflow, readme))


links = []
for workflow, readme in workflows:
    link = os.path.join(build_dir, f"{workflow}_link")
    links.append(link)
    with open(link + ".rst", "w") as f:
        f.write(f".. mdinclude:: {readme}")
