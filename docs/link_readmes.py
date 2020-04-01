import os

workflow_base_rst = "workflow_base.rst"
build_dir = "readme_links"


def write_toc_tree(links, f):
    print(".. toctree::", file=f)
    print("    :maxdepth: 2", file=f)
    print("", file=f)
    for link in links:
        print(f"    {link}.rst", file=f)


workflows = []
for root, dirs, files in os.walk(".."):
    for file in files:
        if file == "README.md":
            if root == "..":
                workflow = "readme"
            else:
                workflow = os.path.split(root)[-1]
            workflows.append((workflow, os.path.join(root, file)))


links = []
for workflow, readme in workflows:
    link = os.path.join(build_dir, f"{workflow}_link")
    links.append(link)
    with open(link + ".rst", "w") as f:
        f.write(f".. mdinclude:: {readme}")

