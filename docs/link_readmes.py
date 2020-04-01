import os


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
    link = f"{workflow}_link"
    links.append(link)
    with open(link + ".rst", "w") as f:
        f.write(f".. mdinclude:: {readme}")


print(".. toctree::")
print("    :maxdepth: 2")
print("")
for link in links:
    print(f"    {link}")
