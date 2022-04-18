from typing import Type
import yaml
import dacite
import dataclasses
import dacite.types


def _make_repr(
    dumper: Type[yaml.SafeDumper],
    cls,
    tag=yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
):
    names = [field.name for field in dataclasses.fields(cls)]

    def _represent(dumper: yaml.SafeDumper, data: cls):
        mapping = {}
        for name in names:
            mapping[name] = getattr(data, name)

        return dumper.represent_mapping(tag, mapping)

    dumper.add_representer(cls, _represent)


class Visitor:
    def visit_dataclass(self, cls):
        if cls in self.visited:
            return

        _make_repr(self.dumper, cls)
        self.visited.add(cls)

    def visit_leaf(self, cls):
        pass

    def visit_union(self, cls):
        pass


class DumperVisitor(Visitor):
    def __init__(self, dumper):
        self.visited = set()
        self.dumper = dumper

    def visit_dataclass(self, node: "DataclassNode"):
        if node.cls in self.visited:
            return

        tag = (
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG
            if node.name
            else "!" + node.cls.__name__
        )

        _make_repr(
            self.dumper,
            node.cls,
            tag,
        )
        self.visited.add(node.cls)


class LoaderVisitor(Visitor):
    def __init__(self, loader: Type[yaml.SafeLoader]):
        self.visited = set()
        self.loader = loader

    def visit_dataclass(self, node):
        cls = node.cls
        if cls in self.visited:
            return

        def _constructor(loader: yaml.SafeLoader, node):
            mapping = loader.construct_mapping(node)
            return cls(**mapping)

        if not node.name:
            tag = "!" + cls.__name__
            self.loader.add_constructor(tag, _constructor)
            self.visited.add(cls)


def to_yaml(obj):
    dumper = yaml.SafeDumper
    cls = obj.__class__
    type_graph = build_type_tree("top", cls)
    visitor = DumperVisitor(dumper)
    for node in walk_post_order(type_graph):
        node.accept(visitor)
    return yaml.dump(obj, Dumper=dumper)


def from_yaml(cls, yaml_str):
    loader = yaml.SafeLoader
    visitor = LoaderVisitor(loader)
    type_graph = build_type_tree("top", cls)
    for node in walk_post_order(type_graph):
        node.accept(visitor)

    return dacite.from_dict(cls, yaml.load(yaml_str, Loader=loader))


def is_union(cls: Type) -> bool:
    return dacite.types.is_union(cls)


class Node:
    def __init__(self, name, cls, children):
        self.name = name
        self.cls = cls
        self.children = children


class UnionNode(Node):
    def accept(self, visitor):
        visitor.visit_union(self)


class GenericNode(Node):
    def accept(self, visitor):
        visitor.visit_generic(self)


class DataclassNode(Node):
    def accept(self, visitor):
        visitor.visit_dataclass(self)


class OtherNode(Node):
    def accept(self, visitor):
        visitor.visit_leaf(self)


def walk_post_order(node: Node):
    stack = [(True, node)]
    while stack:
        first_visit, node = stack.pop()

        if first_visit:
            stack.append((False, node))
            for kids in node.children:
                stack.append((first_visit, kids))
        else:
            yield node


def build_type_tree(name, cls):
    if dataclasses.is_dataclass(cls):
        return DataclassNode(
            name,
            cls,
            children=[
                build_type_tree(field.name, field.type)
                for field in dataclasses.fields(cls)
            ],
        )
    elif is_union(cls):
        return UnionNode(
            name,
            cls,
            [
                build_type_tree("", child)
                for k, child in enumerate(dacite.types.extract_generic(cls))
            ],
        )
    elif dacite.types.is_generic(cls):
        children = list(dacite.types.extract_generic(cls))
        return GenericNode(
            name, cls, [build_type_tree("", child) for child in children]
        )
    else:
        return OtherNode(name, cls, [])
