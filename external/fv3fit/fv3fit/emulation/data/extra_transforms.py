


class DumpableTransform(abc.ABC):
    """
    Base class for transforms requiring some extra
    state information to operate on a dataset with operations
    to serialize to disk.
    """

    def __init__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def __call__(self, dataset: AnyDataset) -> OutputDataset:
        pass

    @abc.abstractmethod
    def dump(self, path: str):
        pass

    @abc.abstractclassmethod
    def load(self, path: str):
        pass


class Standardize(DumpableTransform):

    OUTPUT_FILE = "standardization_info.pkl"

    def __init__(self, std_info: StandardizeInfo):
        self.std_info = std_info

    def __call__(self, dataset: AnyDataset) -> AnyDataset:
        standardized = {}
        for varname in dataset:
            mean, std = self.std_info[varname]
            standardized[varname] = (dataset[varname] - mean) / std
        return standardized

    def dump(self, path: str):
        with open(os.path.join(path, self.OUTPUT_FILE), "wb") as f:
            pickle.dump(self.std_info, f)