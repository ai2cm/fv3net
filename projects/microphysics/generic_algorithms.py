from typing import Callable, List, Tuple, TypeVar

Model = TypeVar("Model")
Dataset = TypeVar("Dataset")

TrainFn = Callable[[int, Dataset], Model]
Generate = Callable[[int, List[Model], int], Dataset]


class TrainOnline:
    def __init__(self, train: TrainFn, generate: Generate) -> None:
        self.train = train
        self.generate = generate

    def __call__(
        self, duration: int, cycles: int, initial_model: Model
    ) -> Tuple[List[Model], List[Dataset]]:
        models = [initial_model]
        datasets = []
        for cycle in range(cycles):
            data = self.generate(cycle, models, duration)
            datasets.append(data)
            models.append(self.train(cycle, data))
        return models, datasets
