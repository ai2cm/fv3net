import abc
from typing import Hashable, Iterable, Optional
from fv3fit._shared.predictor import Predictor
from fv3fit._shared.taper_function import MaskTaperFunction, TaperFunction
import xarray as xr


class NoveltyDetector(Predictor, abc.ABC):
    """
    An abstract class that corresponds to a predictor that can determine whether a data
    point is in-sample or not relative to some training dataset.

    Any novelty detector that extends this class must implement the predict method of
    the Predictor class, which is to return a score for each column representing the
    likelihood of it being out of sample; higher scores are more likely to be
    novelties. The predict_novelties method implemented here gives a Boolean classifier
    for the outputs that deems a sample an outlier if its score exceeds a specified
    cutoff and returns a dataset with both the score and the classification.
    """

    _NOVELTY_OUTPUT_VAR = "is_novelty"
    _SCORE_OUTPUT_VAR = "novelty_score"
    _TAPER_RATE_OUTPUT_VAR = "taper_rate"

    def __init__(self, input_variables: Iterable[Hashable]):
        output_variables = [self._NOVELTY_OUTPUT_VAR, self._SCORE_OUTPUT_VAR]
        super().__init__(input_variables, output_variables)

    def predict_novelties(
        self,
        X: xr.Dataset,
        cutoff: Optional[float] = None,
        taper_function: Optional[TaperFunction] = None,
    ) -> xr.Dataset:
        if cutoff is None:
            cutoff = self._get_default_cutoff()
        if taper_function is None:
            taper_function = self._get_default_taper_function()

        score_dataset = self.predict(X)

        is_novelty = xr.where(score_dataset[self._SCORE_OUTPUT_VAR] > cutoff, 1, 0)
        score_dataset[self._NOVELTY_OUTPUT_VAR] = is_novelty

        taper_rate = taper_function.get_taper_value(
            score_dataset[self._SCORE_OUTPUT_VAR]
        )
        score_dataset[self._TAPER_RATE_OUTPUT_VAR] = taper_rate

        return score_dataset

    @abc.abstractmethod
    def _get_default_cutoff(self) -> float:
        pass

    def _get_default_taper_function(self) -> TaperFunction:
        return MaskTaperFunction(self._get_default_cutoff())
