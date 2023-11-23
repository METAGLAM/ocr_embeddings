from typing import Union, List, Dict

import numpy.typing as npt
import pandas as pd

# Generic typing

Number = Union[int, float]
Array = npt.ArrayLike
Dataset = Union[pd.DataFrame, pd.Series, Array]
StrList = List[str]
IntList = List[int]
MetricsDict = Dict[str, Number]
