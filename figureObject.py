import dataclasses
import numpy as np


@dataclasses.dataclass
class Figure:
    name: str
    img: np.array
    dims: dict
