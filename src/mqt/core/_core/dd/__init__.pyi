import numpy as np
import numpy.typing as npt

class StateGenerator:
    def __init__(self, seed: int = 0) -> None: ...
    def get_random_state_from_structured_dd(
        self, levels: int, nodes_per_level: list[int]
    ) -> npt.NDArray[np.complex_]: ...
