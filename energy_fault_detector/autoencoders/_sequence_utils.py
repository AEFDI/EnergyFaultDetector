from typing import Literal, Optional, List

import numpy as np
import pandas as pd

ReducerMode = Literal["first_full_rest_last", "all_last"]


def sequences_to_dataframe(
    sequences: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
    columns: Optional[List[str]] = None,
    mode: ReducerMode = "first_full_rest_last",
) -> pd.DataFrame:
    """Reduce 3D sequence predictions to a 2D time-indexed DataFrame.

    Expects sequences with shape (n_sequences, sequence_length, n_features) and optional per-sequence timestamps
    with shape (n_sequences, sequence_length). The output is a 2D DataFrame with one row per chosen time step.

    Two reduction modes are supported:

    - "first_full_rest_last":
        - If there is only one sequence: return all its time steps.
        - If there are multiple sequences:
            - keep all time steps of the first sequence,
            - for all subsequent sequences, keep only the last time step.

        This yields exactly one prediction per original timestamp for typical sliding-window setups.
    - "all_last":
        For each sequence, keep only the last time step.
        This yields one prediction per sequence (and per last-timestamp of each window).

    Args:
        sequences: Array of shape (n_sequences, sequence_length, n_features).
        timestamps: Optional array of shape (n_sequences, sequence_length) with datetime-like values.
        columns: Optional list of column names for the output DataFrame.
        mode: Reduction mode to use, one of {"first_full_rest_last", "all_last"}.

    Returns:
        A pandas DataFrame with one row per selected time step and `columns` as column names.
        If `timestamps` is provided, it is used as the index; otherwise the index is a default RangeIndex.

    Raises:
        ValueError: If an unknown reduction mode is provided.
    """
    if sequences.size == 0:
        return pd.DataFrame(columns=columns)

    if mode == "first_full_rest_last":
        if len(sequences) == 1:
            return pd.DataFrame(
                data=sequences[0],
                index=None if timestamps is None else timestamps[0],
                columns=columns,
            )

        data = np.vstack([sequences[0], sequences[1:, -1]])
        if timestamps is not None:
            index = np.hstack([timestamps[0], np.array(timestamps)[1:, -1]])
        else:
            index = None

        return pd.DataFrame(data=data, index=index, columns=columns)

    if mode == "all_last":
        data = sequences[:, -1, :]
        if timestamps is not None:
            index = np.array(timestamps)[:, -1]
        else:
            index = None

        return pd.DataFrame(data=data, index=index, columns=columns)

    raise ValueError(f"Unknown mode: {mode}")
