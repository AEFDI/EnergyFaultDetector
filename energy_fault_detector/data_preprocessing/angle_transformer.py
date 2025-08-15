import logging
from typing import Optional, List, Union

import numpy as np
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from energy_fault_detector.core.data_transformer import DataTransformer

logger = logging.getLogger('energy_fault_detector')


class AngleTransformer(DataTransformer):
    """Transforms features containing angles to their sine and cosine values.
    Currently, assumes all angles are in degrees.

    Attributes:
        angles: List of feature names (str) which are angles that need to be transformed.
        trust_bad_angles: If True angle features that neither suit the value range (-180, 180) nor (0, 360) will be
            transformed the same way as valid angles. If False, out of range angle features will be dropped.
    """

    def __init__(self, angles: List[str] = None, trust_bad_angles: bool = False):
        super().__init__()
        self.angles: List[str] = angles if angles is not None else []
        self.trust_bad_angles = trust_bad_angles

    # pylint: disable=attribute-defined-outside-init
    # noinspection PyAttributeOutsideInit
    def fit(self, x: Union[np.array, pd.DataFrame], y: Optional[np.array] = None) -> 'AngleTransformer':
        """Sets feature names in and out and detects value ranges of angle features."""

        self.feature_names_in_ = x.columns.to_list()
        self.n_features_in_ = len(x.columns)

        features_out = [col for col in x.columns if col not in self.angles]
        angle_features = []
        invalid_range_features = []
        ranges = {}
        for col in self.angles:
            if col in self.feature_names_in_:
                min_val = x[col].min()
                max_val = x[col].max()
                if -180 <= min_val < 0 <= max_val <= 180:
                    ranges[col] = (-180, 180)
                elif 0 <= min_val and max_val <= 360:
                    ranges[col] = (0, 360)
                else:
                    if self.trust_bad_angles:
                        logger.info('Angle feature %s does not fit both valid angle ranges (-180, 180) or (0, 360). '
                                    'However, the trust bad angles setting is enabled, thus out of range values will '
                                    'be adjusted to the range (0, 360)' % col)
                        features_out.extend([f'{col}_sine', f'{col}_cosine'])
                        ranges[col] = (0, 360)
                        angle_features.extend([col])
                        invalid_range_features.extend([col])
                        continue
                    else:
                        logger.info('Angle feature %s does not fit both valid angle ranges (-180, 180) or (0, 360). '
                                    'Thus it will be dropped.' % col)
                        angle_features.extend([col])  # add it to angle_features so that it can be removed in transform
                        invalid_range_features.extend([col])
                        continue

                features_out.extend([f'{col}_sine', f'{col}_cosine'])
                angle_features.extend([col])

        self.feature_names_out_ = features_out
        self.angles_features_ = angle_features
        self.ranges_ = ranges
        self.invalid_range_features_ = invalid_range_features
        return self

    # pylint: disable=attribute-defined-outside-init
    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Transforms the angle angles to sine and cosine values."""
        check_is_fitted(self)
        x_ = x.copy()
        return self._convert_angles(x_, self.angles_features_, self.invalid_range_features_)

    # pylint: disable=attribute-defined-outside-init
    def inverse_transform(self, x: pd.DataFrame) -> pd.DataFrame:
        """Get the angles based on their sine values and drop the sine/cosine values.
        NOTE: if sine/cosine values are out of possible range (-1, 1), values below -1 are set to -1
        and values above 1 to 1.
        """

        for col in self.angles_features_:
            if col in self.invalid_range_features_ and not self.trust_bad_angles:
                # Invalid range features will not be part of the dataset after transform if the user does not trust bad
                # angles. So these features will be skipped
                continue
            sine = x[f'{col}_sine'].copy()
            cosine = x[f'{col}_cosine'].copy()

            sine[sine < -1] = -1
            sine[sine > 1] = 1
            cosine[cosine < -1] = -1
            cosine[cosine > 1] = 1

            x[col] = np.nan
            x[col] = np.arctan2(sine, cosine) * 180 / np.pi  # maps to -180, 180
            if self.ranges_[col] == (0, 360):
                x.loc[x[col] < 0, col] = x.loc[x[col] < 0, col] + 360

            x = x.drop([f'{col}_sine', f'{col}_cosine'], axis=1)

        if self.trust_bad_angles:
            x_inv = x[self.feature_names_in_]
        else:
            x_inv = x[[feature for feature in self.feature_names_in_ if feature not in self.invalid_range_features_]]
        return x_inv

    def _convert_angles(self, data: pd.DataFrame, angle_columns: List[str], invalid_range_features: List[str],
                        degrees: bool = True):
        """ Converts angles into continuous sine and cosine features.
        Note that in case of out of range angles values and the trust bad angles option being enabled, the sine / cosine
        transform will work just fine, there is no need to convert out of range values using modulo because of the
        periodicity at play in sine and cosine.
        """
        if degrees:
            constant = np.pi / 180
        else:
            constant = 1.

        for col in angle_columns:
            valid_range = col not in invalid_range_features
            valid_for_conversion = valid_range or (not valid_range and self.trust_bad_angles)
            if valid_for_conversion:
                data[f'{col}_sine'] = np.sin(data[col] * constant)
                data[f'{col}_cosine'] = np.cos(data[col] * constant)
            data = data.drop(col, axis=1)
        return data

    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Returns the list of feature names output by the angle transformation step."""
        check_is_fitted(self)
        return self.feature_names_out_
