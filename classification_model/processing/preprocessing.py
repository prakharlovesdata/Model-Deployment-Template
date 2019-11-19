import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class Log_transformation(BaseEstimator, TransformerMixin):
    """ Log transform the variables """

    def __init__(self, variables=None) -> None:
        if not isinstance(variables, list):
            self.variables = [variables]
        else:
            self.variables = variables

    def fit(self, X: pd.DataFrame, y: pd.Series = None
            ) -> None:
        """Fit statement to accomodate the sklearn pipeline."""

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the transforms to the dataframe."""

        X = X.copy()
        for feature in self.variables:
            X[feature] = np.log1p(X[feature])

        return X
