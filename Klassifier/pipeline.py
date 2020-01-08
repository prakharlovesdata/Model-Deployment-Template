from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier

from processing import preprocessing as pp

from Klassifier.config import config

import logging


_logger = logging.getLogger(__name__)


source_pipe = Pipeline(
    [
        ('log transform',
            pp.Log_transformation(variables=config.FEATURES)),
        ('scaler', StandardScaler()),
        ('knn_model', KNeighborsClassifier(n_neighbors=7))
    ]
)
