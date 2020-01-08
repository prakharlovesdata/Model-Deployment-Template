import numpy as np
import pandas as pd

from Klassifier.processing.data_management import load_pipeline
from Klassifier.config import config
from Klassifier.processing.validation import validate_inputs

import logging
import typing as t


_logger = logging.getLogger(__name__)

pipeline_file_name = f'{config.PIPELINE_SAVE_FILE}.pkl'
_price_pipe = load_pipeline(file_name=pipeline_file_name)


def make_prediction(*, input_data: t.Union[pd.DataFrame, dict],
                    ) -> dict:
    """Make a prediction using a saved model pipeline.
    Args:
        input_data: Array of model prediction inputs.
    Returns:
        Predictions for each input row, as well as the model version.
    """

    data = pd.DataFrame(input_data)
    validated_data = validate_inputs(input_data=data)

    prediction = _price_pipe.predict(validated_data[config.FEATURES])

    results = {'predictions': prediction}

    _logger.info(
        f'Making predictions with model version: 0.0.1 '
        f'Inputs: {validated_data} '
        f'Predictions: {results}')

    return results
