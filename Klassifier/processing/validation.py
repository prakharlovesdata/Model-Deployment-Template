from Klassifier.config import config

import pandas as pd

def validate_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for unprocessable values."""

    validated_data = input_data.copy()

    # check for numerical variables with NA not seen during training
    if input_data[config.FEATURES].isnull().any().any():
        validated_data = validated_data.fillna(validated_data.mean())


    return validated_data
