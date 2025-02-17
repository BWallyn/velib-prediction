# =================
# ==== IMPORTS ====
# =================

import pandas as pd
import pandera as pa
from pydantic import BaseModel, field_validator

# ===============
# ==== CLASS ====
# ===============


class ValidatedDataFrame(BaseModel):
    """DataFrame validation class."""

    df: pd.DataFrame

    # ==== Validators ====
    @field_validator("df")
    def validate_df(cls, values):
        # Define the schema
        dict_schema = {
            "stationcode": pa.Column(str),
            "is_installed": pa.Column(int, checks=pa.Check.isin([0, 1])),
            "capacity": pa.Column(int, checks=pa.Check.greater_than(0)),
            "numdocksavailable": pa.Column(
                int, checks=pa.Check.greater_than_or_equal_to(0)
            ),
            "mechanical": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
            "ebike": pa.Column(int, checks=pa.Check.greater_than_or_equal_to(0)),
            "is_renting": pa.Column(int, checks=pa.Check.isin([0, 1])),
            "is_returning": pa.Column(int, checks=pa.Check.isin([0, 1])),
            "code_insee_commune": pa.Column(str),
            "duedate_year": pa.Column(
                "int32", checks=pa.Check.greater_than_or_equal_to(2020), coerce=True
            ),
            "duedate_month": pa.Column(
                "int32", checks=pa.Check.between(1, 12), coerce=True
            ),
            "duedate_day": pa.Column(
                "int32", checks=pa.Check.between(1, 31), coerce=True
            ),
            "duedate_weekday": pa.Column(
                "int32", checks=pa.Check.between(0, 6), coerce=True
            ),
            "duedate_weekend": pa.Column(
                "int32", checks=pa.Check.isin([0, 1]), coerce=True
            ),
        }
        schema = pa.DataFrameSchema(dict_schema)
        # Validate the DataFrame
        schema.validate(values)
        return values

    # ==== Freezing the class ====
    class Config:
        frozen = True
        arbitrary_types_allowed = True
