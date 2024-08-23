"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import (
    add_datetime_col,
    drop_unused_columns,
    set_date_format,
    update_values_bool_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=drop_unused_columns,
                inputs=["df_raw", "params:cols_to_remove"],
                outputs="df_wtht_unused_cols",
                name="remove_unused_cols"
            ),
            node(
                func=set_date_format,
                inputs="df_wtht_unused_cols",
                outputs="df_date_set",
                name="set_date_format"
            ),
            node(
                func=add_datetime_col,
                inputs="df_date_set",
                outputs="df_date_added",
                name="add_date_column",
            ),
            node(
                func=update_values_bool_columns,
                inputs=["df_date_added", "params:boolean_columns"],
                outputs="df_with_bool_cols_upd",
                name="reset_values_in_boolean_columns"
            )
        ],
        inputs="df_raw",
        outputs="df_with_bool_cols_upd",
        namespace="data_engineering"
    )
