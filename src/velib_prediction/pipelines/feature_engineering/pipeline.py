"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import drop_unused_columns, set_date_format, update_values_bool_columns


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
                func=update_values_bool_columns,
                inputs=["df_date_set", "params:boolean_columns"],
                outputs="df_with_bool_cols_upd",
                name="reset_values_in_boolean_columns"
            )
        ],
        inputs="df_raw",
        outputs="df_with_bool_cols_upd",
        namespace="feature_engineering"
    )
