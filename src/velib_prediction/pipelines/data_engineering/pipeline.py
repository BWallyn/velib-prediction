"""
This is a boilerplate pipeline 'data_engineering'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.data_engineering.nodes import (
    add_datetime_col,
    create_idx,
    drop_unused_columns,
    list_parquet_files,
    merge_datasets,
    set_date_format,
    update_values_bool_columns,
)


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=list_parquet_files,
                inputs="params:path_data",
                outputs="list_files",
                name="list_dataset_files",
            ),
            node(
                func=merge_datasets,
                inputs="list_files",
                outputs="df_raw",
                name="merge_all_datasets",
            ),
            node(
                func=create_idx,
                inputs="df_raw",
                outputs="df_w_idx",
                name="add_index_to_dataset"
            ),
            node(
                func=drop_unused_columns,
                inputs=["df_w_idx", "params:cols_to_remove"],
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
        inputs=None,
        outputs=["df_with_bool_cols_upd", "df_raw"],
        namespace="data_engineering"
    )
