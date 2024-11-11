"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.train_model.nodes import (
    add_lags_sma,
    get_split_train_val_cv,
    train_model_cv_mlflow,
)
from velib_prediction.utils.utils import drop_columns, rename_columns, sort_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=rename_columns,
                inputs=["df_train_prepared", "params:dict_rename_target"],
                outputs="df_w_target_renamed",
                name="Rename_target_column",
            ),
            node(
                func=add_lags_sma,
                inputs=[
                    "df_w_target_renamed",
                    "params:lags_to_try",
                    "params:feat_id",
                    "params:feat_date",
                    "params:feat_target",
                    "params:n_shift",
                ],
                outputs="df_train_w_lags",
                name="Add_lag_to_dataset",
            ),
            node(
                func=sort_dataframe,
                inputs=["df_train_w_lags", "params:feat_date"],
                outputs="df_sorted",
                name="Sort_dataframe_by_date",
            ),
            node(
                func=drop_columns,
                inputs=["df_sorted", "params:feat_date"],
                outputs="df_w_date_col",
                name="Drop_date_column",
            ),
            node(
                func=get_split_train_val_cv,
                inputs=["df_w_date_col", "params:n_splits"],
                outputs="list_train_valid",
                name="Create_split_expanding_windows"
            ),
            node(
                func=train_model_cv_mlflow,
                inputs=[
                    "params:run_name",
                    "params:experiment_id",
                    "list_train_valid",
                    "params:list_feat_cat",
                    "params:catboost_parameters",
                    "params:verbose",
                ],
                outputs="model",
                name="Train_catboost_model_using_cross_validation"
            ),
        ],
        inputs="df_train_prepared",
        namespace="train_model"
    )
