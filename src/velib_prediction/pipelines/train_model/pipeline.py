"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.train_model.nodes import (
    add_lags_sma,
    create_mlflow_experiment_if_needed,
    split_train_valid_last_hours,
    train_model_bayesian_opti,
    train_model_mlflow,
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
                func=split_train_valid_last_hours,
                inputs=["df_w_date_col", "params:n_hours"],
                outputs=["df_train", "df_valid"],
                name="Create_split_train_valid"
            ),
            node(
                func=create_mlflow_experiment_if_needed,
                inputs=[
                    "params:experiment_folder",
                    "params:experiment_name",
                    "params:experiment_id",
                ],
                outputs="experiment_id_created",
                name="Create_MLflow_experiment_id_if_needed",
            ),
            node(
                func=train_model_bayesian_opti,
                inputs=[
                    "params:run_name",
                    "experiment_id_created",
                    "params:search_params",
                    "params:list_feat_cat",
                    "df_train",
                    "df_valid",
                    "params:feat_cat",
                    "params:n_trials",
                ],
                outputs="best_params",
                name="Find_best_parameters_using_bayesian_optimization"
            ),
            # node(
            #     func=train_model_mlflow,
            #     inputs=[
            #         "experiment_id_created",
            #         "df_train",
            #         "df_test_w_date_feat",
            #     ]
            # )
        ],
        inputs="df_train_prepared",
        namespace="train_model"
    )
