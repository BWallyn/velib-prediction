"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.train_model.nodes import (
    create_mlflow_experiment_if_needed,
    select_columns,
    split_train_valid_last_hours,
    train_final_model,
    train_model_bayesian_opti,
)
from velib_prediction.utils.utils import rename_columns, sort_dataframe


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        pipe=[
            node(
                func=rename_columns,
                inputs=["df_training_feat_engineered", "params:dict_rename_target"],
                outputs="df_w_target_renamed",
                name="Rename_target_column",
            ),
            node(
                func=sort_dataframe,
                inputs=["df_w_target_renamed", "params:feat_date"],
                outputs="df_training_sorted",
                name="Sort_dataframe_by_date",
            ),
            node(
                func=split_train_valid_last_hours,
                inputs=["df_training_sorted", "params:feat_date", "params:n_hours"],
                outputs=["df_train", "df_valid"],
                name="Create_split_train_valid",
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
                func=select_columns,
                inputs=["df_train", "params:model_features", "params:feat_target"],
                outputs="df_train_col_selected",
                name="Select_columns_train",
            ),
            node(
                func=select_columns,
                inputs=["df_valid", "params:model_features", "params:feat_target"],
                outputs="df_valid_col_selected",
                name="Select_columns_valid",
            ),
            node(
                func=train_model_bayesian_opti,
                inputs=[
                    "params:run_name",
                    "experiment_id_created",
                    "params:search_params",
                    "df_train_col_selected",
                    "df_valid_col_selected",
                    "params:feat_cat",
                    "params:n_trials",
                ],
                outputs="best_params",
                name="Find_best_parameters_using_bayesian_optimization",
            ),
            node(
                func=rename_columns,
                inputs=["df_test_feat_engineered", "params:dict_rename_target"],
                outputs="df_test_target_renamed",
                name="Rename_target_column_test_set",
            ),
            node(
                func=sort_dataframe,
                inputs=["df_test_target_renamed", "params:feat_date"],
                outputs="df_test_sorted",
                name="Sort_test_by_date",
            ),
            node(
                func=select_columns,
                inputs=[
                    "df_training_sorted",
                    "params:model_features",
                    "params:feat_target",
                ],
                outputs="df_training",
                name="Select_columns_training",
            ),
            node(
                func=select_columns,
                inputs=[
                    "df_test_sorted",
                    "params:model_features",
                    "params:feat_target",
                ],
                outputs="df_test_col_selected",
                name="Select_columns_test",
            ),
            node(
                func=train_final_model,
                inputs=[
                    "experiment_id_created",
                    "df_training",
                    "df_test_col_selected",
                    "params:feat_cat",
                    "best_params",
                ],
                outputs="model_velib",
                name="Train_final_model",
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
        inputs=["df_training_feat_engineered", "df_test_feat_engineered"],
        outputs=["model_velib", "df_training_sorted", "df_test_sorted"],
        namespace="train_model",
    )
