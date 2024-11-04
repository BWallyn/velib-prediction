"""
This is a boilerplate pipeline 'train_model'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.train_model.nodes import get_split_train_val_cv


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_split_train_val_cv,
                inputs=["df_train_prepared", "params:n_splits"],
                outputs="list_train_valid",
                name="Create_split_expanding_windows"
            ),
        ],
        inputs="df_train_prepared",
        namespace="train_model"
    )
