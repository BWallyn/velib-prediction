"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import get_holidays, split_train_test


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=get_holidays,
                inputs=None,
                outputs="df_holidays",
                name="download_holiday_dates"
            ),
            node(
                func=split_train_test,
                inputs=["df_with_bool_cols_upd", "params:feat_date", "params:delta_days"],
                outputs=["df_train", "df_test"],
                name="split_train_test"
            )
        ],
        inputs=["df_with_bool_cols_upd"],
        namespace="feature_engineering"
    )
