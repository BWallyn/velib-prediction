"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import extract_date_features, get_holidays, split_train_test


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
            ),
            node(
                func=extract_date_features,
                inputs=["df_train", "params:feat_date"],
                outputs="df_train_w_date_feat",
                name="Add_date_feat_train"
            ),
            node(
                func=extract_date_features,
                inputs=["df_test", "params:feat_date"],
                outputs="df_test_w_date_feat",
                name="Add_date_feat_test"
            ),
        ],
        inputs=["df_with_bool_cols_upd"],
        namespace="feature_engineering"
    )
