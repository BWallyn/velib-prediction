"""
This is a boilerplate pipeline 'feature_engineering'
generated using Kedro 0.19.7
"""

from kedro.pipeline import Pipeline, node, pipeline

from velib_prediction.pipelines.feature_engineering.nodes import (
    add_holidays_period,
    extract_date_features,
    get_holidays,
    split_train_test,
    validate_dataset,
)


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
                outputs="df_training_date",
                name="Add_date_feat_train"
            ),
            node(
                func=extract_date_features,
                inputs=["df_test", "params:feat_date"],
                outputs="df_test_date",
                name="Add_date_feat_test"
            ),
            # TODO fix holidays merge
            node(
                func=add_holidays_period,
                inputs=["df_training_date", "df_holidays", "params:feat_date_holidays", "params:zone_a"],
                outputs="df_train_w_holidays_a",
                name="add_holidays_zone_a_train"
            ),
            node(
                func=add_holidays_period,
                inputs=["df_train_w_holidays_a", "df_holidays", "params:feat_date_holidays", "params:zone_b"],
                outputs="df_train_w_holidays_b",
                name="add_holidays_zone_b_train"
            ),
            node(
                func=add_holidays_period,
                inputs=["df_train_w_holidays_b", "df_holidays", "params:feat_date_holidays", "params:zone_c"],
                outputs="df_train_w_holidays_c",
                name="add_holidays_zone_c_train"
            ),
            # node(
            #     func=add_holidays_period,
            #     inputs=["df_test_w_date_feat", "df_holidays", "params:feat_date_holidays", "params:zone_a"],
            #     outputs="df_test_w_holidays",
            #     name="add_holidays_zone_a_test"
            # ),
            # node(
            #     func=add_holidays_period,
            #     inputs=["df_test_w_holidays", "df_holidays", "params:feat_date_holidays", "params:zone_b"],
            #     outputs="df_test_w_holidays_b",
            #     name="add_holidays_zone_b_test"
            # ),
            # node(
            #     func=add_holidays_period,
            #     inputs=["df_test_w_holidays_b", "df_holidays", "params:feat_date_holidays", "params:zone_c"],
            #     outputs="df_test_w_holidays_c",
            #     name="add_holidays_zone_c_test"
            # ),
            node(
                func=validate_dataset,
                inputs="df_training_date",
                outputs="df_training_feat_engineered",
                name="Validate_training_data",
            ),
            node(
                func=validate_dataset,
                inputs="df_test_date",
                outputs="df_test_feat_engineered",
                name="Validate_test_data",
            ),
        ],
        inputs=["df_with_bool_cols_upd"],
        outputs=["df_training_feat_engineered", "df_test_feat_engineered"],
        namespace="feature_engineering"
    )
