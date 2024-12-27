from typing import Any

from pydantic import BaseModel, field_validator


class BayesianOptiSearchSpace(BaseModel):
    """Create a class to define the search space for the Bayesian Optimization
    """
    hyperparams_search_space: dict[str, Any]

    @field_validator("hyperparams_search_space")
    def validate_hyperparams_search_space(cls, v):
        """Validate the hyperparameter search space

        Raises:
            ValueError: If some provided distributions are not supported
        """
        proposed_distributions = set(
            [sampling_params["sampling_type"] for _, sampling_params in v.items()]
        )
        supported_distributions = {"uniform", "loguniform", "uniform", "categorical"}
        if (proposed_distributions - supported_distributions) != set():
            raise ValueError(
                f"Some provided distributions are not supported: {proposed_distributions - supported_distributions}"
            )
        return v
