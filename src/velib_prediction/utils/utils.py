# =================
# ==== IMPORTS ====
# =================

import pandas as pd

# ===================
# ==== FUNCTIONS ====
# ===================

def rename_columns(df: pd.DataFrame, dict_to_rename: dict[str, str]) -> pd.DataFrame:
    """Rename columns of dataset

    Args:
        df (pd.DataFrame): Input DataFrame
        dict_to_rename (dict[str, str]): Dict of the columns to rename
    Returns:
        (pd.DataFrame): Output DataFrame
    """
    return df.rename(columns=dict_to_rename)
