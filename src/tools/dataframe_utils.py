import pandas as pd


def create_one_hot_encoding(
        df: pd.DataFrame, column: str, ignore_index: bool = False,
        drop_column: bool = False, **kwargs) -> pd.DataFrame:
    """
    Given a dataframe and column, this function creates one hot encoding of
    column. Also, it removes old column.
    Args:
        df (pd.DataFrame): dataframe to apply one hot encoding.
        column (str): column name.
        ignore_index: ignore index or not when combine dataframe and one hot
            encoding results.
        drop_column (optional, bool): drop original column or not. Default
            value is True.
    Returns:
        a pd.DataFrame with new one hot encoding columns
    """
    ohe_df = pd.get_dummies(df[column], **kwargs)
    frames = [
        df,
        ohe_df,
    ]
    result_df = pd.concat(frames, axis=1, ignore_index=ignore_index)

    if drop_column:
        result_df.drop(columns=[column], inplace=True)

    return result_df
