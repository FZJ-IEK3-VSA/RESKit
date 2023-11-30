from pathlib import Path
import os
import pandas as pd


def load_dataset(datasetname: str = "Initial"):
    """Returns the selected dataset as Sereis.

    Args:
        datasetname (str, optional): Name of the dataset from. Defaults to 'Initial'.

    Returns:
        [pd.Series]: [description]
    """

    df = pd.read_excel(
        os.path.join(Path(__file__).absolute().parent, "CSP_database.xlsx"), index_col=0
    )

    assert datasetname in df.columns

    df = df[datasetname]

    return df


if __name__ == "__main__":
    df = load_dataset("Validation 1")
    assert isinstance(df, pd.Series)
    print(df["dataset name"])
    print(df)
