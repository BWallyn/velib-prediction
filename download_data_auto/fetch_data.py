from ..src.velib_prediction.pipelines.download_data.nodes import (
    download_data,
    save_data,
)


def main():
    df_velib = download_data()
    save_data(df_velib, path_data="data/01_raw")


if __name__ == "__main__":
    main()