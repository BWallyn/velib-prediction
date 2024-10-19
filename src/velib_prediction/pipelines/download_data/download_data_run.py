# =================
# ==== IMPORTS ====
# =================

from nodes import download_data, save_data

# ===================
# ==== FUNCTIONS ====
# ===================

def main(url: str, path_data: str) -> None:
    """Download and save data

    Args:
        url (str): Path to the dataset online
        path_data (str): Path to the folder where to store the dataset downloaded
    Returns:
        None
    """
    df = download_data(url)
    save_data(df, path_data=path_data)

# =============
# ==== RUN ====
# =============

if __name__ == '__main__':
    # Options
    URL = "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/velib-disponibilite-en-temps-reel/records?refine=nom_arrondissement_communes%3A%22Paris%22&refine=nom_arrondissement_communes%3A%22Levallois-Perret%22&refine=nom_arrondissement_communes%3A%22Puteaux%22&refine=nom_arrondissement_communes%3A%22Suresnes%22&refine=nom_arrondissement_communes%3A%22Boulogne-Billancourt%22&refine=nom_arrondissement_communes%3A%22Clichy%22"
    PATH_DATA = '../../../../data/01_raw/'

    # Run
    main(url=URL, path_data=PATH_DATA)
