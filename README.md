# 🚲 Velib rides prediction

The goal of this project is to predict the number of rides in Velib stations located in and around Paris.

![Velib](./reports/images/velib-velo-electrique.jpeg)

The app to present details about the project can be found here: https://bwallyn-velib-prediction.streamlit.app


## 🛰️ Architecture

All this project is designed using [Kedro](https://docs.kedro.org).
You can find the architecture of the project here: https://bwallyn.github.io/velib-prediction/


## 📡 Data

Data are extracted from the [Open Data Paris website](https://opendata.paris.fr/explore/dataset/velib-disponibilite-en-temps-reel/information/?disjunctive.name&disjunctive.is_installed&disjunctive.is_renting&disjunctive.is_returning&disjunctive.nom_arrondissement_communes).

A Github action is running every hour to fetch data from the Open Data Paris website and save it in the main branch.


## 🧪 Feature engineering

The feature engineering is located in the `feature_engineering` Kedro pipeline.

The feature engineering consists:
- Information about the date the data was saved.
- Information about the holidays downloaded from the web.


## 🚀 Model

The model used to predict the number of remaining bikes is a Gradient Boosting Model. To do so, we use [CatBoost](https://catboost.ai).
The training of the model can be found in the Kedro pipeline `train_model`.


## 📈 Results


## 📱 App

To present the data and the predictions of the model, a streamlit app is available [here](https://bwallyn-velib-prediction.streamlit.app).


## Guidelines

### Installation guide

All the dependencies are declared in the `pyproject.toml` for `uv` installation.

To install them, run:

```
uv pip install -r pyproject.toml
```

### Development guide

To add developments, you can follow the guide:
- Install all the dependencies and the extra dependencies
```
uv sync --extra dev
```
- Create a new branch using the following guidelines:
  - `feature/<branch-name>` to add a feature to the project.
  - `fix/<branch-name>` to fix a bug in the project.
- Code.
- Add unit tests.
- Create a pull request to the `dev` branch.
- Merge and if it validated create a pull request to the `main` branch.



## A word on Kedro
### Overview

This is your new Kedro project with Kedro-Viz setup, which was generated using `kedro 0.19.7`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

### Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

### How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

### How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

### How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and `src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

### Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. Install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

### How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

#### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

#### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

#### IPython
And if you want to run an IPython session:

```
kedro ipython
```

#### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

[Further information about using notebooks for experiments within Kedro projects](https://docs.kedro.org/en/develop/notebooks_and_ipython/kedro_and_notebooks.html).
### Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html).
