import itertools
from typing import List
import logging
import mosek
from itertools import combinations
import pandas as pd
import os

from tools import (
    get_project_path,
    create_folder,
    remove_values_of_list_from_list,
    create_folder_benchmark,
)

logger_mosek = logging.getLogger("Mosek_logger")

all_possible_cuts = [
    "betaibetaj",
    "Tij",
    "RLT",
    "triangularization",
    "ReLU_active_ub_neurons",
    "ReLU_active_lb_neurons",
    "ReLU_active_ub_decomposed",
    "ReLU_active_lb_decomposed",
    "beta_logits_comparaison",
]


def compute_cuts_str(cuts: List):
    """
    Compute the cuts string.
    """
    cuts_str = "_".join(cuts)
    cuts_str = cuts_str.replace("^", "")
    return cuts_str


def create_all_cuts_to_test(self):
    """
    Create all cuts to test.
    """
    self.cuts_to_test = [[]]

    if self.cuts is not None:
        if self.all_combinations_cuts:
            if self.name == "LanSDP":
                self.cuts = remove_values_of_list_from_list(
                    self.cuts,
                    [
                        "betaibetaj",
                        "Adversariales",
                        "Tij",
                        "zbar",
                    ],
                )
            elif self.name == "MdSDP":
                self.cuts = remove_values_of_list_from_list(self.cuts, ["zbar"])

            for r in range(1, len(self.cuts) + 1):
                for combo in combinations(self.cuts, r):
                    self.cuts_to_test.append(list(combo))
        else:
            self.cuts_to_test = [self.cuts]
    print(
        f"Number of cuts to test: {len(self.cuts_to_test)}\nCuts to test: {self.cuts_to_test}"
    )


def print_solution_to_file_for_cb_solver(mat, index_matrix, dim, file_cb):
    """
    Print the solutions of the optimization to a file.
    """
    logger_mosek.info("Writing solutions to conic bundle file...")
    for j in range(dim):
        for i in range(j):
            file_cb.write(f"{index_matrix} {i} {j} {mat[i][j]} ")
            file_cb.write("\n")


def print_dual_variable_to_file_for_cb_solver(list_cstr, file_cb):
    """
    Print the dual variables of the optimization to a file.
    """
    logger_mosek.info("Writing dual variables to conic bundle file...")
    for ind, cstr in enumerate(list_cstr):
        name = cstr["name"]
        dual_value = cstr["dual_value"]
        file_cb.write(f"{name} : {dual_value} ")
        file_cb.write("\n")


def concat_dataframes_with_missing_columns(df1, df2):
    """
    Concatène deux DataFrames en gérant les colonnes manquantes.
    Si une colonne existe dans un DataFrame mais pas dans l'autre,
    elle est ajoutée avec des valeurs None pour les lignes correspondantes.

    Parameters:
    -----------
    df1 : pandas.DataFrame
        Premier DataFrame à concaténer
    df2 : pandas.DataFrame
        Second DataFrame à concaténer

    Returns:
    --------
    pandas.DataFrame
        Le DataFrame résultant de la concaténation des deux DataFrames d'entrée
    """
    # Créer des copies pour éviter de modifier les originaux
    df1_copy = df1.copy()
    df2_copy = df2.copy()

    # Obtenir l'ensemble de toutes les colonnes des deux DataFrames
    all_columns = set(df1_copy.columns).union(set(df2_copy.columns))

    # Ajouter les colonnes manquantes à df1 avec des valeurs None
    for col in all_columns - set(df1_copy.columns):
        df1_copy[col] = None

    # Ajouter les colonnes manquantes à df2 avec des valeurs None
    for col in all_columns - set(df2_copy.columns):
        df2_copy[col] = None

    # Assurer que les deux DataFrames ont les mêmes colonnes et dans le même ordre
    df1_copy = df1_copy[sorted(all_columns)]
    df2_copy = df2_copy[sorted(all_columns)]

    # Concaténer les deux DataFrames
    result_df = pd.concat([df1_copy, df2_copy], ignore_index=True)

    return result_df


def replace_none_with_false(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Remplace toutes les valeurs None (ou NaN) d'une colonne spécifique par False.

    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame à modifier
    column_name : str
        Le nom de la colonne dans laquelle remplacer les valeurs None par False

    Returns:
    --------
    pandas.DataFrame
        Le DataFrame avec les valeurs None remplacées par False dans la colonne spécifiée
    """
    # Créer une copie du DataFrame pour éviter de modifier l'original
    df_copy = df.copy()

    # Vérifier si la colonne existe dans le DataFrame
    if column_name not in df_copy.columns:
        raise ValueError(f"La colonne '{column_name}' n'existe pas dans le DataFrame")

    # Remplacer les valeurs None/NaN par False dans la colonne spécifiée
    df_copy[column_name] = df_copy[column_name].fillna(False)

    return df_copy


def check_cuts(row):
    """
    Vérifie si la valeur de la colonne 'cuts' est égale à 1.

    Parameters:
    -----------
    row : pandas.Series
        Une ligne du DataFrame

    Returns:
    --------
    bool
        True si la valeur de 'cuts' est égale à 1, sinon False
    """
    cuts_str = ""
    if row["Tij"]:
        cuts_str += "Tij, "
    if row["triangularization"]:
        cuts_str += "tri, "
    if row["RLT"]:
        cuts_str += "RLT, "
    if row["allMC"]:
        cuts_str += "allMC, "
    if (
        not row["Tij"]
        and not row["RLT"]
        and not row["triangularization"]
        and not row["allMC"]
    ):
        cuts_str += "$\\emptyset$"
    else:
        cuts_str = cuts_str[:-2]
    return cuts_str


def add_cuts_to_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ajoute une colonne 'cuts' au DataFrame en fonction des colonnes existantes.

    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame à modifier

    Returns:
    --------
    pandas.DataFrame
        Le DataFrame avec la nouvelle colonne 'cuts'
    """
    for cut in all_possible_cuts:
        if cut not in df.columns:
            df[cut] = False
    return df
