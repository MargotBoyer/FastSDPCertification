import os
import sys
import pandas as pd
import numpy as np
import argparse

# CONSTANTS

infinity = 1e8


def get_m_indexes_of_higher_values_in_list(L, m, indexes_pruned=[]):
    """
    Returns the indexes of the m highest values in the list L.
    """
    return [
        i
        for i in sorted(range(len(L)), key=lambda i: L[i], reverse=True)
        if i not in indexes_pruned
    ][:m]


def deduct_two_lists(L1, L2):
    """
    Deducts the values of L2 from L1 and returns the result.
    """
    return [x - y for x, y in zip(L1, L2)]


def remove_values_of_list_from_list(L1, L2):
    """
    Removes the values of L2 from L1 and returns the result.
    """
    return [x for x in L1 if x not in L2]


def round_list_depth_3(L, decimal: int = 6):
    """
    Rounds the values in a 3D list to 2 decimal places.
    """
    return [
        [[round(x, decimal) for x in sublist] for sublist in sublist2] for sublist2 in L
    ]


def round_list_depth_2(L, decimal: int = 6):
    """
    Rounds the values in a 2D list to 2 decimal places.
    """
    return [[round(x, decimal) for x in sublist] for sublist in L]


def change_to_zero_negative_values(L, dim: int = 2):
    """
    Change negative values to zero in a list.
    """
    return (
        [max(0, x) for x in L]
        if dim == 1
        else (
            [[max(0, x) for x in sublist] for sublist in L]
            if dim == 2
            else [
                [[max(0, x) for x in sublist] for sublist in sublist2] for sublist2 in L
            ]
        )
    )


def exists_two_similar_pairs_in_three_lists(L, T, S):
    vu = set()
    for i, (num, t, s) in enumerate(zip(L, T, S)):
        if (num, t, s) in vu:
            return (True, num, t, s)
        vu.add((num, t, s))
    return (False, None, None, None)


def sort_lists_by_first(L1, *other_lists):
    """
    Sorts three lists based on the values of the first list.
    """
    for i, lst in enumerate(other_lists):
        if len(lst) != len(L1):
            raise ValueError(
                f"La liste à l'index {i+1} a une longueur différente de la liste principale"
            )

    indexed_values = list(enumerate(L1))
    sorted_indexed_values = sorted(indexed_values, key=lambda x: x[1])
    sorted_indices = [i for i, _ in sorted_indexed_values]

    sorted_L1 = [L1[i] for i in sorted_indices]

    sorted_other_lists = []
    for lst in other_lists:
        sorted_lst = [lst[i] for i in sorted_indices]
        sorted_other_lists.append(sorted_lst)

    return [sorted_L1] + sorted_other_lists


def deduplicate_and_sum(I_list, L_list, num_matrix_list, values_list):
    """
    Detect doublons (i,l) in I x L, adds corresponding values,
    and returns deduplicated lists I_dedupe, L_dedupe, values_summed.

    Args:
        I_list : List of indices i
        L_list : List of indices l
        num_matrix_list: List of matrix numbers
        values_list: List of values associated with (i,l, num_matrix)

    Returns:
        tuple: (I_dedupe, L_dedupe, values_summed)
    """
    if (
        len(I_list) != len(L_list)
        or len(I_list) != len(values_list)
        or len(num_matrix_list) != len(I_list)
        or len(num_matrix_list) != len(L_list)
        or len(num_matrix_list) != len(values_list)
        or len(L_list) != len(values_list)
    ):
        raise ValueError("Les trois listes doivent avoir la même taille")

    couple_sums = {}
    for idx in range(len(I_list)):
        couple = (I_list[idx], L_list[idx], num_matrix_list[idx])

        if couple in couple_sums.keys():
            couple_sums[couple] += values_list[idx]
        else:
            couple_sums[couple] = values_list[idx]

    I_dedupe = []
    L_dedupe = []
    num_matrix_dedupe = []
    values_summed = []

    for (i_val, l_val, num_matrix_val), sum_val in couple_sums.items():
        I_dedupe.append(i_val)
        L_dedupe.append(l_val)
        num_matrix_dedupe.append(num_matrix_val)
        values_summed.append(sum_val)

    return I_dedupe, L_dedupe, num_matrix_dedupe, values_summed


def divide_list_by(L, scalar):
    """
    Divides each element of the list L by the scalar value.

    Args:
        L (list): List of numbers to be divided.
        scalar (float): The scalar value to divide by.

    Returns:
        list: New list with each element divided by the scalar.
    """
    if scalar == 0:
        raise ValueError("Division by zero is not allowed.")
    return [x / scalar for x in L]


def summing_values_two_dicts(dict1, dict2):
    """
    Returns dict1 update with dict2 values summed.

    Args:
        dict1 (dict): First dictionary of type key : (layer, neuron).
        dict2 (dict): First dictionary of type key : (layer, neuron).

    Returns:
        dict: New concatenated dictionnary with summed values.
    """
    for (layer, neuron), value in dict2.items():
        if (layer, neuron) in dict1:
            dict1[(layer, neuron)] += value
        else:
            dict1[(layer, neuron)] = value
    return dict1


def add_row_from_dict(df, row_dict):
    """
    Ajoute une ligne à un DataFrame à partir d'un dictionnaire.

    Parameters:
    -----------
    df : pandas.DataFrame
        Le DataFrame auquel ajouter une ligne
    row_dict : dict
        Le dictionnaire contenant les données à ajouter

    Returns:
    --------
    pandas.DataFrame
        Le DataFrame mis à jour avec la nouvelle ligne

    Notes:
    ------
    - Si le dictionnaire contient des clés absentes du DataFrame, de nouvelles colonnes
      sont ajoutées avec des valeurs None pour toutes les lignes existantes
    - Si le DataFrame contient des colonnes absentes du dictionnaire, des valeurs None
      sont ajoutées pour ces colonnes dans la nouvelle ligne
    """
    new_row_df = pd.DataFrame([row_dict], columns=df.columns)

    return pd.concat([df, new_row_df], ignore_index=True)


def get_git_root():
    """Trouve et retourne le chemin racine du repository git"""
    current_path = os.path.abspath(".")

    # Remonte jusqu'à trouver le dossier .git ou la racine du système
    while current_path != os.path.dirname(
        current_path
    ):  # Arrête à la racine du système
        if os.path.exists(os.path.join(current_path, ".git")):
            return current_path
        current_path = os.path.dirname(current_path)

    # Si on ne trouve pas de dossier .git, on peut fallback sur une autre méthode
    # Par exemple, utiliser le nom du dossier racine du projet
    import inspect

    return os.path.dirname(
        os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    )


# Ajoute automatiquement le chemin racine au sys.path lors de l'import
git_root = get_git_root()
if git_root not in sys.path:
    sys.path.append(git_root)


def get_project_path(relative_path):
    """Retourne le chemin absolu à partir d'un chemin relatif à la racine du projet"""
    return os.path.join(git_root, relative_path)


def str_to_list(arg):
    # Enlève les crochets et sépare les éléments
    return list(map(int, arg.strip("[]").split(",")))


def check_condition_decorator(condition_method_name):
    """
    Décorateur qui vérifie une condition définie dans une méthode de classe
    avant d'exécuter la méthode décorée.

    Args:
        condition_method_name: Nom de la méthode qui vérifie la condition (sans 'self.')
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Obtient la méthode de vérification à partir du nom
            check_method = getattr(self, condition_method_name)
            if check_method(*args, **kwargs):
                return func(self, *args, **kwargs)
            else:
                return None

        return wrapper

    return decorator


def add_functions_to_class(*functions):
    """
    This decorator allows you to add multiple functions to a class
    """

    def decorator(classe):
        # Parcourir toutes les fonctions à ajouter
        for function in functions:
            # Ajouter chaque fonction à la classe
            setattr(classe, function.__name__, function)
        return classe

    return decorator


def count_calls(counter_name):
    """
    This decorator counts the number of times a method is called.
    Args:
        counter_name: Name of the class attribute to store the counter
    """

    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Vérifie si le compteur existe, sinon le crée
            if not hasattr(self.__class__, counter_name):
                setattr(self.__class__, counter_name, 0)
            # Incrémente le compteur
            current = getattr(self.__class__, counter_name)
            setattr(self.__class__, counter_name, current + 1)
            # Appelle la fonction originale
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


# Fonction auxiliaire pour convertir une chaîne en liste de flottants
def parse_float_list(arg):
    return [float(x) for x in arg.split(",")]


# Fonction auxiliaire pour convertir une chaîne en liste de chaînes
def parse_string_list(arg):
    return arg.split(",")
