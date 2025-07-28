import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class InfeasibilityAnalyzer:
    """Classe pour analyser l'infaisabilité dans les modèles Gurobi."""

    def __init__(self, model=None):
        """
        Initialise l'analyseur avec un modèle existant ou crée un nouveau modèle.

        Args:
            model: Un modèle Gurobi existant (optionnel)
        """
        self.model = model if model is not None else gp.Model()
        self.iis_computed = False
        self.iis_constraints = []
        self.iis_bounds = []

    def set_model(self, model):
        """
        Définit le modèle à analyser.

        Args:
            model: Un modèle Gurobi
        """
        self.model = model
        self.iis_computed = False

    def check_feasibility(self):
        """
        Vérifie si le modèle est faisable.

        Returns:
            bool: True si le modèle est faisable, False sinon
        """
        self.model.optimize()

        status = self.model.status

        if status == GRB.OPTIMAL:
            print("Le modèle est FAISABLE avec une solution optimale.")
            return True
        elif status == GRB.INFEASIBLE:
            print("Le modèle est INFAISABLE.")
            return False
        elif status == GRB.UNBOUNDED:
            print("Le modèle est NON BORNÉ.")
            return None
        else:
            print(f"Statut du modèle: {status}")
            return None

    def compute_iis(self):
        """
        Calcule l'IIS (Irreducible Inconsistent Subsystem) du modèle.
        """
        is_feasible = self.check_feasibility()

        if is_feasible is not False:
            print("Impossible de calculer l'IIS: le modèle n'est pas infaisable.")
            return

        print("\nCalcul de l'IIS (ensemble minimal de contraintes incompatibles)...")
        self.model.computeIIS()
        self.iis_computed = True

        # Collecter les contraintes infaisables
        self.iis_constraints = []
        for constr in self.model.getConstrs():
            if constr.IISConstr:
                self.iis_constraints.append(constr)

        # Collecter les bornes infaisables
        self.iis_bounds = []
        for var in self.model.getVars():
            if var.IISLb or var.IISUb:
                self.iis_bounds.append(var)

        print(
            f"IIS calculé: {len(self.iis_constraints)} contraintes et {len(self.iis_bounds)} bornes."
        )

    def print_iis(self):
        """
        Affiche les détails de l'IIS.
        """
        if not self.iis_computed:
            print("L'IIS n'a pas encore été calculé. Exécutez compute_iis() d'abord.")
            return

        print("\n===== ANALYSE D'INFAISABILITÉ =====")
        print(f"Nombre total de contraintes dans l'IIS: {len(self.iis_constraints)}")
        print(f"Nombre de bornes de variables dans l'IIS: {len(self.iis_bounds)}")

        print("\n----- CONTRAINTES INFAISABLES -----")
        for i, constr in enumerate(self.iis_constraints):
            print(
                f"Contrainte {i+1}: {constr.ConstrName}   :         --- {constr.Sense} {constr.RHS}"
            )

        print("\n----- BORNES DE VARIABLES INFAISABLES -----")
        for i, var in enumerate(self.iis_bounds):
            bounds_info = []
            if var.IISLb:
                bounds_info.append(f"Borne inférieure: {var.LB}")
            if var.IISUb:
                bounds_info.append(f"Borne supérieure: {var.UB}")

            print(f"Variable {i+1}: {var.VarName} - {', '.join(bounds_info)}")

    def relax_model(self, relax_type="all", return_relaxed=True):
        """
        Crée une version relaxée du modèle infaisable.

        Args:
            relax_type: Type de relaxation ('all', 'constraints', 'bounds')
            return_relaxed: Si True, retourne le modèle relaxé

        Returns:
            Un modèle Gurobi relaxé (optionnel)
        """
        if not self.iis_computed:
            print("L'IIS n'a pas encore été calculé. Exécutez compute_iis() d'abord.")
            return None

        # Créer une copie du modèle original
        relaxed_model = self.model.copy()

        # Variables pour suivre les relaxations
        relaxations = []

        # Ajouter des variables d'écart pour les contraintes infaisables
        if relax_type in ["all", "constraints"]:
            for constr in self.iis_constraints:
                constr_name = constr.ConstrName
                constr_relaxed = relaxed_model.getConstrByName(constr_name)

                if constr.Sense == "=":
                    # Pour les contraintes d'égalité, ajouter deux variables d'écart
                    slack_pos = relaxed_model.addVar(
                        name=f"slack_pos_{constr_name}", lb=0
                    )
                    slack_neg = relaxed_model.addVar(
                        name=f"slack_neg_{constr_name}", lb=0
                    )

                    # Obtenir l'expression de la contrainte originale
                    expr = relaxed_model.getRow(constr_relaxed)
                    relaxed_model.remove(constr_relaxed)

                    # Ajouter la contrainte relaxée
                    relaxed_model.addConstr(
                        expr + slack_pos - slack_neg == constr.RHS,
                        name=f"{constr_name}_relaxed",
                    )

                    # Pénaliser les variables d'écart dans l'objectif
                    relaxed_model.setObjective(
                        relaxed_model.getObjective() + 1000 * (slack_pos + slack_neg)
                    )
                    relaxations.append((constr_name, "=", "slack_pos & slack_neg"))

                elif constr.Sense == "<":
                    # Pour les contraintes ≤, ajouter une variable d'écart positive
                    slack = relaxed_model.addVar(name=f"slack_{constr_name}", lb=0)

                    # Obtenir l'expression de la contrainte originale
                    expr = relaxed_model.getRow(constr_relaxed)
                    relaxed_model.remove(constr_relaxed)

                    # Ajouter la contrainte relaxée
                    relaxed_model.addConstr(
                        expr - slack <= constr.RHS, name=f"{constr_name}_relaxed"
                    )

                    # Pénaliser la variable d'écart dans l'objectif
                    relaxed_model.setObjective(
                        relaxed_model.getObjective() + 1000 * slack
                    )
                    relaxations.append((constr_name, "≤", "slack"))

                elif constr.Sense == ">":
                    # Pour les contraintes ≥, ajouter une variable d'écart négative
                    slack = relaxed_model.addVar(name=f"slack_{constr_name}", lb=0)

                    # Obtenir l'expression de la contrainte originale
                    expr = relaxed_model.getRow(constr_relaxed)
                    relaxed_model.remove(constr_relaxed)

                    # Ajouter la contrainte relaxée
                    relaxed_model.addConstr(
                        expr + slack >= constr.RHS, name=f"{constr_name}_relaxed"
                    )

                    # Pénaliser la variable d'écart dans l'objectif
                    relaxed_model.setObjective(
                        relaxed_model.getObjective() + 1000 * slack
                    )
                    relaxations.append((constr_name, "≥", "slack"))

        # Relaxer les bornes infaisables
        if relax_type in ["all", "bounds"]:
            for var in self.iis_bounds:
                var_name = var.VarName
                var_relaxed = relaxed_model.getVarByName(var_name)

                if var.IISLb:
                    old_lb = var.LB
                    # Relâcher la borne inférieure
                    var_relaxed.setAttr("LB", -GRB.INFINITY)
                    relaxations.append((var_name, "LB", f"{old_lb} → -∞"))

                if var.IISUb:
                    old_ub = var.UB
                    # Relâcher la borne supérieure
                    var_relaxed.setAttr("UB", GRB.INFINITY)
                    relaxations.append((var_name, "UB", f"{old_ub} → +∞"))

        # Afficher les relaxations effectuées
        print("\n----- RELAXATIONS EFFECTUÉES -----")
        if len(relaxations) == 0:
            print("Aucune relaxation effectuée.")
        else:
            for name, type_relax, info in relaxations:
                print(f"{name}: {type_relax} relaxé avec {info}")

        if return_relaxed:
            return relaxed_model
        return None

    def visualize_constraints_conflicts(self):
        """
        Visualise les conflits entre contraintes à l'aide d'une matrice de chaleur.
        Fonctionne uniquement pour les modèles avec 2 variables.
        """
        if not self.iis_computed or len(self.iis_constraints) == 0:
            print("L'IIS n'a pas été calculé ou aucune contrainte infaisable trouvée.")
            return

        # Vérifier si le modèle a 2 variables pour la visualisation 2D
        vars_list = list(self.model.getVars())
        if len(vars_list) != 2:
            print(
                f"La visualisation requiert un modèle avec exactement 2 variables. "
                f"Ce modèle a {len(vars_list)} variables."
            )
            return

        # Extraire les noms des variables
        x_name, y_name = vars_list[0].VarName, vars_list[1].VarName

        # Créer une grille pour tracer les contraintes
        x_min, x_max = -10, 10  # Ajuster selon votre problème
        y_min, y_max = -10, 10  # Ajuster selon votre problème

        x = np.linspace(x_min, x_max, 1000)

        plt.figure(figsize=(12, 8))

        # Tracer chaque contrainte de l'IIS
        for i, constr in enumerate(self.iis_constraints):
            try:
                # Obtenir les coefficients de la contrainte
                row = self.model.getRow(constr)
                coefs = {}
                for j in range(row.size()):
                    var = row.getVar(j)
                    coef = row.getCoeff(j)
                    coefs[var.VarName] = coef

                # Vérifier si les deux variables sont présentes dans la contrainte
                if x_name not in coefs or y_name not in coefs:
                    print(
                        f"La contrainte {constr.ConstrName} ne contient pas les deux variables."
                    )
                    continue

                # Calculer la fonction y en fonction de x pour la contrainte
                # ax + by ≤/=/≥ c => y = (c - ax) / b
                a, b = coefs[x_name], coefs[y_name]
                if b == 0:
                    print(
                        f"La contrainte {constr.ConstrName} a un coefficient nul pour {y_name}."
                    )
                    continue

                c = constr.RHS
                y_values = (c - a * x) / b

                # Tracer la contrainte
                plt.plot(
                    x,
                    y_values,
                    label=f"{constr.ConstrName}: {a}{x_name} + {b}{y_name} {constr.Sense} {c}",
                )

                # Ombrer la région faisable selon le sens de la contrainte
                if constr.Sense == "<":
                    mask = y_values < y_max
                    plt.fill_between(x[mask], y_values[mask], y_max, alpha=0.1)
                elif constr.Sense == ">":
                    mask = y_values > y_min
                    plt.fill_between(x[mask], y_min, y_values[mask], alpha=0.1)

            except Exception as e:
                print(f"Erreur lors du tracé de la contrainte {constr.ConstrName}: {e}")

        # Tracer les bornes des variables
        for var in self.iis_bounds:
            if var.VarName == x_name:
                if var.IISLb:
                    plt.axvline(
                        x=var.LB,
                        color="r",
                        linestyle="--",
                        label=f"{x_name} ≥ {var.LB}",
                    )
                if var.IISUb:
                    plt.axvline(
                        x=var.UB,
                        color="r",
                        linestyle="--",
                        label=f"{x_name} ≤ {var.UB}",
                    )
            elif var.VarName == y_name:
                if var.IISLb:
                    plt.axhline(
                        y=var.LB,
                        color="r",
                        linestyle="--",
                        label=f"{y_name} ≥ {var.LB}",
                    )
                if var.IISUb:
                    plt.axhline(
                        y=var.UB,
                        color="r",
                        linestyle="--",
                        label=f"{y_name} ≤ {var.UB}",
                    )

        plt.xlabel(x_name)
        plt.ylabel(y_name)
        plt.title("Visualisation des contraintes infaisables")
        plt.grid(True)
        plt.legend()
        plt.show()

    def sensitivity_analysis(self):
        """
        Effectue une analyse de sensibilité des contraintes infaisables.
        """
        if not self.iis_computed:
            print("L'IIS n'a pas encore été calculé. Exécutez compute_iis() d'abord.")
            return

        print("\n----- ANALYSE DE SENSIBILITÉ -----")
        print(
            "Estimation des modifications nécessaires pour rendre le modèle faisable:"
        )

        for constr in self.iis_constraints:
            # Créer un modèle de test pour chaque contrainte
            test_model = self.model.copy()
            test_constr = test_model.getConstrByName(constr.ConstrName)

            # Récupérer la ligne de la contrainte
            row = test_model.getRow(test_constr)
            rhs = test_constr.RHS
            sense = test_constr.Sense

            # Créer une variable d'écart
            slack = test_model.addVar(
                name=f"slack_{constr.ConstrName}", lb=-GRB.INFINITY
            )
            slack_abs_value = test_model.addVar(
                name=f"slack_abs_{constr.ConstrName}", lb=0
            )
            test_model.addConstr(
                slack_abs_value >= slack, name=f"{constr.ConstrName}_relaxed_abs_value"
            )
            test_model.addConstr(
                slack_abs_value >= -slack, name=f"{constr.ConstrName}_relaxed_abs_value"
            )

            # Remplacer la contrainte par une version relaxée
            test_model.remove(test_constr)

            if sense == "=":
                test_model.addConstr(
                    row + slack == rhs, name=f"{constr.ConstrName}_relaxed"
                )
            elif sense == "<":
                test_model.addConstr(
                    row - slack <= rhs, name=f"{constr.ConstrName}_relaxed"
                )
            elif sense == ">":
                test_model.addConstr(
                    row + slack >= rhs, name=f"{constr.ConstrName}_relaxed"
                )

            # Minimiser la variable d'écart
            test_model.setObjective(slack_abs_value, GRB.MINIMIZE)

            # Optimiser
            test_model.optimize()

            if test_model.status == GRB.OPTIMAL:
                slack_value = slack.X
                print(
                    f"Contrainte {constr.ConstrName}: modification minimale nécessaire = {abs(slack_value):.6f}"
                )

                if slack_value > 0 and sense in ["=", ">"]:
                    print(
                        f"  → Augmenter le membre de droite d'au moins {slack_value:.6f}"
                    )
                elif slack_value < 0 and sense in ["=", "<"]:
                    print(
                        f"  → Diminuer le membre de droite d'au moins {abs(slack_value):.6f}"
                    )
            else:
                print(
                    f"Contrainte {constr.ConstrName}: impossible de déterminer la modification minimale"
                )

        # Analyse des bornes
        for var in self.iis_bounds:
            var_name = var.VarName

            if var.IISLb:
                # Tester la relaxation de la borne inférieure
                test_model = self.model.copy()
                test_var = test_model.getVarByName(var_name)

                # Sauvegarder l'ancienne borne
                old_lb = test_var.LB

                # Chercher la valeur minimale nécessaire
                test_model.setObjective(test_var, GRB.MINIMIZE)
                test_var.setAttr("LB", -GRB.INFINITY)  # Relâcher temporairement

                # Optimiser
                test_model.optimize()

                if test_model.status == GRB.OPTIMAL:
                    min_value = test_var.X
                    print(f"Variable {var_name}: borne inférieure actuelle = {old_lb}")
                    print(
                        f"  → Diminuer la borne inférieure à au plus {min_value:.6f} (actuellement {old_lb})"
                    )
                else:
                    print(
                        f"Variable {var_name}: impossible de déterminer la borne inférieure minimale"
                    )

            if var.IISUb:
                # Tester la relaxation de la borne supérieure
                test_model = self.model.copy()
                test_var = test_model.getVarByName(var_name)

                # Sauvegarder l'ancienne borne
                old_ub = test_var.UB

                # Chercher la valeur maximale nécessaire
                test_model.setObjective(test_var, GRB.MAXIMIZE)
                test_var.setAttr("UB", GRB.INFINITY)  # Relâcher temporairement

                # Optimiser
                test_model.optimize()

                if test_model.status == GRB.OPTIMAL:
                    max_value = test_var.X
                    print(f"Variable {var_name}: borne supérieure actuelle = {old_ub}")
                    print(
                        f"  → Augmenter la borne supérieure à au moins {max_value:.6f} (actuellement {old_ub})"
                    )
                else:
                    print(
                        f"Variable {var_name}: impossible de déterminer la borne supérieure maximale"
                    )

    def find_feasible_solution_region(self):
        """
        Pour les problèmes avec au plus 2 variables, cette méthode tente de trouver
        une région faisable en relaxant progressivement les contraintes.
        """
        vars_list = list(self.model.getVars())
        if len(vars_list) > 2:
            print(
                "Cette méthode n'est disponible que pour les modèles avec au plus 2 variables."
            )
            return

        if not self.iis_computed:
            self.compute_iis()

        print("\n----- RECHERCHE DE SOLUTION FAISABLE -----")

        # Créer une copie du modèle pour les tests
        relaxed_model = self.model.copy()

        # Relaxer progressivement chaque contrainte de l'IIS
        for i, constr in enumerate(self.iis_constraints):
            constr_name = constr.ConstrName
            test_constr = relaxed_model.getConstrByName(constr_name)

            # Relaxer cette contrainte
            relaxed_model.remove(test_constr)

            # Vérifier si le modèle est maintenant faisable
            relaxed_model.optimize()

            if relaxed_model.status == GRB.OPTIMAL:
                print(
                    f"Le modèle devient faisable en relaxant la contrainte: {constr_name}"
                )

                # Obtenir la solution
                solution = {}
                for var in relaxed_model.getVars():
                    solution[var.VarName] = var.X

                print("Solution faisable trouvée:")
                for var_name, value in solution.items():
                    print(f"  {var_name} = {value:.6f}")

                # Vérifier la contrainte originale
                row = self.model.getRow(constr)
                lhs_value = 0
                for j in range(row.size()):
                    var = row.getVar(j)
                    coef = row.getCoeff(j)
                    lhs_value += coef * solution[var.VarName]

                print(f"Valeur du membre gauche pour {constr_name}: {lhs_value:.6f}")
                print(f"Membre droite pour {constr_name}: {constr.RHS}")
                print(f"Écart: {abs(lhs_value - constr.RHS):.6f}")

                return
            else:
                # Remettre la contrainte si elle n'a pas rendu le modèle faisable
                expr = self.model.getRow(constr)
                if constr.Sense == "=":
                    relaxed_model.addConstr(expr == constr.RHS, name=constr_name)
                elif constr.Sense == "<":
                    relaxed_model.addConstr(expr <= constr.RHS, name=constr_name)
                elif constr.Sense == ">":
                    relaxed_model.addConstr(expr >= constr.RHS, name=constr_name)

        # Tester les bornes des variables
        for var in self.iis_bounds:
            var_name = var.VarName
            test_var = relaxed_model.getVarByName(var_name)

            if var.IISLb:
                # Sauvegarder l'ancienne borne
                old_lb = test_var.LB

                # Relaxer la borne inférieure
                test_var.setAttr("LB", -GRB.INFINITY)

                # Vérifier si le modèle est maintenant faisable
                relaxed_model.optimize()

                if relaxed_model.status == GRB.OPTIMAL:
                    print(
                        f"Le modèle devient faisable en relaxant la borne inférieure de {var_name}"
                    )

                    # Obtenir la solution
                    solution = {}
                    for v in relaxed_model.getVars():
                        solution[v.VarName] = v.X

                    print("Solution faisable trouvée:")
                    for v_name, value in solution.items():
                        print(f"  {v_name} = {value:.6f}")

                    print(
                        f"Valeur minimale nécessaire pour {var_name}: {solution[var_name]:.6f}"
                    )
                    print(f"Borne inférieure actuelle: {old_lb}")
                    print(f"Écart: {abs(solution[var_name] - old_lb):.6f}")

                    return
                else:
                    # Remettre la borne si elle n'a pas rendu le modèle faisable
                    test_var.setAttr("LB", old_lb)

            if var.IISUb:
                # Sauvegarder l'ancienne borne
                old_ub = test_var.UB

                # Relaxer la borne supérieure
                test_var.setAttr("UB", GRB.INFINITY)

                # Vérifier si le modèle est maintenant faisable
                relaxed_model.optimize()

                if relaxed_model.status == GRB.OPTIMAL:
                    print(
                        f"Le modèle devient faisable en relaxant la borne supérieure de {var_name}"
                    )

                    # Obtenir la solution
                    solution = {}
                    for v in relaxed_model.getVars():
                        solution[v.VarName] = v.X

                    print("Solution faisable trouvée:")
                    for v_name, value in solution.items():
                        print(f"  {v_name} = {value:.6f}")

                    print(
                        f"Valeur maximale nécessaire pour {var_name}: {solution[var_name]:.6f}"
                    )
                    print(f"Borne supérieure actuelle: {old_ub}")
                    print(f"Écart: {abs(solution[var_name] - old_ub):.6f}")

                    return
                else:
                    # Remettre la borne si elle n'a pas rendu le modèle faisable
                    test_var.setAttr("UB", old_ub)

        print(
            "Aucune solution faisable trouvée en relaxant une seule contrainte ou borne."
        )
        print(
            "Le problème peut nécessiter la relaxation de plusieurs contraintes simultanément."
        )


# Exemple d'utilisation
def example_infeasible_model():
    """
    Crée un modèle quadratique infaisable simple pour démonstration.
    """
    model = gp.Model("infeasible_quadratic")

    # Créer des variables
    x = model.addVar(lb=0, ub=5, name="x")
    y = model.addVar(lb=0, ub=5, name="y")

    # Ajouter des contraintes incompatibles
    model.addConstr(x + y <= 3, name="c1")
    model.addConstr(2 * x + y >= 8, name="c2")
    model.addConstr(x <= 3, name="c3")
    model.addConstr(y <= 2, name="c4")

    # Objectif quadratique (non convexe)
    model.setObjective(x * x - y * y + x * y, GRB.MINIMIZE)

    return model


def analyze_model(model):
    """
    Analyse un modèle Gurobi pour l'infaisabilité.

    Args:
        model: Un modèle Gurobi
    """
    analyzer = InfeasibilityAnalyzer(model)

    # Vérifier la faisabilité
    is_feasible = analyzer.check_feasibility()

    if is_feasible is False:
        # Calculer l'IIS
        analyzer.compute_iis()

        # Afficher les détails de l'IIS
        analyzer.print_iis()

        # Effectuer une analyse de sensibilité
        analyzer.sensitivity_analysis()

        # Visualiser les contraintes en conflit (pour les modèles 2D)
        try:
            analyzer.visualize_constraints_conflicts()
        except Exception as e:
            print(f"Erreur lors de la visualisation: {e}")

        # Rechercher une solution faisable
        analyzer.find_feasible_solution_region()

        # Créer un modèle relaxé
        relaxed_model = analyzer.relax_model(relax_type="all")

        # Optimiser le modèle relaxé
        if relaxed_model is not None:
            print("\n----- OPTIMISATION DU MODÈLE RELAXÉ -----")
            relaxed_model.optimize()

            if relaxed_model.status == GRB.OPTIMAL:
                print("\nSolution du modèle relaxé:")
                for var in relaxed_model.getVars():
                    if not var.VarName.startswith("slack"):
                        print(f"{var.VarName} = {var.X:.6f}")

                print("\nVariables d'écart (montrant les violations des contraintes):")
                for var in relaxed_model.getVars():
                    if var.VarName.startswith("slack") and abs(var.X) > 1e-6:
                        print(f"{var.VarName} = {var.X:.6f}")
            else:
                print(
                    f"Le modèle relaxé n'est pas optimal. Statut: {relaxed_model.status}"
                )


# Code principal de démonstration
if __name__ == "__main__":
    try:
        # Créer un modèle infaisable d'exemple
        model = example_infeasible_model()

        # Analyser le modèle
        analyze_model(model)

    except gp.GurobiError as e:
        print(f"Erreur Gurobi: {e}")
    except Exception as e:
        print(f"Erreur: {e}")
