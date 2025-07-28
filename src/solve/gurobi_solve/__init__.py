import os
import logging
from .gurobi_generic_solver import GurobiSolver
from .quadmodels.Lan_quad import LanQuad
from .quadmodels.Md_quad import MdQuad
from .quadmodels.Mzbar_quad import MzbarQuad
from .lpmodels.LP_attack import ClassicLP
from tools.utils import get_project_path


logger_gurobi = logging.getLogger("Gurobi_logger")
logger_gurobi.setLevel(logging.DEBUG)
handler = logging.FileHandler(get_project_path("results/Gurobi_logger.log"))
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger_gurobi.addHandler(handler)
logger_gurobi.disabled = True


__all__ = ["GurobiSolver", "LanQuad", "MdQuad", "MzbarQuad", "ClassicLP"]
