from pydantic import BaseModel, validator, model_validator
from typing import List, Optional, Any, Union
from pathlib import Path
import yaml
import torch
from torchvision import transforms
from torch.utils.data import Dataset


from .utils import get_project_path


class MiniDataset(Dataset):
    def __init__(self, x, y):
        self.data = [
            [(x.squeeze(0), y.squeeze(0))]
        ]  # enlever batch dim (1, 784) → (784)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# On veut simuler : dataloader → (label y, list_x) où list_x = [(x, ytrue), ...]
class GroupedByLabelDataset:
    def __init__(self, label_to_data, ytrue):
        self.label_to_data = label_to_data
        self.ytrue = ytrue.squeeze(0)  # (1,) → scalaire

    def __iter__(self):
        for label, xs in self.label_to_data.items():
            list_x = [(x, self.ytrue) for x in xs]
            yield label, list_x

    def __len__(self):
        return len(self.label_to_data)


class DataConfig(BaseModel):
    name: str
    y: int
    x: Union[Any, str]

    @validator("x", pre=True)  # pre = True donc s'exécute avant la validation du type
    def validate_before_x(
        cls, x, values
    ):  # Here values has all the already validated values by order of assignment
        if isinstance(x, (str, Path)):
            path = Path(get_project_path(x.replace("\\", "/")))
            if not path.exists():
                raise ValueError(f"Dataset file not found: {path}")
            examples = torch.load(path, weights_only=False)

            y = values.get("y")
            if y in examples.keys():
                assert examples[y][1] == y

                transform = transforms.ToTensor()
                return transform(examples[y][0].numpy()).view(-1).tolist()
            else:
                raise FileNotFoundError(
                    f"Not example found with label {y} in path : {path}"
                )
        else:
            return x  # x already defined explicitely

    bounds_method: str = "GREAT_BOUNDS"
    ytarget: Optional[int] = None
    L: Optional[List[float]] = None
    U: Optional[List[float]] = None

    @model_validator(mode="after")
    def process_bounds(self) -> "DataConfig":

        if self.bounds_method == "GREAT_BOUNDS":
            n = self.network.n
            K = self.network.K

            L = [[self.L[k]] * n[k] for k in range(K + 1)]
            U = [[self.U[k]] * n[k] for k in range(K + 1)]

            # for j in range(n[0]):
            #     L[0][j] = max(
            #         self.data.L[0], self.data.x[j] - self.certification_problem.epsilon
            #     )
            #     U[0][j] = min(
            #         self.data.U[0], self.data.x[j] + self.certification_problem.epsilon
            #     )

            self.L = L
            self.U = U
        else:
            self.L = None
            self.U = None

        return self

    @model_validator(mode="after")
    def create_dataset(self) -> "DataConfig":
        # Créer une map par label
        print("ytrue in Data Config:", self.y)
        ytrue = torch.tensor([self.y], dtype=torch.int64).unsqueeze(0)
        print("ytrue in Data Config apres tensor operator : ", ytrue)
        x_tensor = torch.tensor(self.x, dtype=torch.float32).unsqueeze(0)
        label_to_data = {int(ytrue.item()): [x_tensor.squeeze(0)]}

        self.dataset = GroupedByLabelDataset(
            label_to_data=label_to_data,
            ytrue=ytrue,
        )

        return self


class DatasetConfig(BaseModel):
    name: str
    path: str

    num_classes: int
    num_samples: int


class MosekSolverConfig(BaseModel):
    certification_model_name: str

    @validator("certification_model_name")
    def validate_sdp_model_name(cls, v, values):
        if v not in ["LanSDP", "MdSDP", "MzbarSDP"]:
            raise ValueError(
                f"SDP model name {v} must be one of 'LanSDP', 'MdSDP', or 'MzbarSDP'."
            )
        return v

    cuts: Optional[List[str]] = []
    all_combinations_cuts: Optional[bool] = False
    RLT_props: Optional[List[float]] = [0.0]

    @validator("RLT_props")
    def validate_rlt_prop(cls, v, values):
        if (
            "cuts" in values
            and values["cuts"] is not None
            and "RLT" in values["cuts"]
            and v is None
        ):
            raise ValueError("RLT cuts are required, but RLT_prop is None.")
        return v

    MATRIX_BY_LAYERS: bool = (
        True  # Whether to use chordal decomposition of variable matrices
    )
    LAST_LAYER: bool = (
        False  # Whether to use the last layer of the network (logits) as variables
    )
    use_fusion: bool = False  # Whether to use the fusion API for MOSEK
    use_callback: bool = False  # Whether to use the callback for MOSEK
    use_active_neurons: Optional[bool] = (
        False  # Whether to use active neurons in the certification problem as variables
    )
    use_inactive_neurons: Optional[bool] = (
        False  # Whether to use inactive neurons in the certification problem as variables
    )
    keep_penultimate_actives : Optional[bool] = False
    @validator("keep_penultimate_actives")
    def validate_keep_penultimate_actives(cls, v, values):
        if v is False and values.get("use_active_neurons"):
            raise ValueError("Withdraw of active neurons on penultimate layer incompatible with use_active_neurons = True")
        return v


    bounds_method: str = "IBP"
    alpha_1: Optional[float] = None  # Lower bound for the McCormick envelope

    @validator("alpha_1")
    def validate_alpha_1(cls, v, values):
        if v is None and values.get("certification_model_name") in [
            "MdSDP",
            "MzbarSDP",
        ]:
            raise ValueError("alpha_1 must be defined for MdSDP and MzbarSDP models.")
        return v

    alpha_2: Optional[float] = None  # Upper bound for the McCormick envelope

    @validator("alpha_2")
    def validate_alpha_2(cls, v, values):
        if v is None and values.get("certification_model_name") in [
            "MdSDP",
            "MzbarSDP",
        ]:
            raise ValueError("alpha_2 must be defined for MdSDP and MzbarSDP models.")
        return v


class GurobiSolverConfig(BaseModel):
    certification_model_name: str

    @validator("certification_model_name")
    def validate_sdp_model_name(cls, v, values):
        if v not in ["LanQuad", "MdQuad", "MzbarQuad", "ClassicLP"]:
            raise ValueError(
                f"Model name {v} must be one of 'LanQuad', 'MdQuad', 'MzbarQuad','ClassicLP'."
            )
        return v

    LAST_LAYER: bool = (
        False  # Whether to use the last layer of the network (logits) as variables
    )
    use_active_neurons: Optional[bool] = (
        False  # Whether to use active neurons in the certification problem as variables
    )
    use_inactive_neurons: Optional[bool] = (
        False  # Whether to use inactive neurons in the certification problem as variables
    )
    bounds_method: str = "IBP"


class NetworkConfig(BaseModel):
    name: str
    path: str
    K: int
    n: List[int]
    dropout: Optional[float] = 0


class ConicBundleConfig(BaseModel):
    filename: str
    McCormick: Optional[str] = "none"


class FullCertificationConfig(BaseModel):
    epsilon: float
    models: Optional[List[Union[MosekSolverConfig, GurobiSolverConfig]]] = None
    data: Union[DataConfig, DatasetConfig]
    network: NetworkConfig
    conic_solver: Optional[ConicBundleConfig] = None


class Adversarial_Network_Training(BaseModel):
    data: str
    train_path: str
    test_path: str
    evaluate_robustness_path: str = None
    num_classes: int
    adversarial_attack: str
    batch_size: int
    num_epochs: int
    lr: float
    epsilon: float
    epsilon_test: Optional[float] = None
    n: List[int]
    K: int
    name_network: str
    compute_bounds_method: Optional[str] = None
    alpha: Optional[float] = None
    steps: Optional[int] = None
    random_start: Optional[bool] = None
    dropout: Optional[float] = 0
