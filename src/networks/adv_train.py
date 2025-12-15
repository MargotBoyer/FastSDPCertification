import wandb
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml
from pydantic import ValidationError
import gc

from adversarial_attacks import (
    PGDAttack,
    LPAttack,
    SDPAttack,
    # LP_Attack_Optimized,
    # LPAttack2,
    LPAttack3Parallel,
    CrownIBP_Attack,
)
from tools import Adversarial_Network_Training, get_project_path


def evaluate_clean(model, testloader, device):
    """Évaluation sur données propres"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def evaluate_robust(model, testloader, device, pgd_config, max_batches=None):
    """Évaluation de la robustesse avec PGD"""
    model.eval()

    # Attaque PGD pour l'évaluation (plus forte)
    eval_pgd_config = pgd_config.copy()
    eval_pgd_config["steps"] = 20  # Plus d'étapes pour l'évaluation

    pgd_attack = PGDAttack(
        model=model,
        eps=eval_pgd_config["eps"],
        alpha=eval_pgd_config["alpha"],
        steps=eval_pgd_config["steps"],
        random_start=eval_pgd_config["random_start"],
        norm=eval_pgd_config["norm"],
    )

    correct = 0
    total = 0

    for batch_idx, (inputs, labels) in enumerate(testloader):
        # if batch_idx != 8 and batch_idx != 9:
        #     continue

        if max_batches and batch_idx >= max_batches:
            break
        print("Evaluating batch ", batch_idx)
        inputs, labels = inputs.to(device), labels.to(device)

        #print("STUDYY : batch_index =", batch_idx)
        # inputs = inputs[1:6]
        # labels = labels[1:6]
        # Générer exemples adversariaux
        with torch.enable_grad():
            adv_inputs = pgd_attack.forward(inputs, labels)

        # Évaluer sur exemples adversariaux
        with torch.no_grad():
            outputs = model(adv_inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            #print(f"STUDY : predicted : {predicted}, label = {labels}")
            correct += (predicted == labels).sum().item()

    return correct / total


# def simple_adversarial_training_loop(
#     model,
#     trainloader,
#     testloader,
#     device,
#     num_epochs,
#     lr=0.001,
#     eps=8 / 255,
#     adversarial_attack: str = "PGD",
#     lambda_: float = 1,
# ):
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)
#     criterion = nn.CrossEntropyLoss()
#     print("Adversarial training with attack: ", adversarial_attack)

#     if adversarial_attack == "PGD":
#         attack = PGDAttack(
#             model=model,
#             eps=eps,
#             alpha=eps / 10,
#             steps=40,
#             random_start=True,
#             norm="inf",
#         )

#     elif adversarial_attack == "LP":
#         attack = LPAttack(
#             model=model,
#             num_classes=model.n[-1],
#             eps=eps,
#             targeted=False,
#             norm="inf",
#         )
#     elif adversarial_attack == "SDP":
#         attack = SDPAttack(
#             model=model,
#             num_classes=model.n[-1],
#             eps=eps,
#             norm="inf",
#         )

#     model.to(device)
#     for epoch in range(num_epochs):
#         print("Training epoch: ", epoch)
#         model.train()
#         running_loss = 0.0

#         for inputs, labels in tqdm(trainloader):
#             inputs, labels = inputs.to(device), labels.to(device)

#             optimizer.zero_grad()

#             # Générer exemples adversariaux
#             model.eval()  # Important: mode eval pour PGD

#             if adversarial_attack == "PGD":
#                 with torch.enable_grad():
#                     adv_inputs = attack.forward(inputs, labels)

#                 model.train()  # Retour en mode train
#                 outputs = model(adv_inputs)
#                 loss = criterion(outputs, labels)

#             elif adversarial_attack == "LP" or adversarial_attack == "SDP":

#                 model.extract_weights()
#                 # outputs = model(inputs)
#                 # base_loss = criterion(outputs, labels)

#                 if adversarial_attack == "LP":
#                     loss_pertubation, adv_inputs = attack.forward(inputs, labels).requires_grad_(True)
#                     print("Adversarial inputs shape: ", adv_inputs.shape)

#                 model.train()
#                 #outputs = model(adv_inputs)
#                 loss = criterion(inputs, labels) + lambda_ * loss_pertubation

#                 print("Inputs shape: ", inputs.shape)
#                 print("Labels shape: ", labels.shape)
#                 print("Loss on adversarial examples with LP: ", loss.item())

#             loss.backward()
#             optimizer.step()

#             running_loss += loss.item()

#         if epoch % 10 == 0:
#             avg_loss = running_loss / len(trainloader)
#             print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

#             # Évaluation clean
#             clean_acc = evaluate_clean(model, testloader, device)
#             print(f"Clean Accuracy: {clean_acc:.2%}")

#             # Évaluation robuste (optionnel, plus lent)
#             # robust_acc = evaluate_robust(model, testloader, device,
#             #                            {'eps': eps, 'alpha': eps/4, 'steps': 20,
#             #                             'random_start': True, 'norm': 'inf'})
#             # print(f"Robust Accuracy: {robust_acc:.2%}")


def complex_adversarial_training_loop(
    model,
    trainloader,
    testloader,
    robust_datas_loader,
    device,
    adversarial_attack: str = "PGD",
    lambda_: float = 1,
    project_name: str = "adversarial-training",
    experiment_name: str = None,
    log_frequency: int = 1,  # Log tous les N epochs
    use_wandb: bool = False,
    epsilon_to_test: float = None,
    **kwargs,
):
    """
    Entraînement adversarial avec logging W&B

    Args:
        use_wandb: Activer/désactiver W&B
        project_name: Nom du projet W&B
        experiment_name: Nom de l'expérience
        log_frequency: Fréquence de logging (epochs)
    """
    yaml_file = kwargs.get("yaml_file", "adversarial_network_training.yaml")
    print("KWARGS  : ", kwargs)
    lr = kwargs.get("lr", 0.001)
    eps = kwargs.get("epsilon")
    num_epochs = kwargs.get("num_epochs", 100)
    print("lambda : ", lambda_)
    # Configuration W&B
    if use_wandb:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                "learning_rate": lr,
                "epsilon": eps,
                "num_epochs": num_epochs,
                "adversarial_attack": adversarial_attack,
                "lambda": lambda_,
                "model_architecture": str(model),
                "batch_size": trainloader.batch_size,
            },
            save_code=False,
            dir="wandb",
            mode="online",
            settings=wandb.Settings(_disable_stats=True),
        )
        # Optionnel: surveiller les gradients et paramètres
        # wandb.watch(model, log="gradients", log_freq=500)  # gradients plutot que all
        # wandb.watch(model, log="none")

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, factor=0.5, patience=5
    # )
    criterion = nn.CrossEntropyLoss()
    print("Adversarial training with attack: ", adversarial_attack)

    # Configuration des attaques
    if adversarial_attack == "PGD":
        alpha = kwargs.get("alpha")
        steps = kwargs.get("steps")
        random_start = kwargs.get("random_start")
        attack = PGDAttack(
            model=model,
            eps=eps,
            alpha=alpha,
            steps=steps,
            random_start=random_start,
            norm="inf",
        )
        print("PGD Attack configured with eps:", eps, "alpha:", alpha, "steps:", steps)
        print(attack)

    elif adversarial_attack == "LP":
        attack = LPAttack3Parallel(
            model=model,
            num_classes=model.n[-1],
            eps=eps,
            targeted=False,
            norm="inf",
            compute_bounds_method=kwargs.get("compute_bounds_method", "alpha-CROWN"),
        )
    elif adversarial_attack == "SDP":
        attack = SDPAttack(
            model=model,
            num_classes=model.n[-1],
            eps=eps,
            norm="inf",
        )

    elif adversarial_attack == "CROWN-IBP":
        attack = CrownIBP_Attack(
            model=model,
            shape=kwargs.get("shape", (1, 1, 28, 28)),  # Exemple pour MNIST
            device=device,
            epsilon=eps,
            kappa=1.0,
            criterion=criterion,
        )

    # Listes pour stocker les métriques (backup si pas de W&B)
    train_losses = []
    clean_accuracies = []
    robust_accuracies_EPS_0_3 = []
    robust_accuracies_EPS_0_1 = []
    robust_accuracies_EPS_0_0_1 = []
    robust_accuracies_EPS_to_test = []
    epochs_logged = []

    model.to(device)

    for epoch in range(num_epochs):
        print(f"Training epoch: {epoch}")
        model.train()
        running_loss = 0.0
        batch_losses = []

        # Barre de progression avec tqdm
        pbar = tqdm(trainloader, desc=f"Epoch {epoch}/{num_epochs}")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            print(f"    Processing batch {batch_idx + 1}/{len(trainloader)}")
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # Générer exemples adversariaux
            model.eval()  # Important: mode eval pour PGD

            if adversarial_attack == "PGD":
                with torch.enable_grad():
                    adv_inputs = attack.forward(inputs, labels)
                model.train()  # Retour en mode train

                # Entraînement sur exemples adversariaux
                outputs = model(adv_inputs)
                loss = criterion(outputs, labels)

            elif adversarial_attack == "LP" or adversarial_attack == "SDP":
                # W, b = model.extract_weights()

                if adversarial_attack == "LP":
                    W, b = model.extract_weights()
                    loss_pertubation, adv_inputs = attack(
                        inputs, labels, model.K, model.n, W, b
                    )
                    print("Adversarial inputs shape: ", adv_inputs.shape)

                loss_pertubation = torch.tensor(
                    loss_pertubation,
                    dtype=torch.float32,
                    requires_grad=True,
                    device=device,
                )

                model.train()
                # Using optimal values of LP or SDP as losses
                print("Inputs shape: ", inputs.shape)
                print("Labels shape: ", labels.shape)
                outputs = model(inputs)
                print("Outputs shape: ", outputs.shape)
                loss = criterion(outputs, labels) + lambda_ * loss_pertubation

                # Using solutions of LP or SDP as adversarial inputs
                # outputs = model(adv_inputs.requires_grad_(True))
                # loss = criterion(outputs, torch.repeat_interleave(labels, model.n[model.K] - 1))

            elif adversarial_attack == "CROWN-IBP":
                # CROWN-IBP attack
                loss = attack.compute_crown_ibp_loss(inputs, labels)
                print("CROWN-IBP loss:", loss.item())

            loss.backward()
            optimizer.step()
            # scheduler.step()

            batch_loss = loss.item()
            running_loss += batch_loss
            batch_losses.append(batch_loss)

            # Mise à jour de la barre de progression
            pbar.set_postfix({"Loss": f"{batch_loss:.4f}"})

            # Log par batch (optionnel, pour un suivi très détaillé)
            # if use_wandb and batch_idx % 5 == 0:  # Log tous les 50 batches
            #     wandb.log(
            #         {"batch_loss": batch_loss, "epoch": epoch, "batch": batch_idx}
            #     )

        # Métriques d'époque
        avg_train_loss = running_loss / len(trainloader)
        train_losses.append(avg_train_loss)

        # Évaluation périodique
        if epoch % log_frequency == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}, Loss: {avg_train_loss:.4f}")

            # Évaluation clean
            model.eval()
            clean_acc = evaluate_clean(model, testloader, device)
            clean_accuracies.append(clean_acc)
            print(f"Clean Accuracy: {clean_acc:.2%}")

            # Évaluation robuste (optionnel)
            robust_acc = None
            if epoch % (log_frequency * 2) == 0:  # Moins fréquent car plus lent
                robust_acc = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": eps,
                        "alpha": eps / 4,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_0_3.append(robust_acc)
                print(f"Robust Accuracy EPS=0.3: {robust_acc:.2%}")

                robust_acc_eps_0_1 = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": 0.1,
                        "alpha": 0.01,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_0_1.append(robust_acc_eps_0_1)

                robust_acc_eps_0_0_1 = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": 0.01,
                        "alpha": 0.001,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_0_0_1.append(robust_acc_eps_0_0_1)

                robust_acc_eps_to_test = evaluate_robust(
                    model,
                    robust_datas_loader,
                    device,
                    {
                        "eps": epsilon_to_test,
                        "alpha": 0.001,
                        "steps": 20,
                        "random_start": True,
                        "norm": "inf",
                    },
                )
                robust_accuracies_EPS_to_test.append(robust_acc_eps_to_test)

            epochs_logged.append(epoch)

            # Logging W&B
            if use_wandb:
                log_dict = {
                    "train_loss": avg_train_loss,
                    "clean_accuracy": clean_acc,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }

                if robust_acc is not None:
                    log_dict[f"robust_accuracy_epsilon={eps}"] = robust_acc
                    log_dict["robust_accuracy_epsilon=0.1"] = robust_acc_eps_0_1
                    log_dict["robust_accuracy_epsilon=0.01"] = robust_acc_eps_0_0_1
                    log_dict[f"robust_accuracy_epsilon={epsilon_to_test}"] = (
                        robust_acc_eps_to_test
                    )

                wandb.log(log_dict)

                # # Log histogramme des pertes par batch
                # wandb.log({"batch_losses_histogram": wandb.Histogram(batch_losses)})

        # Libérer la mémoire GPU/RAM à chaque epoch
        torch.cuda.empty_cache()
        gc.collect()

    # Graphiques finaux (sauvegarde locale)
    if not use_wandb:
        plot_training_curves(
            epochs_logged, train_losses, clean_accuracies, robust_accuracies_EPS_0_3
        )

    # Fermeture W&B
    if use_wandb:
        wandb.finish()

    return {
        "train_losses": train_losses,
        "clean_accuracies": clean_accuracies,
        "robust_accuracies": robust_accuracies_EPS_0_3,
        "epochs": epochs_logged,
    }


def plot_training_curves(epochs, train_losses, clean_accs, robust_accs):
    """Génère des graphiques de training curves en local"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Loss curve
    ax1.plot(epochs, train_losses, "b-", label="Training Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss")
    ax1.legend()
    ax1.grid(True)

    # Accuracy curves
    ax2.plot(epochs, clean_accs, "g-", label="Clean Accuracy")
    if robust_accs:
        robust_epochs = epochs[::2]  # Moins fréquent
        ax2.plot(robust_epochs, robust_accs, "r-", label="Robust Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Model Accuracy")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300, bbox_inches="tight")
    plt.show()


def load_adversarial_training_config(config_path):
    """
    Charge la configuration d'entraînement adversarial depuis un fichier YAML.

    Args:
        config_path (str): Chemin vers le fichier de configuration YAML.

    Returns:
        dict: Configuration chargée.
    """

    print("PATH CONFIG : ", get_project_path(config_path))

    with open(get_project_path(config_path), "r") as file:
        raw_config = yaml.safe_load(file)

    try:
        validated_config = Adversarial_Network_Training(**raw_config)
    except ValidationError as e:
        print(f"Erreur de validation du fichier YAML :\n{e}")
        print("raw config:", raw_config)
        raise

    return validated_config.dict()
