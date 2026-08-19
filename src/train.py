"""
Script d'entraînement, avec intégration MLflow complète :
- Tracking (params, métriques par epoch, artefacts) via un backend SQLite
  local (nécessaire pour utiliser le Model Registry).
- Model logging : le modèle entraîné est sauvegardé au format standard
  MLflow (mlflow.pytorch.log_model), pas juste un fichier .pth brut.
- Model Registry : le modèle est enregistré sous un nom stable
  (ex. "unet-lesion-segmentation"), et promu au statut "Production" s'il
  bat le modèle actuellement en Production pour cette architecture.
  C'est ce que l'application web de démo ira charger plus tard.

Usage :
    python -m src.train --config configs/config.yaml
"""
import argparse
import os

import mlflow
import mlflow.pytorch
import torch
import yaml
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from src.data.dataset import ISICDataset
from src.losses.losses import get_loss
from src.metrics.metrics import SegmentationMetrics
from src.models.deeplabv3 import DeepLabV3
from src.models.unet import UNet

# Chemins ABSOLUS vers Google Drive, volontairement — pas de chemin
# relatif comme "sqlite:///mlflow.db". Un chemin relatif dépend du
# dossier courant au moment de l'exécution : si le script est lancé par
# erreur depuis /content (disque local éphémère de Colab) plutôt que
# /content/drive/MyDrive/repo_clone (Drive, persistant), tout le suivi
# MLflow serait perdu à la fin de la session sans aucun message d'erreur.
# Avec un chemin absolu, ça écrit toujours au bon endroit sur Drive,
# peu importe d'où le script est appelé.
_REPO_ROOT = "/content/drive/MyDrive/repo_clone"
MLFLOW_TRACKING_URI = f"sqlite:////{_REPO_ROOT.lstrip('/')}/mlflow.db"
MLFLOW_EXPERIMENT_NAME = "segmentation-ablation"
CLASS_NAMES = ["fond", "lesion"]


def flatten_config(cfg: dict, parent_key: str = "") -> dict:
    """Aplati un dict imbriqué pour mlflow.log_params (qui attend des
    paires clé/valeur simples, pas de structure imbriquée)."""
    flat = {}
    for k, v in cfg.items():
        key = f"{parent_key}.{k}" if parent_key else k
        if isinstance(v, dict):
            flat.update(flatten_config(v, key))
        else:
            flat[key] = v
    return flat


def build_model(cfg: dict) -> torch.nn.Module:
    architecture = cfg["model"]["architecture"]
    num_classes = cfg["model"]["num_classes"]
    if architecture == "unet":
        return UNet(in_channels=3, num_classes=num_classes, base_filters=cfg["model"].get("base_filters", 32))
    if architecture == "deeplabv3":
        return DeepLabV3(in_channels=3, num_classes=num_classes)
    raise ValueError(f"architecture inconnue : {architecture!r} (attendu : unet | deeplabv3)")


def get_or_create_experiment(name: str) -> str:
    """Récupère l'ID de l'expérience MLflow, la crée si besoin avec un
    emplacement d'artefacts explicite (persistant sur Drive)."""
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is not None:
        return experiment.experiment_id
    return mlflow.create_experiment(name, artifact_location=f"file:{_REPO_ROOT}/mlruns")


def run_epoch(model, loader, loss_fn, metrics, device, optimizer=None) -> dict:
    """Une passe sur le dataset. Si `optimizer` est fourni : mode train
    (backward + step). Sinon : mode évaluation (no_grad)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    metrics.reset()

    total_loss = 0.0
    n_batches = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)

            logits = model(images)
            loss = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            metrics.update(logits.detach(), masks)
            total_loss += loss.item()
            n_batches += 1

    results = metrics.compute()
    results["loss"] = total_loss / n_batches
    return results


def maybe_promote_to_production(client: MlflowClient, model_name: str, version: str, metric_value: float) -> None:
    """Promeut la nouvelle version au statut Production si elle bat la
    version Production actuelle (ou s'il n'y en a pas encore)."""
    production_versions = client.get_latest_versions(model_name, stages=["Production"])

    should_promote = True
    if production_versions:
        current_prod = production_versions[0]
        current_metric = float(current_prod.tags.get("mean_dice", -1))
        should_promote = metric_value > current_metric
        if not should_promote:
            print(f"Version {version} ({metric_value:.4f}) ne bat pas la Production actuelle "
                  f"({current_prod.version}, {current_metric:.4f}) — pas de promotion.")

    if should_promote:
        client.transition_model_version_stage(
            name=model_name, version=version, stage="Production", archive_existing_versions=True,
        )
        print(f"✅ {model_name} v{version} promu en Production (mean_dice={metric_value:.4f}).")


def train_from_config(cfg: dict) -> dict:
    """Exécute un run d'entraînement complet à partir d'un dict de config
    déjà chargé (permet d'être appelé en boucle depuis run_ablation.py,
    sans repasser par un fichier .yaml à chaque run)."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # --- Données ---
    data_cfg = cfg["data"]
    train_ds = ISICDataset(data_cfg["images_dir"], data_cfg["masks_dir"], split="train",
                            image_size=data_cfg["image_size"])
    val_ds = ISICDataset(data_cfg["images_dir"], data_cfg["masks_dir"], split="val",
                          image_size=data_cfg["image_size"])

    batch_size = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    # --- Modèle, loss, optimiseur ---
    model = build_model(cfg).to(device)
    loss_fn = get_loss(cfg["loss"]["type"], num_classes=cfg["model"]["num_classes"],
                        combo_weights=tuple(cfg["loss"].get("combo_weights", (0.5, 0.5))))
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])

    train_metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"], class_names=CLASS_NAMES)
    val_metrics = SegmentationMetrics(num_classes=cfg["model"]["num_classes"], class_names=CLASS_NAMES)

    # --- MLflow : tracking + registry ---
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_id = get_or_create_experiment(MLFLOW_EXPERIMENT_NAME)
    client = MlflowClient()

    run_name = cfg["run_name"]
    model_name = f"{cfg['model']['architecture']}-lesion-segmentation"

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name):
        mlflow.log_params(flatten_config(cfg))

        best_val_dice = -1.0
        best_state_dict = None
        epochs_without_improvement = 0
        patience = cfg["training"]["early_stopping_patience"]

        for epoch in range(cfg["training"]["epochs"]):
            train_results = run_epoch(model, train_loader, loss_fn, train_metrics, device, optimizer)
            val_results = run_epoch(model, val_loader, loss_fn, val_metrics, device, optimizer=None)

            mlflow.log_metrics({f"train_{k}": v for k, v in train_results.items()}, step=epoch)
            mlflow.log_metrics({f"val_{k}": v for k, v in val_results.items()}, step=epoch)

            print(f"Epoch {epoch+1}/{cfg['training']['epochs']} — "
                  f"train_loss={train_results['loss']:.4f} val_loss={val_results['loss']:.4f} "
                  f"val_mean_dice={val_results['mean_dice']:.4f}")

            if val_results["mean_dice"] > best_val_dice:
                best_val_dice = val_results["mean_dice"]
                best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping à l'epoch {epoch+1} (patience={patience}).")
                    break

        # Recharge les meilleurs poids avant de logger le modèle final
        model.load_state_dict(best_state_dict)
        mlflow.log_metric("best_val_mean_dice", best_val_dice)

        # --- Model logging + Model Registry ---
        # serialization_format="pickle" (plutôt que le défaut "pt2", qui
        # trace le graphe et exige une signature stricte de type TensorSpec)
        # : plus simple, plus robuste, et suffisant pour notre usage
        # (rechargement du modèle dans l'appli de démo).
        input_example = next(iter(val_loader))[0][:1].cpu().numpy()
        model_info = mlflow.pytorch.log_model(
            model, name="model", input_example=input_example, serialization_format="pickle",
        )
        registered = mlflow.register_model(model_info.model_uri, model_name)

        # Attache la métrique en tag pour permettre les comparaisons futures
        client.set_model_version_tag(model_name, registered.version, "mean_dice", str(best_val_dice))
        client.set_model_version_tag(model_name, registered.version, "run_name", run_name)

        maybe_promote_to_production(client, model_name, registered.version, best_val_dice)

    print(f"\nTerminé. Meilleur val_mean_dice : {best_val_dice:.4f}")
    print(f"Modèle enregistré : {model_name} (version {registered.version})")

    return {
        "run_name": run_name,
        "architecture": cfg["model"]["architecture"],
        "loss": cfg["loss"]["type"],
        "best_val_mean_dice": best_val_dice,
        "model_version": registered.version,
    }


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train_from_config(cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)