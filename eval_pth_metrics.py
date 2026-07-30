from collections import OrderedDict

import torch

import config
from model import ESM_FoldX_DDAffinity, ESM_RAAD_FoldX_DDAffinity


def build_model(use_precomputed_esm=None):
    if use_precomputed_esm is None:
        use_precomputed_esm = getattr(config, "USE_PRECOMPUTED_ESM", False)

    if config.USE_DYNAMIC_MODELING:
        return ESM_RAAD_FoldX_DDAffinity(
            esm_model_name=config.ESM_MODEL_NAME,
            hidden_dim=config.HIDDEN_DIM,
            raad_hidden_dim=config.RAAD_HIDDEN_DIM,
            raad_layers=config.RAAD_LAYERS,
            dropout=config.DROPOUT,
            edge_types=config.EDGE_TYPES,
            rball_radius=config.RBALL_RADIUS,
            knn_k=config.KNN_K,
            use_atom_features=config.USE_ATOM_FEATURES,
            use_precomputed_esm=use_precomputed_esm,
            local_radius=getattr(config, "MUTATION_LOCAL_RADIUS", 10.0),
            esm_mutation_window_radius=getattr(config, "ESM_MUTATION_WINDOW_RADIUS", 8),
            esm_local_max_tokens=getattr(config, "ESM_LOCAL_MAX_TOKENS", 32),
            struct_local_max_residues=getattr(config, "STRUCT_LOCAL_MAX_RESIDUES", 32),
            coords_agg=getattr(config, "COORDS_AGG", "mean"),
        )

    return ESM_FoldX_DDAffinity(
        esm_model_name=config.ESM_MODEL_NAME,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT,
        use_precomputed_esm=use_precomputed_esm,
    )


def load_weights(model, ckpt_path: str, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint does not contain a state dict: {ckpt_path}")

    load_errors = []
    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as raw_error:
        load_errors.append(f"raw state_dict failed: {raw_error}")
        stripped_state = OrderedDict()
        prefixed_state = OrderedDict()
        for key, value in checkpoint.items():
            stripped_state[key.replace("module.", "", 1)] = value
            prefixed_state[key if key.startswith("module.") else f"module.{key}"] = value

        try:
            model.load_state_dict(stripped_state)
        except RuntimeError as stripped_error:
            load_errors.append(
                f"module-prefix stripped state_dict failed: {stripped_error}"
            )
            try:
                model.load_state_dict(prefixed_state)
            except RuntimeError as prefixed_error:
                load_errors.append(
                    f"module-prefix added state_dict failed: {prefixed_error}"
                )
                raise RuntimeError(
                    "Checkpoint is incompatible with the local-token32 model. "
                    "Use a checkpoint trained by this experiment; partial "
                    "state-dict loading is intentionally disabled. Attempts: "
                    + " | ".join(load_errors)
                ) from prefixed_error

    model.to(device)
    model.eval()
    return model
