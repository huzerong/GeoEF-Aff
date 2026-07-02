import torch
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from contextlib import nullcontext
import logging
import config

logger = logging.getLogger(__name__)


def is_distributed_ready() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def is_main_process() -> bool:
    return not is_distributed_ready() or torch.distributed.get_rank() == 0


def get_device(model: Module) -> torch.device:
    """获取模型实际运行的设备"""
    if hasattr(model, 'module'):
        return next(model.module.parameters()).device
    return next(model.parameters()).device

def train_model(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    optimizer: Optimizer,
    device: torch.device,
    scaler=None,
    use_bf16: bool = False,
    gradient_accumulation_steps: int = None,
    distributed: bool = False,
) -> float:
    model.train()
    grad_accum_steps = gradient_accumulation_steps or config.GRADIENT_ACCUMULATION_STEPS
    total_loss = 0.0
    total_samples = 0
    processed_batches = 0
    pbar = tqdm(dataloader, desc="Training", disable=distributed and not is_main_process())

    # 获取模型实际设备
    actual_device = get_device(model)

    optimizer.zero_grad()  # Initialize gradients

    all_preds = []
    all_labels = []
    nan_count = 0
    grad_norm = None
    
    for i, batch in enumerate(pbar):
        if batch is None:
            continue

        try:
            postfix_dict = {}

            antibody_seqs = batch["antibody_seq"]
            antigen_seqs = batch["antigen_seq"]
            mutant_antibody_seqs = batch["mutant_antibody_seq"]
            mutant_antigen_seqs = batch["mutant_antigen_seq"]
            foldx_energies = batch["foldx_energy"].to(device)
            foldx_features = batch.get("foldx_features")
            if foldx_features is not None:
                foldx_features = foldx_features.to(device)
            delta_g = batch["delta_g"].to(device)
            structure_data = batch.get("structure_data")
            wt_esm_emb = batch.get("wt_esm_embedding")
            mut_esm_emb = batch.get("mut_esm_embedding")
            mutation_esm_emb = batch.get("mutation_esm_embedding")
            if wt_esm_emb is not None:
                wt_esm_emb = wt_esm_emb.to(device)
            if mut_esm_emb is not None:
                mut_esm_emb = mut_esm_emb.to(device)
            if mutation_esm_emb is not None:
                mutation_esm_emb = mutation_esm_emb.to(device)

            # Skip empty batches (can happen with DataParallel)
            if len(foldx_energies) == 0:
                continue

            if structure_data is not None:
                for key in structure_data:
                    if isinstance(structure_data[key], torch.Tensor):
                        structure_data[key] = structure_data[key].to(device)

            sync_gradients = (
                (processed_batches + 1) % grad_accum_steps == 0
                or (i + 1) == len(dataloader)
            )
            if distributed and hasattr(model, "require_backward_grad_sync"):
                model.require_backward_grad_sync = sync_gradients
            sync_context = nullcontext()

            if scaler is not None:
                with sync_context, torch.amp.autocast(device_type=device.type):
                    pred_delta_g = model(
                        antibody_seqs=antibody_seqs,
                        antigen_seqs=antigen_seqs,
                        mutant_antibody_seqs=mutant_antibody_seqs,
                        mutant_antigen_seqs=mutant_antigen_seqs,
                        foldx_energies=foldx_energies,
                        structure_data=structure_data,
                        wt_esm_embedding=wt_esm_emb,
                        mut_esm_embedding=mut_esm_emb,
                        mutation_esm_embedding=mutation_esm_emb,
                        foldx_features=foldx_features,
                    )
                loss = criterion(pred_delta_g.float(), delta_g.float())
                loss = loss / grad_accum_steps

                # NaN/Inf check — skip batch if loss is bad
                if not torch.isfinite(loss):
                    logger.warning(f"NaN/Inf loss at batch {i}, skipping")
                    optimizer.zero_grad()
                    nan_count += 1
                    continue

                scaler.scale(loss).backward()

                if (processed_batches + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 裁剪L2到1
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            elif use_bf16:
                with sync_context, torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                    pred_delta_g = model(
                        antibody_seqs=antibody_seqs,
                        antigen_seqs=antigen_seqs,
                        mutant_antibody_seqs=mutant_antibody_seqs,
                        mutant_antigen_seqs=mutant_antigen_seqs,
                        foldx_energies=foldx_energies,
                        structure_data=structure_data,
                        wt_esm_embedding=wt_esm_emb,
                        mut_esm_embedding=mut_esm_emb,
                        mutation_esm_embedding=mutation_esm_emb,
                        foldx_features=foldx_features,
                    )
                loss = criterion(pred_delta_g.float(), delta_g.float())
                loss = loss / grad_accum_steps

                if not torch.isfinite(loss):
                    logger.warning(f"NaN/Inf loss at batch {i}, skipping")
                    optimizer.zero_grad()
                    nan_count += 1
                    continue

                loss.backward()

                if (processed_batches + 1) % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 裁剪
                    optimizer.step()
                    optimizer.zero_grad()
            else:
                pred_delta_g = model(
                    antibody_seqs=antibody_seqs,
                    antigen_seqs=antigen_seqs,
                    mutant_antibody_seqs=mutant_antibody_seqs,
                    mutant_antigen_seqs=mutant_antigen_seqs,
                    foldx_energies=foldx_energies,
                    structure_data=structure_data,
                    wt_esm_embedding=wt_esm_emb,
                    mut_esm_embedding=mut_esm_emb,
                    mutation_esm_embedding=mutation_esm_emb,
                    foldx_features=foldx_features,
                )
                loss = criterion(pred_delta_g.float(), delta_g.float())
                loss = loss / grad_accum_steps

                # NaN/Inf check — skip batch if loss is bad
                if not torch.isfinite(loss):
                    logger.warning(f"NaN/Inf loss at batch {i}, skipping")
                    optimizer.zero_grad()
                    nan_count += 1
                    continue

                loss.backward()

                if (processed_batches + 1) % grad_accum_steps == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 裁剪
                    optimizer.step()
                    optimizer.zero_grad()

            # Scale loss back up for logging
            batch_loss = loss.item() * grad_accum_steps
            batch_size = delta_g.numel()
            total_loss += batch_loss * batch_size
            total_samples += batch_size
            processed_batches += 1

            # Collect predictions and labels for running Pearson
            all_preds.append(pred_delta_g.detach().float().cpu().numpy())
            all_labels.append(delta_g.detach().float().cpu().numpy())

        except Exception as e:
            print(f"Error in training batch {i}: {e}")
            import traceback
            traceback.print_exc()
            raise  # 不跳过 batch，直接报错停止

        postfix_dict["loss"] = f"{batch_loss:.4f}"
        if grad_norm is not None:
            postfix_dict["gnorm"] = f"{grad_norm:.2f}"

        # Calculate Pearson every 50 steps
        if (i + 1) % 50 == 0:
            try:
                current_preds = np.concatenate(all_preds)
                current_labels = np.concatenate(all_labels)
                if len(current_preds) > 1:
                    running_pearson, _ = pearsonr(current_labels, current_preds)
                    postfix_dict["pearson"] = f"{running_pearson:.4f}"
            except Exception:
                pass

        if not getattr(pbar, "disable", False):
            pbar.set_postfix(postfix_dict)
    
    # Process any remaining gradients after the loop
    if processed_batches > 0 and processed_batches % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()

    if nan_count > 0:
        logger.warning(f"Skipped {nan_count} batches due to NaN/Inf loss")

    if distributed and hasattr(model, "require_backward_grad_sync"):
        model.require_backward_grad_sync = True

    if distributed and is_distributed_ready():
        stats = torch.tensor([total_loss, float(total_samples)], device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_samples = int(stats[1].item())

    return total_loss / max(total_samples, 1)


def evaluate_model(
    model: Module,
    dataloader: DataLoader,
    criterion: Module,
    device: torch.device,
    distributed: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []

    # 获取模型实际设备（处理DataParallel情况）
    actual_device = get_device(model)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", disable=distributed and not is_main_process()):
            if batch is None:
                continue

            antibody_seqs = batch["antibody_seq"]
            antigen_seqs = batch["antigen_seq"]
            mutant_antibody_seqs = batch["mutant_antibody_seq"]
            mutant_antigen_seqs = batch["mutant_antigen_seq"]
            foldx_energies = batch["foldx_energy"].to(device)
            foldx_features = batch.get("foldx_features")
            if foldx_features is not None:
                foldx_features = foldx_features.to(device)
            delta_g = batch["delta_g"].to(device)
            structure_data = batch.get("structure_data")
            wt_esm_emb = batch.get("wt_esm_embedding")
            mut_esm_emb = batch.get("mut_esm_embedding")
            mutation_esm_emb = batch.get("mutation_esm_embedding")
            if wt_esm_emb is not None:
                wt_esm_emb = wt_esm_emb.to(device)
            if mut_esm_emb is not None:
                mut_esm_emb = mut_esm_emb.to(device)
            if mutation_esm_emb is not None:
                mutation_esm_emb = mutation_esm_emb.to(device)

            # Skip empty batches (can happen with DataParallel)
            if len(foldx_energies) == 0:
                continue

            if structure_data is not None:
                for key in structure_data:
                    if isinstance(structure_data[key], torch.Tensor):
                        structure_data[key] = structure_data[key].to(device)

            pred_delta_g = model(
                antibody_seqs=antibody_seqs,
                antigen_seqs=antigen_seqs,
                mutant_antibody_seqs=mutant_antibody_seqs,
                mutant_antigen_seqs=mutant_antigen_seqs,
                foldx_energies=foldx_energies,
                structure_data=structure_data,
                wt_esm_embedding=wt_esm_emb,
                mut_esm_embedding=mut_esm_emb,
                mutation_esm_embedding=mutation_esm_emb,
                foldx_features=foldx_features,
            )
            loss = criterion(pred_delta_g.float(), delta_g.float())
            batch_size = delta_g.numel()
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(pred_delta_g.float().cpu().numpy())
            all_labels.append(delta_g.float().cpu().numpy())

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    if distributed and is_distributed_ready():
        stats = torch.tensor([total_loss, float(total_samples)], device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        total_loss = stats[0].item()
        total_samples = int(stats[1].item())

        gathered_preds = [None for _ in range(torch.distributed.get_world_size())]
        gathered_labels = [None for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather_object(gathered_preds, all_preds)
        torch.distributed.all_gather_object(gathered_labels, all_labels)
        all_preds = np.concatenate([arr for arr in gathered_preds if arr is not None and len(arr) > 0]) if gathered_preds else np.array([])
        all_labels = np.concatenate([arr for arr in gathered_labels if arr is not None and len(arr) > 0]) if gathered_labels else np.array([])

    avg_loss = total_loss / max(total_samples, 1)

    if len(all_preds) > 1:
        pearson_corr, _ = pearsonr(all_labels, all_preds)
    else:
        pearson_corr = 0.0

    return avg_loss, pearson_corr, all_labels, all_preds
