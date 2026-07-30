import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from torch.nn import Module
from torch.optim import Optimizer
from contextlib import nullcontext
import logging
import config

logger = logging.getLogger(__name__)


class BeneficialGroupRankLoss(Module):
    requires_group_ids = True

    def __init__(
        self,
        smooth_l1_beta: float = 1.0,
        beneficial_threshold: float = 0.0,
        beneficial_sample_weight: float = 2.0,
        pairwise_weight: float = 0.25,
        site_pair_weight: float = 1.0,
        complex_pair_weight: float = 0.25,
        pairwise_temperature: float = 0.5,
        pairwise_min_label_gap: float = 0.2,
        pairwise_max_pairs: int = 4096,
    ):
        super().__init__()
        self.smooth_l1_beta = smooth_l1_beta
        self.beneficial_threshold = beneficial_threshold
        self.beneficial_sample_weight = beneficial_sample_weight
        self.pairwise_weight = pairwise_weight
        self.site_pair_weight = site_pair_weight
        self.complex_pair_weight = complex_pair_weight
        self.pairwise_temperature = pairwise_temperature
        self.pairwise_min_label_gap = pairwise_min_label_gap
        self.pairwise_max_pairs = pairwise_max_pairs

    def beneficial_huber_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.reshape(-1).float()
        target = target.reshape(-1).float()
        per_sample = F.smooth_l1_loss(
            pred,
            target,
            beta=self.smooth_l1_beta,
            reduction="none",
        )
        weight = torch.ones_like(target)
        weight = torch.where(
            target < float(self.beneficial_threshold),
            weight.new_full(weight.shape, float(self.beneficial_sample_weight)),
            weight,
        )
        return (weight * per_sample).sum() / weight.sum().clamp_min(1.0)

    @staticmethod
    def _encode_group_ids(group_ids, count: int, device: torch.device) -> torch.Tensor:
        if group_ids is None:
            return torch.arange(count, dtype=torch.long, device=device)
        if isinstance(group_ids, torch.Tensor):
            values = group_ids.detach().cpu().reshape(-1).tolist()
        else:
            values = list(group_ids)
        if len(values) != count:
            raise ValueError(f"Expected {count} group ids, got {len(values)}.")
        mapping = {}
        encoded = []
        for value in values:
            key = str(value)
            if key not in mapping:
                mapping[key] = len(mapping)
            encoded.append(mapping[key])
        return torch.tensor(encoded, dtype=torch.long, device=device)

    def group_pairwise_ranking_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        complex_ids=None,
        site_ids=None,
    ) -> torch.Tensor:
        pred = pred.reshape(-1).float()
        target = target.reshape(-1).float()
        if int(target.numel()) < 2:
            return pred.new_tensor(0.0)

        complex_codes = self._encode_group_ids(complex_ids, target.numel(), pred.device)
        site_codes = self._encode_group_ids(site_ids, target.numel(), pred.device)
        same_complex = complex_codes.unsqueeze(1) == complex_codes.unsqueeze(0)
        same_site = site_codes.unsqueeze(1) == site_codes.unsqueeze(0)

        pred_diff = pred.unsqueeze(1) - pred.unsqueeze(0)
        target_diff = target.unsqueeze(1) - target.unsqueeze(0)
        pair_mask = same_complex & (
            target_diff.abs() > float(self.pairwise_min_label_gap)
        )
        upper_mask = torch.triu(torch.ones_like(pair_mask, dtype=torch.bool), diagonal=1)
        pair_idx = torch.nonzero(pair_mask & upper_mask, as_tuple=False)
        if pair_idx.numel() == 0:
            return pred.new_tensor(0.0)

        if self.pairwise_max_pairs and pair_idx.shape[0] > self.pairwise_max_pairs:
            if self.training:
                selected = torch.randperm(
                    pair_idx.shape[0],
                    device=pair_idx.device,
                )[: self.pairwise_max_pairs]
            else:
                selected = torch.linspace(
                    0,
                    pair_idx.shape[0] - 1,
                    steps=self.pairwise_max_pairs,
                    device=pair_idx.device,
                ).round().long()
            pair_idx = pair_idx[selected]

        i = pair_idx[:, 0]
        j = pair_idx[:, 1]
        selected_pred_diff = pred_diff[i, j]
        selected_target_diff = target_diff[i, j]
        sign = torch.sign(selected_target_diff)
        label_gap_weight = selected_target_diff.abs().clamp(min=0.5, max=3.0)
        group_weight = torch.where(
            same_site[i, j],
            label_gap_weight.new_full(
                label_gap_weight.shape,
                float(self.site_pair_weight),
            ),
            label_gap_weight.new_full(
                label_gap_weight.shape,
                float(self.complex_pair_weight),
            ),
        )
        temperature = max(float(self.pairwise_temperature), 1e-6)
        loss = F.softplus(-sign * selected_pred_diff / temperature) * temperature
        pair_weight = label_gap_weight * group_weight
        return (pair_weight * loss).sum() / pair_weight.sum().clamp_min(1e-8)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        complex_ids=None,
        site_ids=None,
    ) -> torch.Tensor:
        regression = self.beneficial_huber_loss(pred, target)
        pairwise = self.group_pairwise_ranking_loss(
            pred,
            target,
            complex_ids=complex_ids,
            site_ids=site_ids,
        )
        return regression + self.pairwise_weight * pairwise


def compute_training_loss(
    criterion: Module,
    preds: torch.Tensor,
    labels: torch.Tensor,
    batch=None,
) -> torch.Tensor:
    if getattr(criterion, "requires_group_ids", False):
        return criterion(
            preds.float(),
            labels.float(),
            complex_ids=(batch or {}).get("complex_id"),
            site_ids=(batch or {}).get("mutation_site_id"),
        )
    return criterion(preds.float(), labels.float())


def _normalize_group_ids(group_ids, expected_count: int):
    if group_ids is None:
        raise ValueError("Group ids are required.")
    if isinstance(group_ids, torch.Tensor):
        values = group_ids.detach().cpu().reshape(-1).tolist()
    else:
        values = list(group_ids)
    if len(values) != expected_count:
        raise ValueError(
            f"Expected {expected_count} group ids, got {len(values)}."
        )
    return [str(value) for value in values]


def _batch_pair_statistics(
    labels: torch.Tensor,
    complex_ids,
    site_ids,
    min_label_gap: float,
):
    labels = labels.detach().float().cpu().reshape(-1)
    count = int(labels.numel())
    complex_values = _normalize_group_ids(complex_ids, count)
    site_values = _normalize_group_ids(site_ids, count)
    complex_codes = BeneficialGroupRankLoss._encode_group_ids(
        complex_values,
        count,
        labels.device,
    )
    site_codes = BeneficialGroupRankLoss._encode_group_ids(
        site_values,
        count,
        labels.device,
    )
    upper = torch.triu(
        torch.ones(count, count, dtype=torch.bool),
        diagonal=1,
    )
    same_complex = (
        complex_codes.unsqueeze(1) == complex_codes.unsqueeze(0)
    ) & upper
    same_site = (site_codes.unsqueeze(1) == site_codes.unsqueeze(0)) & upper
    label_gap = (
        labels.unsqueeze(1) - labels.unsqueeze(0)
    ).abs() > float(min_label_gap)
    return {
        "same_complex_pairs": int(same_complex.sum().item()),
        "same_site_pairs": int(same_site.sum().item()),
        "valid_complex_pairs": int((same_complex & label_gap).sum().item()),
        "valid_site_pairs": int((same_site & label_gap).sum().item()),
    }


def audit_training_dataloader(
    dataloader: DataLoader,
    device: torch.device,
    distributed: bool = False,
    beneficial_threshold: float = 0.0,
    pairwise_min_label_gap: float = 0.2,
    max_zero_same_site_batch_rate: float = 0.2,
):
    """Audit labels, grouping, mutation alignment, and local-token coverage."""
    # samples, beneficial, batches, complex pairs, site pairs, valid complex,
    # valid site, zero valid rank batches, zero site batches,
    # zero valid site batches, mutation CA nodes, requested mutation sites,
    # matched mutation sites, samples without a match, missing mutation sites,
    # WT mismatch sites
    stats = torch.zeros(16, dtype=torch.float64, device=device)
    max_sequence_tokens = int(getattr(config, "ESM_LOCAL_MAX_TOKENS", 32))
    max_structure_tokens = int(
        getattr(config, "STRUCT_LOCAL_MAX_RESIDUES", 32)
    )
    sequence_token_histogram = torch.zeros(
        max_sequence_tokens + 1,
        dtype=torch.float64,
        device=device,
    )
    structure_token_histogram = torch.zeros(
        max_structure_tokens + 1,
        dtype=torch.float64,
        device=device,
    )
    residues_without_ca = torch.zeros(
        1,
        dtype=torch.float64,
        device=device,
    )
    progress = tqdm(
        dataloader,
        desc="Training audit",
        disable=distributed and not is_main_process(),
    )

    for batch in progress:
        labels = batch["delta_g"].detach().float().cpu().reshape(-1)
        if not bool(torch.isfinite(labels).all()):
            bad_positions = torch.nonzero(
                ~torch.isfinite(labels),
                as_tuple=True,
            )[0].tolist()
            raise ValueError(
                "Training audit found non-finite delta_g values at batch "
                f"positions {bad_positions}."
            )
        sample_count = int(labels.numel())
        pair_stats = _batch_pair_statistics(
            labels,
            batch.get("complex_id"),
            batch.get("mutation_site_id"),
            min_label_gap=pairwise_min_label_gap,
        )
        structure_data = batch.get("structure_data")
        if structure_data is None:
            raise ValueError("Training audit requires structure_data.")
        mutation_ca_mask = structure_data.get("mutation_ca_mask")
        requested_site_counts = structure_data.get("mutation_site_count")
        matched_site_counts = structure_data.get("matched_mutation_site_count")
        mismatch_counts = structure_data.get("mutation_wt_mismatch_count")
        if not isinstance(mutation_ca_mask, torch.Tensor):
            raise ValueError("Training audit requires mutation_ca_mask.")
        if not isinstance(requested_site_counts, torch.Tensor):
            raise ValueError("Training audit requires mutation_site_count.")
        if not isinstance(matched_site_counts, torch.Tensor):
            raise ValueError(
                "Training audit requires matched_mutation_site_count."
            )
        if not isinstance(mismatch_counts, torch.Tensor):
            raise ValueError(
                "Training audit requires mutation_wt_mismatch_count."
            )
        sequence_padding_mask = batch.get("esm_window_padding_mask")
        if not isinstance(sequence_padding_mask, torch.Tensor):
            raise ValueError(
                "Training audit requires esm_window_padding_mask."
            )
        if tuple(sequence_padding_mask.shape) != (
            sample_count,
            max_sequence_tokens,
        ):
            raise ValueError(
                "esm_window_padding_mask has an unexpected shape: "
                f"{tuple(sequence_padding_mask.shape)}."
            )
        sequence_token_counts = (
            ~sequence_padding_mask.detach().cpu().bool()
        ).sum(dim=1).long()
        if bool((sequence_token_counts < 1).any()):
            raise ValueError("Every sample must have at least one local ESM token.")
        sequence_token_histogram += torch.bincount(
            sequence_token_counts.clamp_max(max_sequence_tokens),
            minlength=max_sequence_tokens + 1,
        ).to(device=device, dtype=torch.float64)

        residue_uid = structure_data.get("residue_uid")
        batch_ids = structure_data.get("batch_ids")
        ca_mask = structure_data.get("ca_mask")
        if not isinstance(residue_uid, torch.Tensor):
            raise ValueError("Training audit requires structure residue_uid.")
        if not isinstance(batch_ids, torch.Tensor):
            raise ValueError("Training audit requires structure batch_ids.")
        if not isinstance(ca_mask, torch.Tensor):
            raise ValueError("Training audit requires structure ca_mask.")
        residue_uid = residue_uid.detach().cpu().long()
        batch_ids = batch_ids.detach().cpu().long()
        ca_mask = ca_mask.detach().cpu().bool()
        if not (
            residue_uid.numel() == batch_ids.numel() == ca_mask.numel()
        ):
            raise ValueError(
                "residue_uid, batch_ids, and ca_mask must align with atoms."
            )

        _, residue_inverse = torch.unique(
            residue_uid,
            sorted=True,
            return_inverse=True,
        )
        unique_residue_count = int(residue_inverse.max().item()) + 1
        residue_batch_ids = torch.zeros(
            unique_residue_count,
            dtype=torch.long,
        )
        residue_batch_ids.scatter_(0, residue_inverse, batch_ids)
        residue_counts_per_sample = torch.bincount(
            residue_batch_ids,
            minlength=sample_count,
        )
        if bool((residue_counts_per_sample < 1).any()):
            missing_sample = int(
                torch.nonzero(
                    residue_counts_per_sample < 1,
                    as_tuple=True,
                )[0][0].item()
            )
            raise ValueError(
                f"Sample {missing_sample} has no structure residues."
            )
        selected_structure_counts = residue_counts_per_sample.clamp_max(
            max_structure_tokens
        )
        residue_has_ca = torch.zeros(
            unique_residue_count,
            dtype=torch.bool,
        )
        residue_has_ca[residue_inverse[ca_mask]] = True
        missing_ca_count = int((~residue_has_ca).sum().item())
        structure_token_histogram += torch.bincount(
            selected_structure_counts,
            minlength=max_structure_tokens + 1,
        ).to(device=device, dtype=torch.float64)
        residues_without_ca += float(missing_ca_count)

        mutation_ca_mask = mutation_ca_mask.detach().cpu().bool()
        requested_site_counts = requested_site_counts.detach().cpu().long()
        matched_site_counts = matched_site_counts.detach().cpu().long()
        if requested_site_counts.numel() != sample_count:
            raise ValueError("mutation_site_count must have one value per sample.")
        if matched_site_counts.numel() != sample_count:
            raise ValueError(
                "matched_mutation_site_count must have one value per sample."
            )
        missing_site_counts = (requested_site_counts - matched_site_counts).clamp_min(0)

        values = [
            sample_count,
            int((labels < float(beneficial_threshold)).sum().item()),
            1,
            pair_stats["same_complex_pairs"],
            pair_stats["same_site_pairs"],
            pair_stats["valid_complex_pairs"],
            pair_stats["valid_site_pairs"],
            int(pair_stats["valid_complex_pairs"] == 0),
            int(pair_stats["same_site_pairs"] == 0),
            int(pair_stats["valid_site_pairs"] == 0),
            int(mutation_ca_mask.sum().item()),
            int(requested_site_counts.sum().item()),
            int(matched_site_counts.sum().item()),
            int((matched_site_counts == 0).sum().item()),
            int(missing_site_counts.sum().item()),
            int(mismatch_counts.detach().cpu().sum().item()),
        ]
        stats += torch.tensor(values, dtype=torch.float64, device=device)

    if distributed and is_distributed_ready():
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(
            sequence_token_histogram,
            op=torch.distributed.ReduceOp.SUM,
        )
        torch.distributed.all_reduce(
            structure_token_histogram,
            op=torch.distributed.ReduceOp.SUM,
        )
        torch.distributed.all_reduce(
            residues_without_ca,
            op=torch.distributed.ReduceOp.SUM,
        )

    (
        sample_count,
        beneficial_count,
        batch_count,
        same_complex_pairs,
        same_site_pairs,
        valid_complex_pairs,
        valid_site_pairs,
        zero_valid_rank_batches,
        zero_same_site_batches,
        zero_valid_site_batches,
        mutation_ca_node_count,
        requested_mutation_site_count,
        matched_mutation_site_count,
        samples_without_mutation,
        missing_mutation_site_count,
        wt_mismatch_count,
    ) = stats.detach().cpu().tolist()
    sequence_hist = sequence_token_histogram.detach().cpu().tolist()
    structure_hist = structure_token_histogram.detach().cpu().tolist()

    result = {
        "samples": int(sample_count),
        "beneficial_samples": int(beneficial_count),
        "beneficial_ratio": beneficial_count / max(sample_count, 1.0),
        "batches": int(batch_count),
        "same_complex_pairs_total": int(same_complex_pairs),
        "same_complex_pairs_per_batch": same_complex_pairs / max(batch_count, 1.0),
        "same_site_pairs_total": int(same_site_pairs),
        "same_site_pairs_per_batch": same_site_pairs / max(batch_count, 1.0),
        "valid_ranking_pairs_total": int(valid_complex_pairs),
        "valid_ranking_pairs_per_batch": valid_complex_pairs / max(batch_count, 1.0),
        "valid_same_site_pairs_total": int(valid_site_pairs),
        "valid_same_site_pairs_per_batch": valid_site_pairs / max(batch_count, 1.0),
        "zero_valid_ranking_pair_batches": int(zero_valid_rank_batches),
        "zero_valid_ranking_pair_batch_rate": zero_valid_rank_batches
        / max(batch_count, 1.0),
        "zero_same_site_pair_batches": int(zero_same_site_batches),
        "zero_same_site_pair_batch_rate": zero_same_site_batches
        / max(batch_count, 1.0),
        "zero_valid_same_site_pair_batches": int(zero_valid_site_batches),
        "zero_valid_same_site_pair_batch_rate": zero_valid_site_batches
        / max(batch_count, 1.0),
        "mutation_ca_mask_nodes": int(mutation_ca_node_count),
        "requested_mutation_residues": int(requested_mutation_site_count),
        "matched_mutation_residues": int(matched_mutation_site_count),
        "matched_mutation_residues_per_sample": matched_mutation_site_count
        / max(sample_count, 1.0),
        "samples_without_matched_mutation_residue": int(samples_without_mutation),
        "missing_mutation_residues": int(missing_mutation_site_count),
        "wt_structure_mismatch_sites": int(wt_mismatch_count),
        "sequence_local_token_histogram": {
            str(token_count): int(count)
            for token_count, count in enumerate(sequence_hist)
            if count
        },
        "sequence_local_tokens_at_cap": int(
            sequence_hist[max_sequence_tokens]
        ),
        "structure_local_token_histogram": {
            str(token_count): int(count)
            for token_count, count in enumerate(structure_hist)
            if count
        },
        "structure_local_tokens_at_cap": int(
            structure_hist[max_structure_tokens]
        ),
        "structure_residues_without_ca": int(
            residues_without_ca.detach().cpu().item()
        ),
        "max_zero_same_site_batch_rate": float(max_zero_same_site_batch_rate),
    }

    if is_main_process():
        logger.info(
            "Training audit: "
            f"beneficial={result['beneficial_samples']}/{result['samples']} "
            f"({result['beneficial_ratio']:.2%}) | "
            f"same-complex pairs/batch={result['same_complex_pairs_per_batch']:.1f} | "
            f"same-site pairs/batch={result['same_site_pairs_per_batch']:.1f} | "
            f"valid rank pairs/batch={result['valid_ranking_pairs_per_batch']:.1f} | "
            f"valid same-site pairs/batch={result['valid_same_site_pairs_per_batch']:.1f}"
        )
        logger.info(
            "Training audit alignment: "
            f"matched mutation residues={result['matched_mutation_residues']}/"
            f"{result['requested_mutation_residues']} "
            f"({result['matched_mutation_residues_per_sample']:.2f}/sample) | "
            f"CA mask nodes={result['mutation_ca_mask_nodes']} | "
            f"samples without match={result['samples_without_matched_mutation_residue']} | "
            f"missing mutation residues={result['missing_mutation_residues']} | "
            f"WT/structure mismatch sites={result['wt_structure_mismatch_sites']}"
        )
        logger.info(
            "Training audit zero-pair rates: "
            f"same-site={result['zero_same_site_pair_batch_rate']:.2%} | "
            f"valid same-site={result['zero_valid_same_site_pair_batch_rate']:.2%} | "
            f"valid ranking={result['zero_valid_ranking_pair_batch_rate']:.2%}"
        )
        logger.info(
            "Training audit local tokens: "
            f"sequence histogram={result['sequence_local_token_histogram']} | "
            f"structure histogram={result['structure_local_token_histogram']} | "
            "structure residues without CA="
            f"{result['structure_residues_without_ca']}"
        )

    failure_reasons = []
    if result["samples_without_matched_mutation_residue"] > 0:
        failure_reasons.append(
            "some samples have no matched mutation residue"
        )
    if result["missing_mutation_residues"] > 0:
        failure_reasons.append(
            "some annotated mutation residues are missing from structure masks"
        )
    if result["wt_structure_mismatch_sites"] > 0:
        failure_reasons.append(
            "WT annotations disagree with structure residues"
        )
    if (
        result["zero_valid_same_site_pair_batch_rate"]
        > float(max_zero_same_site_batch_rate)
    ):
        failure_reasons.append(
            "batches without valid same-site ranking pairs reached "
            f"{result['zero_valid_same_site_pair_batch_rate']:.2%}, above the "
            f"{float(max_zero_same_site_batch_rate):.2%} limit"
        )
    result["passed"] = not failure_reasons
    result["failure_reasons"] = failure_reasons
    return result


def _weighted_huber_numpy(
    labels: np.ndarray,
    predictions: np.ndarray,
    beta: float,
    beneficial_threshold: float,
    beneficial_weight: float,
) -> float:
    error = np.abs(predictions - labels)
    beta = max(float(beta), 1e-12)
    per_sample = np.where(
        error < beta,
        0.5 * error ** 2 / beta,
        error - 0.5 * beta,
    )
    weights = np.where(
        labels < float(beneficial_threshold),
        float(beneficial_weight),
        1.0,
    )
    return float(np.sum(weights * per_sample) / max(np.sum(weights), 1e-12))


def _masked_pairwise_loss_numpy(
    labels: np.ndarray,
    predictions: np.ndarray,
    pair_mask: np.ndarray,
    temperature: float,
):
    i, j = np.nonzero(pair_mask)
    if len(i) == 0:
        return float("nan"), float("nan"), 0
    label_diff = labels[i] - labels[j]
    pred_diff = predictions[i] - predictions[j]
    temperature = max(float(temperature), 1e-12)
    losses = np.logaddexp(
        0.0,
        -np.sign(label_diff) * pred_diff / temperature,
    ) * temperature
    weights = np.clip(np.abs(label_diff), 0.5, 3.0)
    ranking_loss = float(np.sum(weights * losses) / np.sum(weights))
    pairwise_accuracy = float(np.mean(pred_diff * label_diff > 0.0))
    return ranking_loss, pairwise_accuracy, int(len(i))


def _rank_ndcg_at_k(labels: np.ndarray, predictions: np.ndarray, k: int) -> float:
    count = len(labels)
    if count < 2:
        return float("nan")
    ideal_order = np.argsort(labels, kind="stable")
    relevance = np.empty(count, dtype=np.float64)
    relevance[ideal_order] = np.arange(count, 0, -1, dtype=np.float64)
    cutoff = min(int(k), count)
    discounts = np.log2(np.arange(2, cutoff + 2, dtype=np.float64))
    predicted_order = np.argsort(predictions, kind="stable")[:cutoff]
    ideal_order = ideal_order[:cutoff]
    dcg = np.sum((np.power(2.0, relevance[predicted_order]) - 1.0) / discounts)
    ideal_dcg = np.sum((np.power(2.0, relevance[ideal_order]) - 1.0) / discounts)
    return float(dcg / ideal_dcg) if ideal_dcg > 0 else float("nan")


def compute_grouped_validation_metrics(
    labels,
    predictions,
    complex_ids,
    site_ids,
    criterion: BeneficialGroupRankLoss,
):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    predictions = np.asarray(predictions, dtype=np.float64).reshape(-1)
    count = len(labels)
    complex_ids = np.asarray(
        _normalize_group_ids(complex_ids, count),
        dtype=object,
    )
    site_ids = np.asarray(
        _normalize_group_ids(site_ids, count),
        dtype=object,
    )

    upper = np.triu(np.ones((count, count), dtype=bool), k=1)
    label_diff = labels[:, None] - labels[None, :]
    valid_gap = np.abs(label_diff) > float(criterion.pairwise_min_label_gap)
    same_complex = (complex_ids[:, None] == complex_ids[None, :]) & upper
    same_site = (site_ids[:, None] == site_ids[None, :]) & upper
    complex_mask = same_complex & valid_gap
    other_site_mask = same_complex & ~same_site & valid_gap
    site_mask = same_site & valid_gap

    complex_loss, _, complex_pair_count = _masked_pairwise_loss_numpy(
        labels,
        predictions,
        complex_mask,
        criterion.pairwise_temperature,
    )
    other_site_loss, _, other_site_pair_count = _masked_pairwise_loss_numpy(
        labels,
        predictions,
        other_site_mask,
        criterion.pairwise_temperature,
    )
    site_loss, site_accuracy, site_pair_count = _masked_pairwise_loss_numpy(
        labels,
        predictions,
        site_mask,
        criterion.pairwise_temperature,
    )

    grouped_indices = {}
    for index, site_id in enumerate(site_ids.tolist()):
        grouped_indices.setdefault(site_id, []).append(index)

    site_spearman = []
    site_ndcg = []
    recall_at_k = {1: [], 3: [], 5: []}
    beneficial_threshold = float(criterion.beneficial_threshold)
    for indices in grouped_indices.values():
        if len(indices) < 2:
            continue
        group_labels = labels[indices]
        group_predictions = predictions[indices]
        if not np.allclose(group_labels, group_labels[0]):
            if np.allclose(group_predictions, group_predictions[0]):
                site_spearman.append(0.0)
            else:
                corr, _ = spearmanr(group_labels, group_predictions)
                if np.isfinite(corr):
                    site_spearman.append(float(corr))
            ndcg = _rank_ndcg_at_k(group_labels, group_predictions, k=5)
            if np.isfinite(ndcg):
                site_ndcg.append(ndcg)

        beneficial_mask = group_labels < beneficial_threshold
        beneficial_count = int(beneficial_mask.sum())
        if beneficial_count == 0:
            continue
        predicted_order = np.argsort(group_predictions, kind="stable")
        for k in recall_at_k:
            selected = predicted_order[: min(k, len(predicted_order))]
            recall_at_k[k].append(
                float(beneficial_mask[selected].sum() / beneficial_count)
            )

    def macro_mean(values):
        return float(np.mean(values)) if values else float("nan")

    return {
        "beneficial_weighted_huber": _weighted_huber_numpy(
            labels,
            predictions,
            beta=criterion.smooth_l1_beta,
            beneficial_threshold=criterion.beneficial_threshold,
            beneficial_weight=criterion.beneficial_sample_weight,
        ),
        "same_complex_ranking_loss": complex_loss,
        "same_complex_other_site_ranking_loss": other_site_loss,
        "same_site_ranking_loss": site_loss,
        "same_site_pairwise_accuracy": site_accuracy,
        "within_site_spearman": macro_mean(site_spearman),
        "within_site_ndcg_at_5": macro_mean(site_ndcg),
        "within_site_beneficial_recall_at_1": macro_mean(recall_at_k[1]),
        "within_site_beneficial_recall_at_3": macro_mean(recall_at_k[3]),
        "within_site_beneficial_recall_at_5": macro_mean(recall_at_k[5]),
        "same_complex_valid_pair_count": complex_pair_count,
        "same_complex_other_site_valid_pair_count": other_site_pair_count,
        "same_site_valid_pair_count": site_pair_count,
        "within_site_spearman_group_count": len(site_spearman),
        "within_site_ndcg_group_count": len(site_ndcg),
        "within_site_beneficial_group_count": len(recall_at_k[1]),
    }


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
    criterion.train()
    model.train()
    grad_accum_steps = gradient_accumulation_steps or config.GRADIENT_ACCUMULATION_STEPS
    total_loss = 0.0
    total_samples = 0
    processed_batches = 0
    pending_batches = 0
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
            wt_esm_window_tokens = batch.get("wt_esm_window_tokens")
            mut_esm_window_tokens = batch.get("mut_esm_window_tokens")
            esm_window_padding_mask = batch.get("esm_window_padding_mask")
            esm_window_mutation_mask = batch.get("esm_window_mutation_mask")
            if wt_esm_emb is not None:
                wt_esm_emb = wt_esm_emb.to(device)
            if mut_esm_emb is not None:
                mut_esm_emb = mut_esm_emb.to(device)
            if mutation_esm_emb is not None:
                mutation_esm_emb = mutation_esm_emb.to(device)
            if wt_esm_window_tokens is not None:
                wt_esm_window_tokens = wt_esm_window_tokens.to(device)
            if mut_esm_window_tokens is not None:
                mut_esm_window_tokens = mut_esm_window_tokens.to(device)
            if esm_window_padding_mask is not None:
                esm_window_padding_mask = esm_window_padding_mask.to(device)
            if esm_window_mutation_mask is not None:
                esm_window_mutation_mask = esm_window_mutation_mask.to(device)

            # Skip empty batches (can happen with DataParallel)
            if len(foldx_energies) == 0:
                continue

            if structure_data is not None:
                for key in structure_data:
                    if isinstance(structure_data[key], torch.Tensor):
                        structure_data[key] = structure_data[key].to(device)

            remaining_batches = max(1, len(dataloader) - i)
            accum_denominator = min(grad_accum_steps, pending_batches + remaining_batches)
            sync_gradients = (pending_batches + 1) >= accum_denominator
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
                        wt_esm_window_tokens=wt_esm_window_tokens,
                        mut_esm_window_tokens=mut_esm_window_tokens,
                        esm_window_padding_mask=esm_window_padding_mask,
                        esm_window_mutation_mask=esm_window_mutation_mask,
                        foldx_features=foldx_features,
                )
                loss = compute_training_loss(
                    criterion,
                    pred_delta_g,
                    delta_g,
                    batch=batch,
                )
                loss = loss / accum_denominator

                # NaN/Inf check — skip batch if loss is bad
                if not torch.isfinite(loss):
                    logger.warning(f"NaN/Inf loss at batch {i}, skipping")
                    optimizer.zero_grad()
                    pending_batches = 0
                    nan_count += 1
                    continue

                scaler.scale(loss).backward()

                if sync_gradients:
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
                        wt_esm_window_tokens=wt_esm_window_tokens,
                        mut_esm_window_tokens=mut_esm_window_tokens,
                        esm_window_padding_mask=esm_window_padding_mask,
                        esm_window_mutation_mask=esm_window_mutation_mask,
                        foldx_features=foldx_features,
                    )
                loss = compute_training_loss(
                    criterion,
                    pred_delta_g,
                    delta_g,
                    batch=batch,
                )
                loss = loss / accum_denominator

                if not torch.isfinite(loss):
                    logger.warning(f"NaN/Inf loss at batch {i}, skipping")
                    optimizer.zero_grad()
                    pending_batches = 0
                    nan_count += 1
                    continue

                loss.backward()

                if sync_gradients:
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
                    wt_esm_window_tokens=wt_esm_window_tokens,
                    mut_esm_window_tokens=mut_esm_window_tokens,
                    esm_window_padding_mask=esm_window_padding_mask,
                    esm_window_mutation_mask=esm_window_mutation_mask,
                    foldx_features=foldx_features,
                )
                loss = compute_training_loss(
                    criterion,
                    pred_delta_g,
                    delta_g,
                    batch=batch,
                )
                loss = loss / accum_denominator

                # NaN/Inf check — skip batch if loss is bad
                if not torch.isfinite(loss):
                    logger.warning(f"NaN/Inf loss at batch {i}, skipping")
                    optimizer.zero_grad()
                    pending_batches = 0
                    nan_count += 1
                    continue

                loss.backward()

                if sync_gradients:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 裁剪
                    optimizer.step()
                    optimizer.zero_grad()

            # Scale loss back up for logging
            batch_loss = loss.item() * accum_denominator
            batch_size = delta_g.numel()
            total_loss += batch_loss * batch_size
            total_samples += batch_size
            processed_batches += 1
            pending_batches = 0 if sync_gradients else pending_batches + 1

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
    if pending_batches > 0:
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
    return_diagnostics: bool = False,
):
    criterion.eval()
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_labels = []
    all_complex_ids = []
    all_site_ids = []
    requires_group_ids = getattr(criterion, "requires_group_ids", False)

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
            wt_esm_window_tokens = batch.get("wt_esm_window_tokens")
            mut_esm_window_tokens = batch.get("mut_esm_window_tokens")
            esm_window_padding_mask = batch.get("esm_window_padding_mask")
            esm_window_mutation_mask = batch.get("esm_window_mutation_mask")
            if wt_esm_emb is not None:
                wt_esm_emb = wt_esm_emb.to(device)
            if mut_esm_emb is not None:
                mut_esm_emb = mut_esm_emb.to(device)
            if mutation_esm_emb is not None:
                mutation_esm_emb = mutation_esm_emb.to(device)
            if wt_esm_window_tokens is not None:
                wt_esm_window_tokens = wt_esm_window_tokens.to(device)
            if mut_esm_window_tokens is not None:
                mut_esm_window_tokens = mut_esm_window_tokens.to(device)
            if esm_window_padding_mask is not None:
                esm_window_padding_mask = esm_window_padding_mask.to(device)
            if esm_window_mutation_mask is not None:
                esm_window_mutation_mask = esm_window_mutation_mask.to(device)

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
                wt_esm_window_tokens=wt_esm_window_tokens,
                mut_esm_window_tokens=mut_esm_window_tokens,
                esm_window_padding_mask=esm_window_padding_mask,
                esm_window_mutation_mask=esm_window_mutation_mask,
                foldx_features=foldx_features,
            )
            loss = compute_training_loss(
                criterion,
                pred_delta_g,
                delta_g,
                batch=batch,
            )
            batch_size = delta_g.numel()
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_preds.append(pred_delta_g.float().cpu().numpy())
            all_labels.append(delta_g.float().cpu().numpy())
            if requires_group_ids:
                complex_ids = batch.get("complex_id")
                site_ids = batch.get("mutation_site_id")
                if complex_ids is None or site_ids is None:
                    raise ValueError(
                        "Group-aware validation requires complex_id and mutation_site_id."
                    )
                all_complex_ids.extend(
                    complex_ids.detach().cpu().reshape(-1).tolist()
                    if isinstance(complex_ids, torch.Tensor)
                    else list(complex_ids)
                )
                all_site_ids.extend(
                    site_ids.detach().cpu().reshape(-1).tolist()
                    if isinstance(site_ids, torch.Tensor)
                    else list(site_ids)
                )

    all_preds = np.concatenate(all_preds) if all_preds else np.array([])
    all_labels = np.concatenate(all_labels) if all_labels else np.array([])

    if distributed and is_distributed_ready():
        if not requires_group_ids:
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
        if requires_group_ids:
            gathered_complex_ids = [
                None for _ in range(torch.distributed.get_world_size())
            ]
            gathered_site_ids = [
                None for _ in range(torch.distributed.get_world_size())
            ]
            torch.distributed.all_gather_object(
                gathered_complex_ids,
                all_complex_ids,
            )
            torch.distributed.all_gather_object(
                gathered_site_ids,
                all_site_ids,
            )
            all_complex_ids = [
                value
                for rank_values in gathered_complex_ids
                if rank_values is not None
                for value in rank_values
            ]
            all_site_ids = [
                value
                for rank_values in gathered_site_ids
                if rank_values is not None
                for value in rank_values
            ]

    if requires_group_ids and len(all_preds) > 0:
        original_pair_cap = getattr(criterion, "pairwise_max_pairs", None)
        if original_pair_cap is not None:
            criterion.pairwise_max_pairs = 0
        try:
            validation_loss = criterion(
                torch.as_tensor(all_preds, dtype=torch.float32, device=device),
                torch.as_tensor(all_labels, dtype=torch.float32, device=device),
                complex_ids=all_complex_ids,
                site_ids=all_site_ids,
            )
        finally:
            if original_pair_cap is not None:
                criterion.pairwise_max_pairs = original_pair_cap
        avg_loss = float(validation_loss.item())
    else:
        avg_loss = total_loss / max(total_samples, 1)

    if len(all_preds) > 1:
        pearson_corr, _ = pearsonr(all_labels, all_preds)
    else:
        pearson_corr = 0.0

    diagnostics = {}
    if (
        return_diagnostics
        and requires_group_ids
        and len(all_preds) > 0
        and isinstance(criterion, BeneficialGroupRankLoss)
    ):
        diagnostics = compute_grouped_validation_metrics(
            labels=all_labels,
            predictions=all_preds,
            complex_ids=all_complex_ids,
            site_ids=all_site_ids,
            criterion=criterion,
        )

    result = (avg_loss, pearson_corr, all_labels, all_preds)
    if return_diagnostics:
        return (*result, diagnostics)
    return result
