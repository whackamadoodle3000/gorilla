from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from bfcl_eval.constants.eval_config import PROJECT_ROOT
from bfcl_eval.utils import (
    get_general_grouping,
    is_live,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _now_stamp() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def _import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "matplotlib is required to generate ACE metrics plots. "
            "Please install it with `pip install matplotlib`."
        ) from exc
    return plt


TRAINING_METRICS_DIR = PROJECT_ROOT / "bfcl_eval" / "data" / "ace" / "metrics" / "training"


@dataclass
class PlaybookTrainingMetricsCollector:
    """
    Collects curator operation stats for ACE training runs.
    """

    chunk_size: int = 100
    sample_records: List[dict] = field(default_factory=list)
    section_counts: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"ADD": 0, "MODIFY": 0, "REMOVE": 0})
    )

    def record_sample(
        self,
        *,
        entry_id: str,
        tool_groups: Sequence[str],
        applied_operations: Sequence[dict],
        outcome: str,
        evaluation_passed: Optional[bool] = None,
        notes: Optional[str] = None,
    ) -> None:
        """
        Record the curator behaviour for a single training sample.
        """

        operation_types = {"ADD": False, "MODIFY": False, "REMOVE": False}
        tracked_operations: List[dict] = []
        for op in applied_operations:
            op_type = op.get("type", "").upper()
            section = op.get("section")
            if op_type not in operation_types or not section:
                continue
            operation_types[op_type] = True
            normalized_section = op.get("normalized_section", section)
            self.section_counts[normalized_section][op_type] += 1
            tracked_operations.append(
                {
                    "type": op_type,
                    "section": section,
                    "normalized_section": normalized_section,
                }
            )

        operation_types["NO_OP"] = not any(operation_types.values())

        self.sample_records.append(
            {
                "entry_id": entry_id,
                "tool_groups": list(tool_groups),
                "flags": operation_types,
                "operations": tracked_operations,
                "outcome": outcome,
                "evaluation_passed": evaluation_passed,
                "notes": notes,
            }
        )

    def _build_batch_view(self) -> List[dict]:
        batches: List[dict] = []
        if not self.sample_records:
            return batches

        for idx in range(0, len(self.sample_records), self.chunk_size):
            chunk = self.sample_records[idx : idx + self.chunk_size]
            size = len(chunk)
            if size == 0:
                continue
            counts = {"ADD": 0, "MODIFY": 0, "REMOVE": 0, "NO_OP": 0}
            pass_counters = {"passed": 0, "failed": 0, "unknown": 0}
            for sample in chunk:
                flags = sample["flags"]
                for key in counts.keys():
                    if flags.get(key):
                        counts[key] += 1
                evaluation_passed = sample.get("evaluation_passed")
                if evaluation_passed is True:
                    pass_counters["passed"] += 1
                elif evaluation_passed is False:
                    pass_counters["failed"] += 1
                else:
                    pass_counters["unknown"] += 1
            ratios = {key: counts[key] / size for key in counts}
            pass_ratio = pass_counters["passed"] / size
            batches.append(
                {
                    "batch_index": len(batches) + 1,
                    "start_sample": idx + 1,
                    "end_sample": idx + size,
                    "size": size,
                    "counts": counts,
                    "ratios": ratios,
                    "pass_counters": pass_counters,
                    "pass_ratio": pass_ratio,
                }
            )
        return batches

    def _build_summary(self) -> dict:
        total = len(self.sample_records)
        counts = {"ADD": 0, "MODIFY": 0, "REMOVE": 0, "NO_OP": 0}
        pass_counters = {"passed": 0, "failed": 0, "unknown": 0}
        for sample in self.sample_records:
            flags = sample["flags"]
            for key in counts:
                if flags.get(key):
                    counts[key] += 1
            evaluation_passed = sample.get("evaluation_passed")
            if evaluation_passed is True:
                pass_counters["passed"] += 1
            elif evaluation_passed is False:
                pass_counters["failed"] += 1
            else:
                pass_counters["unknown"] += 1
        ratios = {key: (counts[key] / total if total else 0.0) for key in counts}
        pass_ratio = pass_counters["passed"] / total if total else 0.0
        return {
            "total_samples": total,
            "counts": counts,
            "ratios": ratios,
            "pass_counters": pass_counters,
            "pass_ratio": pass_ratio,
        }

    def save(self, *, metadata: dict) -> dict:
        """
        Persist collected metrics to disk and generate plots.
        Returns a dict with paths for convenience.
        """

        if not self.sample_records:
            return {}

        output_dir = _ensure_dir(TRAINING_METRICS_DIR)
        timestamp = _now_stamp()

        batches = self._build_batch_view()
        summary = self._build_summary()

        payload = {
            "metadata": metadata,
            "summary": summary,
            "per_batch": batches,
            "per_section": self.section_counts,
            "sample_history": self.sample_records,
        }

        data_path = output_dir / f"ace_training_metrics_{timestamp}.json"
        with data_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        plot_path = output_dir / f"ace_training_operation_mix_{timestamp}.png"
        try:
            plt = _import_matplotlib()
            if batches:
                x = [batch["batch_index"] for batch in batches]
                add_y = [batch["ratios"]["ADD"] for batch in batches]
                mod_y = [batch["ratios"]["MODIFY"] for batch in batches]
                rem_y = [batch["ratios"]["REMOVE"] for batch in batches]
                noop_y = [batch["ratios"]["NO_OP"] for batch in batches]
                pass_y = [batch["pass_ratio"] for batch in batches]

                plt.figure(figsize=(12, 6))
                plt.plot(x, add_y, marker="o", label="ADD")
                plt.plot(x, mod_y, marker="o", label="MODIFY")
                plt.plot(x, rem_y, marker="o", label="REMOVE")
                plt.plot(x, noop_y, marker="o", label="NO OP")
                plt.plot(x, pass_y, marker="o", linestyle="--", color="#2F855A", label="BFCL pass ratio")
                plt.title("ACE curator operation mix & pass rate per 100 training samples")
                plt.xlabel(f"Batch ({self.chunk_size} samples)")
                plt.ylabel("Proportion of samples")
                plt.ylim(0.0, 1.05)
                plt.grid(alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_path, dpi=160)
                plt.close()
            else:  # pragma: no cover - safety
                plot_path.touch()
        except RuntimeError:
            plot_path = None

        return {
            "data_path": str(data_path),
            "plot_path": str(plot_path) if plot_path else None,
        }


@dataclass
class EvaluationMetricsCollector:
    """
    Aggregates evaluation accuracy by tool group and tool subcategory.
    """

    model_name: str
    records: List[dict] = field(default_factory=list)
    tool_group_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    simulated_tool_group_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    subcategory_stats: Dict[str, Dict[str, int]] = field(
        default_factory=lambda: defaultdict(lambda: {"correct": 0, "total": 0})
    )
    tool_count_histogram: Dict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record_entry(
        self,
        *,
        entry_id: str,
        test_category: str,
        tool_groups: Sequence[str],
        valid: bool,
        tool_count: int,
    ) -> None:
        success = bool(valid)
        tool_groups = list(tool_groups)
        self.records.append(
            {
                "entry_id": entry_id,
                "test_category": test_category,
                "tool_groups": tool_groups,
                "valid": success,
                "tool_count": tool_count,
            }
        )

        self.tool_count_histogram[tool_count] += 1

        grouping = get_general_grouping(test_category)
        stats = self.subcategory_stats[grouping]
        stats["total"] += 1
        if success:
            stats["correct"] += 1

        for group in tool_groups:
            self.tool_group_stats[group]["total"] += 1
            if success:
                self.tool_group_stats[group]["correct"] += 1
            if not is_live(test_category):
                simulated_stats = self.simulated_tool_group_stats[group]
                simulated_stats["total"] += 1
                if success:
                    simulated_stats["correct"] += 1

    @staticmethod
    def _compute_accuracy_table(stats: Dict[str, Dict[str, int]]) -> Dict[str, dict]:
        table: Dict[str, dict] = {}
        for key, value in stats.items():
            total = value["total"]
            correct = value["correct"]
            accuracy = correct / total if total else 0.0
            table[key] = {"accuracy": accuracy, "correct": correct, "total": total}
        return dict(sorted(table.items(), key=lambda item: item[0]))

    def finalize(self, *, score_dir: Path, metadata: Optional[dict] = None) -> dict:
        if not self.records:
            return {}

        model_dir = (
            score_dir
            / self.model_name.replace("/", "_")
            / "ace_metrics"
        )
        output_dir = _ensure_dir(model_dir)
        timestamp = _now_stamp()

        tool_table = self._compute_accuracy_table(self.tool_group_stats)
        simulated_table = self._compute_accuracy_table(self.simulated_tool_group_stats)
        subcategory_table = self._compute_accuracy_table(self.subcategory_stats)

        payload = {
            "metadata": metadata or {},
            "tool_group_accuracy": tool_table,
            "simulated_tool_group_accuracy": simulated_table,
            "subcategory_accuracy": subcategory_table,
             "tool_count_distribution": dict(sorted(self.tool_count_histogram.items())),
            "records": self.records,
        }

        data_path = output_dir / f"{self.model_name.replace('/', '_')}_eval_metrics_{timestamp}.json"
        with data_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        subcategory_plot_path = output_dir / f"{self.model_name.replace('/', '_')}_subcategory_accuracy_{timestamp}.png"
        try:
            plt = _import_matplotlib()
            categories = list(subcategory_table.keys())
            if categories:
                accuracies = [subcategory_table[cat]["accuracy"] for cat in categories]
                fig_height = max(3.0, 0.7 * len(categories))
                plt.figure(figsize=(10, fig_height))
                bars = plt.barh(categories, accuracies, color="#4B8BBE")
                plt.xlabel("Accuracy")
                plt.title(f"{self.model_name} – accuracy by tool subcategory")
                plt.xlim(0.0, 1.0)
                for bar, acc in zip(bars, accuracies):
                    plt.text(
                        acc + 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{acc:.1%}",
                        va="center",
                        ha="left",
                    )
                plt.tight_layout()
                plt.savefig(subcategory_plot_path, dpi=160)
                plt.close()
            else:  # pragma: no cover - safety
                subcategory_plot_path.touch()
        except RuntimeError:
            subcategory_plot_path = None

        tool_count_plot_path = output_dir / f"{self.model_name.replace('/', '_')}_tool_count_distribution_{timestamp}.png"
        try:
            plt = _import_matplotlib()
            counts = dict(sorted(self.tool_count_histogram.items()))
            if counts:
                x_vals = list(counts.keys())
                y_vals = [counts[count] for count in x_vals]
                plt.figure(figsize=(10, 5))
                bars = plt.bar(x_vals, y_vals, color="#306998")
                plt.xlabel("Number of tools available")
                plt.ylabel("Test cases")
                plt.title(f"{self.model_name} – distribution of available tools per test case")
                for bar, freq in zip(bars, y_vals):
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + max(y_vals) * 0.01,
                        str(freq),
                        ha="center",
                        va="bottom",
                    )
                plt.tight_layout()
                plt.savefig(tool_count_plot_path, dpi=160)
                plt.close()
            else:  # pragma: no cover
                tool_count_plot_path.touch()
        except RuntimeError:
            tool_count_plot_path = None

        return {
            "data_path": str(data_path),
            "subcategory_plot_path": str(subcategory_plot_path)
            if subcategory_plot_path
            else None,
            "tool_count_plot_path": str(tool_count_plot_path)
            if tool_count_plot_path
            else None,
        }


