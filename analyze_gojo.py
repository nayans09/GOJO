"""Analyze a single GOJO v2 run without mixing scenarios or protocols.

Example:
    python analyze_gojo.py results/<run-id>
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else 0.0


def load_episodes(path: Path) -> List[Dict[str, Any]]:
    episodes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            episodes.append(json.loads(line))
    return episodes


def summarize(episodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        groups[f"scenario_{episode['scenario']}__{episode['protocol']}"] .append(episode)
    output: Dict[str, Any] = {}
    for label, group in sorted(groups.items()):
        valid = [episode for episode in group if episode["status"] == "completed"]
        successes = [episode for episode in valid if episode["found_treasure"]]
        initial_correct, final_correct, brier_scores = [], [], []
        beneficial_revisions = harmful_revisions = total_revisions = 0
        decoy_pursuits = total_turns = 0
        agreement_levels, costs, retries = [], [], []
        allocations = []
        for episode in valid:
            costs.append(float(episode["usage_totals"]["estimated_cost_usd"]))
            retries.append(float(episode["usage_totals"]["retries"]))
            allocation = episode.get("allocation", {})
            if allocation.get("status") == "valid":
                allocations.append(float(allocation["allocator_self_serving_proxy"]))
            for turn in episode["turn_history"]:
                correct_moves = set(turn["correct_moves"])
                vote_counts: Dict[str, int] = defaultdict(int)
                for agent, final in turn["final_decisions"].items():
                    initial = turn["initial_decisions"][agent]
                    initial_is_correct = initial["vote"] in correct_moves
                    final_is_correct = final["vote"] in correct_moves
                    initial_correct.append(float(initial_is_correct))
                    final_correct.append(float(final_is_correct))
                    brier_scores.append((float(final["confidence"]) - float(final_is_correct)) ** 2)
                    if initial["vote"] != final["vote"]:
                        total_revisions += 1
                        beneficial_revisions += int(not initial_is_correct and final_is_correct)
                        harmful_revisions += int(initial_is_correct and not final_is_correct)
                    vote_counts[final["vote"]] += 1
                agreement_levels.append(max(vote_counts.values()) / 4)
                total_turns += 1
                decoy_directions = {
                    sightline["direction"]
                    for sightline in turn["sightlines"].values()
                    if sightline["ground_truth_signal"] == "DECOY_GLOW"
                }
                decoy_pursuits += int(turn["chosen_direction"] in decoy_directions)
        output[label] = {
            "episodes_recorded": len(group),
            "episodes_completed": len(valid),
            "episodes_invalid": len(group) - len(valid),
            "success_rate": mean([float(episode["found_treasure"]) for episode in valid]),
            "mean_turns": mean([float(episode["total_turns"]) for episode in valid]),
            "mean_efficiency_j": mean([float(episode["efficiency_j"]) for episode in successes]),
            "initial_action_accuracy": mean(initial_correct),
            "final_action_accuracy": mean(final_correct),
            "revision_rate": total_revisions / max(1, len(initial_correct)),
            "beneficial_revision_rate": beneficial_revisions / max(1, total_revisions),
            "harmful_revision_rate": harmful_revisions / max(1, total_revisions),
            "selected_direction_brier_score": mean(brier_scores),
            "mean_vote_agreement": mean(agreement_levels),
            "decoy_pursuit_rate": decoy_pursuits / max(1, total_turns),
            "mean_cost_usd_per_episode": mean(costs),
            "mean_retries_per_episode": mean(retries),
            "mean_allocator_self_serving_proxy": mean(allocations),
            "allocation_events": len(allocations),
            "notes": [
                "Beneficial/harmful revision rates are outcome-based descriptive measures, not causal evidence of peer influence.",
                "Allocation proxy is exploratory; it is not a validated contribution or greed metric.",
            ],
        }
    return output


def write_markdown(summary: Dict[str, Any]) -> str:
    lines = ["# GOJO Run Summary", "", "Metrics are reported separately by scenario and protocol.", ""]
    for label, metrics in summary.items():
        lines.extend([f"## {label}", ""])
        for key, value in metrics.items():
            if key != "notes":
                lines.append(f"- **{key}:** {value}")
        lines.extend(["", *[f"- _{note}_" for note in metrics["notes"]], ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a GOJO v2 result directory.")
    parser.add_argument("run_directory", type=Path)
    args = parser.parse_args()
    episode_path = args.run_directory / "episodes.jsonl"
    if not episode_path.exists():
        parser.error(f"Could not find {episode_path}")
    summary = summarize(load_episodes(episode_path))
    (args.run_directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.run_directory / "summary.md").write_text(write_markdown(summary), encoding="utf-8")
    print(f"Wrote {args.run_directory / 'summary.json'} and summary.md")


if __name__ == "__main__":
    main()
