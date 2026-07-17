import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


EVENT_ORDER = [
    "directional_or_detour_error",
    "obstacle_blindness",
    "boundary_error",
    "parse_failure",
    "no_first_error",
]

LABEL_MAP = {
    "directional_or_detour_error": "Directional",
    "obstacle_blindness": "Obstacle blindness",
    "boundary_error": "Boundary",
    "parse_failure": "Parse failure",
    "no_first_error": "No first error",
}

POLICY_ORDER = ["baseline", "legality_shield", "reasoning"]
POLICY_LABEL = {
    "baseline": "Baseline",
    "legality_shield": "Shield",
    "reasoning": "Reasoning",
}

ENV_LABEL = {
    "MiniGrid-SimpleCrossingS9N1-v0": "SimpleCrossing N1",
    "MiniGrid-SimpleCrossingS9N2-v0": "SimpleCrossing N2",
    "MiniGrid-SimpleCrossingS9N3-v0": "SimpleCrossing N3",
    "MiniGrid-FourRooms-v0": "FourRooms",
}


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_div(a, b):
    return a / b if b else 0.0


def classify_error(step):
    if step.get("parse_failure", False) or not step.get("is_valid_format", True):
        return "parse_failure"
    if step.get("hit_obstacle", False):
        return "obstacle_blindness"
    if step.get("hit_wall", False):
        return "boundary_error"
    if not step.get("is_correct", False):
        return "directional_or_detour_error"
    return "correct"


def get_first_event(step_logs):
    for step in step_logs:
        event = classify_error(step)
        if event != "correct":
            return event
    return "no_first_error"


def summarize(results):
    summary = defaultdict(lambda: {
        "episodes": 0,
        "successes": 0,
        "first_events": defaultdict(int),
    })

    for ep in results:
        key = (
            ep["env_name"],
            ep.get("policy_type", "baseline"),
            ep["mode"],
        )

        summary[key]["episodes"] += 1

        if ep.get("reached_goal", False):
            summary[key]["successes"] += 1

        event = get_first_event(ep.get("step_logs", []))
        summary[key]["first_events"][event] += 1

    return summary


def get_envs(summary):
    envs = sorted({key[0] for key in summary.keys()})
    return envs


def get_policies(summary, env_name):
    policies = {policy for env, policy, _ in summary.keys() if env == env_name}
    return [p for p in POLICY_ORDER if p in policies]


def plot_success_by_env(summary, output_dir):
    for env_name in get_envs(summary):
        policies = get_policies(summary, env_name)

        categories = []
        for policy in policies:
            categories.append((policy, "allocentric", f"{POLICY_LABEL[policy]}\nAllocentric"))
            categories.append((policy, "egocentric", f"{POLICY_LABEL[policy]}\nEgocentric"))

        labels = [label for _, _, label in categories]
        values = []

        for policy, mode, _ in categories:
            data = summary.get((env_name, policy, mode))
            values.append(100 * safe_div(data["successes"], data["episodes"]) if data else 0.0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(labels, values)

        ax.set_title(f"{ENV_LABEL.get(env_name, env_name)} Success Rates")
        ax.set_ylabel("Success (%)")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", alpha=0.3)

        for i, value in enumerate(values):
            ax.text(i, min(value + 2, 98), f"{value:.1f}%", ha="center", va="bottom")

        fig.tight_layout()

        safe_name = env_name.replace("-", "_")
        fig.savefig(output_dir / f"{safe_name}_success_rates.png", dpi=300)
        plt.close()


def plot_first_events_by_env(summary, output_dir):
    for env_name in get_envs(summary):
        policies = get_policies(summary, env_name)

        categories = []
        for policy in policies:
            categories.append((policy, "allocentric", f"{POLICY_LABEL[policy]}\nAllocentric"))
            categories.append((policy, "egocentric", f"{POLICY_LABEL[policy]}\nEgocentric"))

        labels = [label for _, _, label in categories]
        x = list(range(len(labels)))
        bottoms = [0.0 for _ in labels]

        fig, ax = plt.subplots(figsize=(11, 5))

        for event in EVENT_ORDER:
            values = []

            for policy, mode, _ in categories:
                data = summary.get((env_name, policy, mode))
                if data is None:
                    values.append(0.0)
                    continue

                count = data["first_events"].get(event, 0)
                values.append(100 * safe_div(count, data["episodes"]))

            if all(v == 0 for v in values):
                continue

            ax.bar(x, values, bottom=bottoms, label=LABEL_MAP[event])
            bottoms = [b + v for b, v in zip(bottoms, values)]

        ax.set_title(f"{ENV_LABEL.get(env_name, env_name)} First-Event Outcomes")
        ax.set_ylabel("Percentage of Episodes (%)")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()

        safe_name = env_name.replace("-", "_")
        fig.savefig(output_dir / f"{safe_name}_first_events.png", dpi=300, bbox_inches="tight")
        plt.close()


def plot_success_sweep(summary, output_dir):
    env_order = [
        "MiniGrid-SimpleCrossingS9N1-v0",
        "MiniGrid-SimpleCrossingS9N2-v0",
        "MiniGrid-SimpleCrossingS9N3-v0",
        "MiniGrid-FourRooms-v0",
    ]

    rows = []

    for env_name in env_order:
        if env_name not in get_envs(summary):
            continue

        for policy in get_policies(summary, env_name):
            for mode in ["allocentric", "egocentric"]:
                data = summary.get((env_name, policy, mode))
                if data is None:
                    continue

                rows.append({
                    "label": f"{ENV_LABEL.get(env_name, env_name)}\n{POLICY_LABEL[policy]}\n{mode[:5]}",
                    "success": 100 * safe_div(data["successes"], data["episodes"]),
                })

    labels = [r["label"] for r in rows]
    values = [r["success"] for r in rows]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(labels, values)

    ax.set_title("Farama MiniGrid Success Rates Across Environments")
    ax.set_ylabel("Success (%)")
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(values):
        ax.text(i, min(value + 2, 98), f"{value:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.tick_params(axis="x", labelrotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "farama_success_sweep.png", dpi=300)
    plt.close()


def main():
    output_dir = Path("plots_farama")
    output_dir.mkdir(exist_ok=True)

    # results = load_results("farama_minigrid_results.json")
    results = load_results("farama_minigrid_results_final.json")
    summary = summarize(results)

    plot_success_by_env(summary, output_dir)
    plot_first_events_by_env(summary, output_dir)
    plot_success_sweep(summary, output_dir)

    print("Saved plots to:", output_dir)
    print("- one success plot per environment")
    print("- one first-event plot per environment")
    print("- farama_success_sweep.png")


if __name__ == "__main__":
    main()