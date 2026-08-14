from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_DIR = Path("presentation_plots")
OUTPUT_DIR.mkdir(exist_ok=True)


POLICY_ORDER = [
    "baseline",
    "reprompt_control",
    "legality_shield",
    "reasoning",
]

POLICY_LABELS = {
    "baseline": "Baseline",
    "reprompt_control": "Reprompt\ncontrol",
    "legality_shield": "Legality\nshield",
    "reasoning": "Reasoning",
}

ENV_ORDER = [
    "SC-N1",
    "SC-N2",
    "SC-N3",
    "FourRooms",
]

ENV_LABELS = {
    "SC-N1": "SC-N1",
    "SC-N2": "SC-N2",
    "SC-N3": "SC-N3",
    "FourRooms": "FourRooms",
}


# -------------------------------------------------------------------
# FINAL 30-SEED RESULTS
# Source: results_tables_Daniel.pdf
# -------------------------------------------------------------------

# T5: Overall pooled over environments and framings
overall_success = {
    "baseline": 8.8,
    "reprompt_control": 25.4,
    "legality_shield": 36.2,
    "reasoning": 20.0,
}

overall_ci = {
    "baseline": (5.8, 13.0),
    "reprompt_control": (20.3, 31.3),
    "legality_shield": (30.4, 42.5),
    "reasoning": (15.4, 25.5),
}

overall_step_accuracy = {
    "baseline": 46.8,
    "reprompt_control": 52.9,
    "legality_shield": 52.8,
    "reasoning": 48.4,
}

# Pooled by framing
framing_success = {
    "allocentric": {
        "baseline": 15.0,
        "reprompt_control": 44.2,
        "legality_shield": 45.8,
        "reasoning": 18.3,
    },
    "egocentric": {
        "baseline": 2.5,
        "reprompt_control": 6.7,
        "legality_shield": 26.7,
        "reasoning": 21.7,
    },
}

# Environment breakdown
allocentric_env = {
    "SC-N1": {
        "baseline": 23.3,
        "reprompt_control": 60.0,
        "legality_shield": 63.3,
        "reasoning": 20.0,
    },
    "SC-N2": {
        "baseline": 6.7,
        "reprompt_control": 43.3,
        "legality_shield": 36.7,
        "reasoning": 3.3,
    },
    "SC-N3": {
        "baseline": 3.3,
        "reprompt_control": 30.0,
        "legality_shield": 33.3,
        "reasoning": 10.0,
    },
    "FourRooms": {
        "baseline": 26.7,
        "reprompt_control": 43.3,
        "legality_shield": 50.0,
        "reasoning": 40.0,
    },
}

egocentric_env = {
    "SC-N1": {
        "baseline": 0.0,
        "reprompt_control": 13.3,
        "legality_shield": 40.0,
        "reasoning": 6.7,
    },
    "SC-N2": {
        "baseline": 0.0,
        "reprompt_control": 0.0,
        "legality_shield": 26.7,
        "reasoning": 10.0,
    },
    "SC-N3": {
        "baseline": 0.0,
        "reprompt_control": 3.3,
        "legality_shield": 26.7,
        "reasoning": 20.0,
    },
    "FourRooms": {
        "baseline": 10.0,
        "reprompt_control": 10.0,
        "legality_shield": 13.3,
        "reasoning": 50.0,
    },
}

# Mechanism numbers
mechanism_accuracy = {
    "baseline": {
        "first_attempt": 46.8,
        "executed": 46.8,
    },
    "reprompt_control": {
        "first_attempt": 42.3,
        "executed": 52.9,
    },
    "legality_shield": {
        "first_attempt": 32.8,
        "executed": 52.8,
    },
    "reasoning": {
        "first_attempt": 48.4,
        "executed": 48.4,
    },
}

reprompt_fired = {
    "baseline": 0.0,
    "reprompt_control": 20.5,
    "legality_shield": 36.3,
    "reasoning": 0.0,
}

executed_accuracy_by_framing = {
    "allocentric": {
        "legality_shield": 54.6,
        "reprompt_control": 57.2,
    },
    "egocentric": {
        "legality_shield": 51.8,
        "reprompt_control": 51.1,
    },
}

# Retry-budget exhaustion / parse-failure terminations, counts out of 120
retry_exhaustion_by_framing = {
    "allocentric": {
        "legality_shield": 30,
        "reprompt_control": 58,
    },
    "egocentric": {
        "legality_shield": 12,
        "reprompt_control": 68,
    },
}

termination = {
    "baseline": {
        "success": 21,
        "early_stop": 198,
        "max_steps": 21,
        "parse_failure": 0,
    },
    "reprompt_control": {
        "success": 61,
        "early_stop": 0,
        "max_steps": 53,
        "parse_failure": 126,
    },
    "legality_shield": {
        "success": 87,
        "early_stop": 0,
        "max_steps": 111,
        "parse_failure": 42,
    },
    "reasoning": {
        "success": 48,
        "early_stop": 67,
        "max_steps": 125,
        "parse_failure": 0,
    },
}

TERMINATION_ORDER = ["success", "early_stop", "max_steps", "parse_failure"]
TERMINATION_LABELS = {
    "success": "Success",
    "early_stop": "Early stop",
    "max_steps": "Max steps",
    "parse_failure": "Parse failure",
}

# -------------------------------------------------------------------
# ROBUSTNESS / SLIDE 9
# -------------------------------------------------------------------

# Reasoning vs baseline pooled by framing
reasoning_vs_baseline = {
    "allocentric": {
        "baseline": 15.0,
        "reasoning": 18.3,
    },
    "egocentric": {
        "baseline": 2.5,
        "reasoning": 21.7,
    },
}

# Cross-model replication of the "hint value"
# Defined as legality_shield success minus reprompt_control success
cross_model_hint_value = {
    "GPT-4o-mini": {
        "allocentric": 1.7,
        "egocentric": 20.0,
    },
    "GPT-4.1-mini": {
        "allocentric": 0.8,
        "egocentric": 18.3,
    },
}


# -------------------------------------------------------------------
# GAZEBO TRANSFER / SLIDE 10
# -------------------------------------------------------------------

# TurtleBot3 Gazebo transfer: 5 seeds per policy x framing condition
gazebo_success = {
    "allocentric": {
        "legality_shield": 40.0,      # 2/5
        "reprompt_control": 40.0,     # 2/5
    },
    "egocentric": {
        "legality_shield": 60.0,      # 3/5
        "reprompt_control": 20.0,     # 1/5
    },
}

gazebo_success_counts = {
    "allocentric": {
        "legality_shield": 2,
        "reprompt_control": 2,
    },
    "egocentric": {
        "legality_shield": 3,
        "reprompt_control": 1,
    },
}

# Useful supporting mechanism result for the slide text:
# Egocentric retry exhaustion: shield 0/5, control 4/5.
gazebo_retry_exhaustion_counts = {
    "allocentric": {
        "legality_shield": 2,
        "reprompt_control": 3,
    },
    "egocentric": {
        "legality_shield": 0,
        "reprompt_control": 4,
    },
}

GAZEBO_N_PER_CELL = 5
GAZEBO_MAX_STEP_ERROR_M = 0.0464


def save_bar_plot(filename, title, labels, values, ylabel="Success (%)", ylim=(0, 100), annotate_fmt="{:.1f}%"):
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(value + 1.5, ylim[1] - 2),
            annotate_fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def save_grouped_bar_plot(filename, title, group_labels, series_dict, ylabel="Success (%)", ylim=(0, 100), annotate=True):
    fig, ax = plt.subplots(figsize=(10, 5))

    x = list(range(len(group_labels)))
    n_series = len(series_dict)
    width = 0.8 / n_series

    for i, (series_name, values) in enumerate(series_dict.items()):
        offsets = [xi - 0.4 + width / 2 + i * width for xi in x]
        bars = ax.bar(offsets, values, width=width, label=series_name)

        if annotate:
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 1.2, ylim[1] - 2),
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close(fig)


def plot_overall_success():
    labels = [POLICY_LABELS[p] for p in POLICY_ORDER]
    values = [overall_success[p] for p in POLICY_ORDER]

    save_bar_plot(
        filename="01_overall_success.png",
        title="Overall Success by Policy (30-seed final run)",
        labels=labels,
        values=values,
        ylabel="Success (%)",
        ylim=(0, 50),
    )


def plot_success_by_framing():
    labels = [POLICY_LABELS[p] for p in POLICY_ORDER]

    series = {
        "Allocentric": [framing_success["allocentric"][p] for p in POLICY_ORDER],
        "Egocentric": [framing_success["egocentric"][p] for p in POLICY_ORDER],
    }

    save_grouped_bar_plot(
        filename="02_success_by_framing.png",
        title="Success by Policy and Action Framing",
        group_labels=labels,
        series_dict=series,
        ylabel="Success (%)",
        ylim=(0, 60),
    )


def plot_core_shield_vs_control():
    labels = ["Allocentric", "Egocentric"]

    series = {
        "Reprompt control": [
            framing_success["allocentric"]["reprompt_control"],
            framing_success["egocentric"]["reprompt_control"],
        ],
        "Legality shield": [
            framing_success["allocentric"]["legality_shield"],
            framing_success["egocentric"]["legality_shield"],
        ],
    }

    save_grouped_bar_plot(
        filename="03_shield_vs_control.png",
        title="Core Finding: Shield vs Control by Framing",
        group_labels=labels,
        series_dict=series,
        ylabel="Success (%)",
        ylim=(0, 60),
    )


def plot_step_accuracy():
    labels = [POLICY_LABELS[p] for p in POLICY_ORDER]
    values = [overall_step_accuracy[p] for p in POLICY_ORDER]

    save_bar_plot(
        filename="04_step_accuracy.png",
        title="Executed Step Accuracy by Policy",
        labels=labels,
        values=values,
        ylabel="Executed step accuracy (%)",
        ylim=(0, 60),
    )


def plot_first_attempt_vs_executed():
    labels = [POLICY_LABELS[p] for p in POLICY_ORDER]

    series = {
        "First-attempt": [mechanism_accuracy[p]["first_attempt"] for p in POLICY_ORDER],
        "Executed": [mechanism_accuracy[p]["executed"] for p in POLICY_ORDER],
    }

    save_grouped_bar_plot(
        filename="05_first_attempt_vs_executed.png",
        title="First-attempt vs Executed Step Accuracy",
        group_labels=labels,
        series_dict=series,
        ylabel="Accuracy (%)",
        ylim=(0, 60),
    )


def plot_reprompt_firing():
    labels = [POLICY_LABELS[p] for p in POLICY_ORDER]
    values = [reprompt_fired[p] for p in POLICY_ORDER]

    save_bar_plot(
        filename="06_reprompt_firing.png",
        title="Reprompt Firing Rate by Policy",
        labels=labels,
        values=values,
        ylabel="Reprompt fired on steps (%)",
        ylim=(0, 40),
    )


def plot_termination_taxonomy():
    labels = [POLICY_LABELS[p] for p in POLICY_ORDER]
    x = list(range(len(labels)))
    bottoms = [0 for _ in labels]

    fig, ax = plt.subplots(figsize=(10, 5))

    for term in TERMINATION_ORDER:
        values = [termination[p][term] for p in POLICY_ORDER]
        bars = ax.bar(x, values, bottom=bottoms, label=TERMINATION_LABELS[term])

        for bar, value, bottom in zip(bars, values, bottoms):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bottom + value / 2,
                    str(value),
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        bottoms = [b + v for b, v in zip(bottoms, values)]

    ax.set_title("Termination Taxonomy by Policy (counts out of 240 episodes)")
    ax.set_ylabel("Episode count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "07_termination_taxonomy.png", dpi=300)
    plt.close(fig)


def plot_mechanism_slide():
    framings = ["Allocentric", "Egocentric"]
    framing_keys = ["allocentric", "egocentric"]

    shield_accuracy = [
        executed_accuracy_by_framing[f]["legality_shield"]
        for f in framing_keys
    ]
    control_accuracy = [
        executed_accuracy_by_framing[f]["reprompt_control"]
        for f in framing_keys
    ]

    shield_exhaustion_counts = [
        retry_exhaustion_by_framing[f]["legality_shield"]
        for f in framing_keys
    ]
    control_exhaustion_counts = [
        retry_exhaustion_by_framing[f]["reprompt_control"]
        for f in framing_keys
    ]

    shield_exhaustion_pct = [count / 120 * 100 for count in shield_exhaustion_counts]
    control_exhaustion_pct = [count / 120 * 100 for count in control_exhaustion_counts]

    x = np.arange(len(framings))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    ax = axes[0]
    shield_bars = ax.bar(
        x - width / 2,
        shield_accuracy,
        width,
        label="Legality shield",
    )
    control_bars = ax.bar(
        x + width / 2,
        control_accuracy,
        width,
        label="Reprompt control",
    )

    ax.set_title(
        "Executed-action accuracy remains similar",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Executed-action accuracy (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(framings)
    ax.set_ylim(0, 65)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bars in [shield_bars, control_bars]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    ax = axes[1]
    shield_bars = ax.bar(
        x - width / 2,
        shield_exhaustion_pct,
        width,
        label="Legality shield",
    )
    control_bars = ax.bar(
        x + width / 2,
        control_exhaustion_pct,
        width,
        label="Reprompt control",
    )

    ax.set_title(
        "Retry-budget exhaustion falls sharply",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_ylabel("Episodes exhausting retry budget (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(framings)
    ax.set_ylim(0, 65)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    for bars, counts in [
        (shield_bars, shield_exhaustion_counts),
        (control_bars, control_exhaustion_counts),
    ]:
        for bar, count in zip(bars, counts):
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}%\n({count}/120)",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.92),
    )

    fig.suptitle(
        "The shield improves recovery — not per-step decision quality",
        fontsize=17,
        fontweight="bold",
        y=0.99,
    )

    fig.text(
        0.5,
        0.045,
        "Egocentric: 56 fewer retry exhaustions → 24 additional successes",
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.017,
        "Allocentric: 28 fewer retry exhaustions → only 2 additional successes",
        ha="center",
        va="center",
        fontsize=10.5,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.89])
    fig.savefig(OUTPUT_DIR / "08_mechanism.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "08_mechanism.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_env_breakdown(env_data, framing_name, filename):
    labels = [ENV_LABELS[e] for e in ENV_ORDER]

    series = {
        POLICY_LABELS[p].replace("\n", " "): [env_data[e][p] for e in ENV_ORDER]
        for p in POLICY_ORDER
    }

    save_grouped_bar_plot(
        filename=filename,
        title=f"{framing_name} Success by Environment",
        group_labels=labels,
        series_dict=series,
        ylabel="Success (%)",
        ylim=(0, 70),
        annotate=True,
    )


def plot_reasoning_vs_baseline():
    labels = ["Allocentric", "Egocentric"]

    series = {
        "Baseline": [
            reasoning_vs_baseline["allocentric"]["baseline"],
            reasoning_vs_baseline["egocentric"]["baseline"],
        ],
        "Reasoning": [
            reasoning_vs_baseline["allocentric"]["reasoning"],
            reasoning_vs_baseline["egocentric"]["reasoning"],
        ],
    }

    save_grouped_bar_plot(
        filename="11_reasoning_vs_baseline.png",
        title="Reasoning vs Baseline by Framing",
        group_labels=labels,
        series_dict=series,
        ylabel="Success (%)",
        ylim=(0, 30),
        annotate=True,
    )


def plot_cross_model_hint_value():
    labels = ["GPT-4o-mini", "GPT-4.1-mini"]

    series = {
        "Allocentric hint gain": [
            cross_model_hint_value["GPT-4o-mini"]["allocentric"],
            cross_model_hint_value["GPT-4.1-mini"]["allocentric"],
        ],
        "Egocentric hint gain": [
            cross_model_hint_value["GPT-4o-mini"]["egocentric"],
            cross_model_hint_value["GPT-4.1-mini"]["egocentric"],
        ],
    }

    save_grouped_bar_plot(
        filename="12_cross_model_hint_value.png",
        title="Hint Value Replicates Across Models",
        group_labels=labels,
        series_dict=series,
        ylabel="Shield minus control success (pp)",
        ylim=(0, 25),
        annotate=True,
    )



def plot_gazebo_success():
    """
    Slide 10: descriptive Gazebo transfer result.

    Shows success for legality shield vs reprompt control,
    split by allocentric and egocentric framing.

    There are only 5 seeds per condition, so this figure is
    descriptive and intentionally does not show inferential statistics.
    """
    framings = ["Allocentric", "Egocentric"]
    framing_keys = ["allocentric", "egocentric"]

    shield_values = [
        gazebo_success[f]["legality_shield"]
        for f in framing_keys
    ]
    control_values = [
        gazebo_success[f]["reprompt_control"]
        for f in framing_keys
    ]

    shield_counts = [
        gazebo_success_counts[f]["legality_shield"]
        for f in framing_keys
    ]
    control_counts = [
        gazebo_success_counts[f]["reprompt_control"]
        for f in framing_keys
    ]

    x = np.arange(len(framings))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.5, 5.5))

    shield_bars = ax.bar(
        x - width / 2,
        shield_values,
        width,
        label="Legality shield",
    )

    control_bars = ax.bar(
        x + width / 2,
        control_values,
        width,
        label="Reprompt control",
    )

    ax.set_title(
        "Gazebo Transfer: Success by Framing",
        fontsize=15,
        fontweight="bold",
    )
    ax.set_ylabel("Episode success (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(framings)
    ax.set_ylim(0, 75)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend(frameon=False)

    # Label each bar with both percentage and raw success count.
    for bars, counts in [
        (shield_bars, shield_counts),
        (control_bars, control_counts),
    ]:
        for bar, count in zip(bars, counts):
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.5,
                f"{value:.0f}%\n({count}/{GAZEBO_N_PER_CELL})",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Small descriptive-transfer note for presentation use.
    fig.text(
        0.5,
        0.025,
        "5 seeds per condition — descriptive transfer",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout(rect=[0, 0.06, 1, 1])

    fig.savefig(
        OUTPUT_DIR / "13_gazebo_success.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "13_gazebo_success.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    plot_overall_success()
    plot_success_by_framing()
    plot_core_shield_vs_control()
    plot_step_accuracy()
    plot_first_attempt_vs_executed()
    plot_reprompt_firing()
    plot_termination_taxonomy()
    plot_mechanism_slide()
    plot_env_breakdown(allocentric_env, "Allocentric", "09_allocentric_env_breakdown.png")
    plot_env_breakdown(egocentric_env, "Egocentric", "10_egocentric_env_breakdown.png")
    plot_reasoning_vs_baseline()
    plot_cross_model_hint_value()
    plot_gazebo_success()

    print("Saved plots to:", OUTPUT_DIR)
    print("Generated:")
    print("  01_overall_success.png")
    print("  02_success_by_framing.png")
    print("  03_shield_vs_control.png")
    print("  04_step_accuracy.png")
    print("  05_first_attempt_vs_executed.png")
    print("  06_reprompt_firing.png")
    print("  07_termination_taxonomy.png")
    print("  08_mechanism.png")
    print("  08_mechanism.pdf")
    print("  09_allocentric_env_breakdown.png")
    print("  10_egocentric_env_breakdown.png")
    print("  11_reasoning_vs_baseline.png")
    print("  12_cross_model_hint_value.png")
    print("  13_gazebo_success.png")
    print("  13_gazebo_success.pdf")


if __name__ == "__main__":
    main()