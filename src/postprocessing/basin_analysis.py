import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))


def convert_mm_month_to_discharge_m3_month(da, river_mask, cell_area):
    snow_melt_m = da / 1000  # mm → m
    return snow_melt_m * cell_area * river_mask


def monthly_discharge_sum_m3(da):
    total_volume_monthly = da.sum(dim=["lat", "lon"])
    return total_volume_monthly


def plot_annual_sums_two_bars(
    scenario, control, river, labels=["Scenario", "Control"], ax=None
):
    n_years = len(scenario)
    year_indices = np.arange(n_years)

    bar_width = 0.4
    offset = bar_width / 2

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
        standalone = True
    else:
        standalone = False

    control_mean = control.mean(dim="year")

    ax.bar(
        year_indices - offset,
        scenario.values,
        width=bar_width,
        color="steelblue",
        alpha=0.8,
        label=labels[0],
    )
    ax.bar(
        year_indices + offset,
        control_mean.values,
        width=bar_width,
        color="grey",
        alpha=0.85,
        label=labels[1],
    )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(year_indices)
    ax.set_xticklabels(year_indices, fontsize=5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Annual Snowmelt [m³]")
    ax.legend(fontsize=8)
    ax.set_title(f"{river}")

    if standalone:
        plt.tight_layout()
        plt.show()


def plot_annual_anomaly(
    annual_scenario_ts,
    annual_control_ts,
    river,
    ax=None,
    title=None,
    save_dir="./results/basin",
):
    # annual_control_ts kann entweder die gemittelte Referenz (Skalar)
    # oder die vollständige Zeitreihe sein – wir brauchen beides
    # Daher: Referenzwert = Mittelwert, SD aus der vollen Zeitreihe
    control_mean = annual_control_ts.mean(dim="year")
    control_std = annual_control_ts.std(dim="year")

    # SD in Prozent des Mittelwerts
    sd_pct = (control_std / control_mean * 100).values

    # Anomalie in Prozent relativ zum Mittelwert
    anomaly = (annual_scenario_ts - control_mean) / control_mean * 100
    n_years = len(anomaly)
    year_indices = np.arange(n_years)
    values = anomaly.values

    pos_vals = np.where(values >= 0, values, 0)
    neg_vals = np.where(values < 0, values, 0)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.06)
        standalone = True
    else:
        standalone = False

    # ±1 SD als gestrichelte Linien
    ax.axhline(
        sd_pct,
        color="#E34444",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )
    ax.axhline(
        -sd_pct,
        color="#E34444",
        linewidth=1.0,
        linestyle="--",
        zorder=0,
    )

    ax.bar(year_indices, pos_vals, color="#4575b4", alpha=0.8)
    ax.bar(year_indices, neg_vals, color="#4575b4", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.8)

    ax.set_xticks(year_indices)
    ax.set_xticklabels(year_indices, fontsize=5)
    ax.set_xlabel("Time [years]")
    ax.set_ylabel(f"{title} Anomaly (%)")
    ax.set_title(f"{river}")

    if standalone:
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{river}_annual_{title}_sum_anomaly.png"), dpi=600
        )
        plt.close(fig)


def plot_monthly_bars_rain_snow(
    scenario_snowmelt,
    scenario_rain,
    control_snowmelt,
    control_rain,
    river,
    ax=None,
    save_dir="./results/basin",
):
    standalone = False
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
        standalone = True

    n_months = len(scenario_snowmelt)
    x = np.arange(n_months)

    for j in x:
        ax.axvline(j, color="lightgrey", linewidth=0.2)
    jan_ticks = np.arange(0, n_months, 12)
    for j in jan_ticks:
        ax.axvline(j, color="grey", linewidth=0.5)

    color_control = "#E34444"
    color_scenario = "#4575b4"

    scale = 1e3  # discharge in 10^3 m^3 s^-1, keeps tick labels short
    scenario_rain_scaled = scenario_rain.values / scale
    scenario_snowmelt_scaled = scenario_snowmelt.values / scale
    control_rain_scaled = control_rain.values / scale
    control_snowmelt_scaled = control_snowmelt.values / scale

    ax.bar(
        x,
        scenario_rain_scaled,
        width=1.0,
        color=color_scenario,
        alpha=0.8,
        label="Scenario Rain",
    )
    ax.bar(
        x,
        scenario_snowmelt_scaled,
        bottom=scenario_rain_scaled,
        width=1.0,
        color=color_scenario,
        alpha=0.4,
        label="Scenario Snowmelt",
    )
    ax.bar(
        x,
        -control_rain_scaled,
        width=1.0,
        color=color_control,
        alpha=0.8,
        label="Control Rain",
    )
    ax.bar(
        x,
        -control_snowmelt_scaled,
        bottom=-control_rain_scaled,
        width=1.0,
        color=color_control,
        alpha=0.4,
        label="Control Snowmelt",
    )

    ax.axhline(0, color="black", linewidth=0.8)
    yticks = ax.get_yticks()
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{abs(y):g}" for y in yticks])
    ax.set_xticks(jan_ticks)
    ax.set_xticklabels(
        [f"Year {i}" for i in range(len(jan_ticks))],
        rotation=45,
        ha="right",
        fontsize=7,
    )
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Time [months]", fontsize=9)
    ax.set_ylabel("Discharge (10³ m³ s⁻¹)", fontsize=10)
    ax.set_title(river, fontsize=13)

    if ax is ax.get_figure().axes[0]:
        ax.legend(fontsize=6)

    if standalone:
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{river}_monthly_bars_rain_snow.png"), dpi=350
        )
        plt.close()


# Ergänzung für das basin_analysis-Modul: Imports (os, numpy, cftime,
# matplotlib.pyplot) sind dort bereits vorhanden.

PHASES = [
    ("Years 1–4", 1, 4),
    ("Years 5–8", 5, 8),
    ("Years 9–14", 9, 14),
]
PHASE_COLORS = ["#07285a", "#4575b4", "#abd9e9"]


def seasonal_cycle_by_phase(da, phases=PHASES):
    # Mittlerer Jahresgang je Phase; Jahr 0 = erstes Jahr der Zeitreihe.
    # Hinweis: PHASES-Grenzen sind aktuell 1-indiziert (z.B. "Years 1-4" = (1, 4)),
    # rel_year hier aber 0-indiziert -> das erste Analysejahr (rel_year 0) faellt
    # aus allen Phasen heraus. Noch nicht korrigiert, siehe Ruecksprache.
    # Die Zeitverschiebung passiert bereits einmalig via change_time() in main.py,
    # bevor diese Funktion ihre Eingabedaten erhaelt - hier also nicht nochmal verschieben.
    rel_year = da["time.year"] - da["time.year"].min()

    cycles = []
    for _, first, last in phases:
        sel = da.where((rel_year >= first) & (rel_year <= last), drop=True)
        cycles.append(sel.groupby("time.month").mean("time"))
    return cycles


def seasonal_cycle_control(da):
    # Mittelwert und Standardabweichung je Kalendermonat über alle Kontrolljahre.
    # Die Zeitverschiebung passiert bereits einmalig via change_time() in main.py,
    # bevor diese Funktion ihre Eingabedaten erhaelt - hier also nicht nochmal verschieben.
    grouped = da.groupby("time.month")
    return grouped.mean("time"), grouped.std("time")


def plot_seasonal_cycle_phases(
    scenario_snowmelt,
    scenario_rain,
    control_snowmelt,
    control_rain,
    river,
    axes=None,
    phases=PHASES,
    save_dir="./results/basin",
    show_component_titles=False,
    show_component_ylabels=True,
    show_river_title=True,
    show_xticklabels=True,
    show_legend=True,
):
    standalone = False
    if axes is None:
        fig, axes = plt.subplots(3, 1, figsize=(4, 9), sharex=True)
        standalone = True

    months = np.arange(1, 13)
    month_labels = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]

    scenario_total = scenario_snowmelt + scenario_rain
    control_total = control_snowmelt + control_rain

    components = [
        (axes[0], scenario_snowmelt, control_snowmelt, "Snowmelt"),
        (axes[1], scenario_rain, control_rain, "Rain"),
        (axes[2], scenario_total, control_total, "Total"),
    ]

    scale = 1e3  # discharge in 10^3 m^3 s^-1, keeps tick labels short

    for ax, scenario, control, label in components:
        control_mean, control_std = seasonal_cycle_control(control)

        # Kontrolle: Mittelwert ±1 SD
        ax.fill_between(
            months,
            (control_mean - control_std).values / scale,
            (control_mean + control_std).values / scale,
            color="lightgrey",
            alpha=0.3,
            linewidth=0,
            label="Control ±1σ",
        )
        ax.plot(
            months,
            control_mean.values / scale,
            color="dimgrey",
            linewidth=1.2,
            linestyle="--",
            label="Control mean",
        )

        # Szenario: ein Jahresgang je Phase
        for (phase_label, _, _), cycle, color in zip(
            phases,
            seasonal_cycle_by_phase(scenario, phases),
            PHASE_COLORS,
        ):
            ax.plot(
                months,
                cycle.values / scale,
                color=color,
                linewidth=1.4,
                label=phase_label,
            )

        ax.set_xticks(months)
        ax.set_xticklabels(month_labels if show_xticklabels else [], fontsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, integer=True))
        # Etwas Kopffreiheit, damit der oberste Tick nicht an Panel-Label/Zeilenrand stoesst
        ax.margins(y=0.08)
        ax.set_ylim(bottom=0)

        if show_component_titles:
            ax.set_title(label, fontsize=10, fontweight="bold")
        if show_component_ylabels:
            ax.set_ylabel(f"{label} discharge (10³ m³ s⁻¹)", fontsize=9)

    if show_river_title:
        axes[0].set_title(river, fontsize=10)

    if show_legend:
        axes[0].legend(fontsize=6)

    if standalone:
        plt.tight_layout()
        plt.savefig(
            os.path.join(save_dir, f"{river}_seasonal_cycle_phases.png"), dpi=600
        )
        plt.close(fig)
