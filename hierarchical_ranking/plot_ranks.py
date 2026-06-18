"""Matplotlib helpers for plotting rank intervals:
* Rank confidence intervals.
* Rank prediction intervals.
* Combined rank confidence and prediction intervals.

Horizontal intervals show each candidate's plausible rank range ``[L, U]`` (rank 1 = worst).
"""

import numpy as np
import pandas as pd

# Constants for plotting
_ENDPOINT_HALF_HEIGHT = 0.3
_PREDICTION_ROW_Y = 0
_PREDICTION_YLIM_LO = -0.5
_DEFAULT_ALIGN_MARGIN = 0.18
_PREDICTION_LINEWIDTH = 4
_ENDPOINT_LINEWIDTH = 1
_INTERVAL_WIDTH_TEXT_POS = (0.05, 0.95)


def _order_ranks(ranks, *, ascending, fixed_order):
    """Add an ``order`` column (y positions 1..n, bottom to top).

    If ``fixed_order`` is set, it lists index labels bottom-to-top. Otherwise
    rows are sorted by interval median rank, with shorter intervals first on ties.
    """
    ranks = ranks.copy()
    if fixed_order is not None:
        fixed_order = list(fixed_order)
        if len(fixed_order) != len(set(fixed_order)):
            raise ValueError('fixed_order contains duplicate labels')
        missing = set(ranks.index) - set(fixed_order)
        extra = set(fixed_order) - set(ranks.index)
        if missing or extra:
            raise ValueError(
                'fixed_order must list exactly the same labels as ranks.index '
                f'(missing from fixed_order: {sorted(missing)}, '
                f'extra in fixed_order: {sorted(extra)})'
            )
        ranks = ranks.reindex(fixed_order)
        ranks['order'] = np.arange(1, len(ranks) + 1)
        return ranks

    ranks['order'] = ((ranks['L'] + ranks['U']) / 2).rank(method='first').astype(int)
    ranks['interval_length'] = ranks['U'] - ranks['L']
    return ranks.sort_values(
        by=['order', 'interval_length'],
        ignore_index=False,
        ascending=[ascending, True],
    )


def _resolve_colors_map(candidate_names, colors_map, default_color):
    """Build a complete label-to-color map, filling gaps with ``default_color``."""
    if colors_map is None:
        return {name: default_color for name in candidate_names}
    return {name: colors_map.get(name, default_color) for name in candidate_names}


def _draw_interval(ax, y, L, U, *, delta, linewidth, color):
    """Draw one horizontal interval with vertical endpoint caps."""
    ax.hlines(y=y, xmin=L - delta, xmax=U + delta, linewidths=linewidth, colors=color)
    ax.vlines(
        x=[L - delta, U + delta],
        ymin=y - _ENDPOINT_HALF_HEIGHT,
        ymax=y + _ENDPOINT_HALF_HEIGHT,
        colors='k',
        linewidth=_ENDPOINT_LINEWIDTH,
    )


def _draw_rank_intervals(ax, ranks, colors_map, *, delta, linewidth):
    """Draw all candidate rank intervals on ``ax``."""
    for name, row in ranks.iterrows():
        _draw_interval(
            ax,
            row['order'],
            row['L'],
            row['U'],
            delta=delta,
            linewidth=linewidth,
            color=colors_map[name],
        )


def _draw_rank_guides(ax, max_xtick):
    """Dashed vertical guides at x-axis rank positions."""
    ax.vlines(
        list(range(1, max_xtick + 1)),
        0,
        1,
        transform=ax.get_xaxis_transform(),
        ls='--',
        color='k',
        alpha=0.1,
        linewidth=0.5,
    )


def _set_x_axis(ax, max_xtick, *, show_ticklabels, show_range, labels_fontsize):
    """Configure rank ticks and x-limits."""
    if show_ticklabels:
        ax.set_xticks(range(1, max_xtick + 1))
        if show_range:
            x_labels = [
                str(i) if i in (1, max_xtick) else ''
                for i in range(1, max_xtick + 1)
            ]
            ax.set_xticklabels(x_labels, fontsize=labels_fontsize)
        else:
            ax.set_xticklabels(range(1, max_xtick + 1), fontsize=labels_fontsize)
    else:
        ax.set_xticks([])
    ax.set_xlim(0, max_xtick + 1)


def _set_y_axis_for_candidates(
    ax, ranks, n_candidates, *, show_ticklabels, show_range, labels_fontsize
):
    """Configure candidate y-ticks when there is no prediction row."""
    ax.set_yticks(range(1, n_candidates + 1))
    if show_ticklabels:
        if show_range:
            y_labels = [
                str(ranks.index[i]) if i in (0, n_candidates - 1) else ''
                for i in range(n_candidates)
            ]
            ax.set_yticklabels(y_labels, fontsize=labels_fontsize)
        else:
            ax.set_yticklabels(ranks.index, fontsize=labels_fontsize)
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(0, n_candidates + 1)


def _configure_axes(
    ax,
    ranks,
    n_candidates,
    max_xtick,
    *,
    xlabel,
    ylabel,
    axis_fontsize,
    show_xticklabels,
    show_yticklabels,
    show_range,
    labels_fontsize,
    has_prediction,
):
    """Set axis titles, x ticks/limits, and candidate y ticks when there is no prediction row."""
    ax.set_xlabel(xlabel, fontsize=axis_fontsize)
    ax.set_ylabel(ylabel, fontsize=axis_fontsize)
    _set_x_axis(
        ax,
        max_xtick,
        show_ticklabels=show_xticklabels,
        show_range=show_range,
        labels_fontsize=labels_fontsize,
    )
    if not has_prediction:
        _set_y_axis_for_candidates(
            ax,
            ranks,
            n_candidates,
            show_ticklabels=show_yticklabels,
            show_range=show_range,
            labels_fontsize=labels_fontsize,
        )


def _annotate_interval_width(ax, interval_width, labels_fontsize):
    """Add average-interval-width text in the upper-left of ``ax``."""
    if interval_width is None:
        return
    ax.text(
        *_INTERVAL_WIDTH_TEXT_POS,
        f'Average width: {round(float(interval_width), 3)}',
        transform=ax.transAxes,
        fontsize=labels_fontsize,
        va='top',
        ha='left',
    )


def _add_legend(
    ax,
    legend_dict,
    *,
    legend_fontsize,
    legend_title_fontsize,
    legend_loc,
    bbox_to_anchor,
    legend_frameon,
):
    """Add a legend from ``handles``, ``labels``, and optional ``title``."""
    title_fs = legend_fontsize if legend_title_fontsize is None else legend_title_fontsize
    legend_kwargs = {
        'handles': legend_dict.get('handles'),
        'labels': legend_dict.get('labels'),
        'fontsize': legend_fontsize,
        'title': legend_dict.get('title'),
        'title_fontsize': title_fs,
        'loc': legend_loc,
    }
    if bbox_to_anchor is not None:
        legend_kwargs['bbox_to_anchor'] = bbox_to_anchor
    if legend_frameon is not None:
        legend_kwargs['frameon'] = legend_frameon
    ax.legend(**legend_kwargs)


def _draw_prediction_interval(
    ax,
    prediction_interval,
    ranks,
    n_candidates,
    *,
    delta,
    prediction_color,
    labels_fontsize,
    show_yticklabels,
    show_candidate_yticklabels,
    prediction_ylabel,
):
    """Draw a prediction interval at y=0 and adjust y-axis ticks."""
    L, U = prediction_interval
    _draw_interval(
        ax,
        _PREDICTION_ROW_Y,
        L,
        U,
        delta=delta,
        linewidth=_PREDICTION_LINEWIDTH,
        color=prediction_color,
    )
    ax.set_ylim(_PREDICTION_YLIM_LO, n_candidates + 1)
    if show_yticklabels:
        ax.set_yticks([_PREDICTION_ROW_Y] + list(range(1, n_candidates + 1)))
        candidate_labels = (
            list(ranks.index)
            if show_candidate_yticklabels
            else [''] * n_candidates
        )
        ax.set_yticklabels(
            [prediction_ylabel] + candidate_labels,
            fontsize=labels_fontsize,
        )
        tick_labels = ax.get_yticklabels()
        if tick_labels:
            tick_labels[0].set_color(prediction_color)
    else:
        ax.tick_params(labelleft=False)


def _style_y_tick_labels(
    ax,
    ranks,
    *,
    has_prediction,
    align_labels,
    align_margin,
    color_labels,
    colors_map,
    label_colors_map,
):
    """Align and/or color y-tick labels after the main plot is drawn."""
    tick_label_colors = colors_map if label_colors_map is None else label_colors_map
    label_models = [None] + list(ranks.index) if has_prediction else list(ranks.index)
    tick_labels_and_models = zip(ax.get_yticklabels(), label_models)

    if align_labels is not None:
        if align_labels not in ('left', 'right'):
            raise ValueError('align_labels must be \'left\', \'right\', or None')
        margin = _DEFAULT_ALIGN_MARGIN if align_margin is None else align_margin
        x_pos = -margin if align_labels == 'left' else margin
        for tick_label, _ in tick_labels_and_models:
            tick_label.set_horizontalalignment(align_labels)
            tick_label.set_x(x_pos)
        tick_labels_and_models = zip(ax.get_yticklabels(), label_models)

    if color_labels:
        for tick_label, model_name in tick_labels_and_models:
            if model_name is not None:
                tick_label.set_color(tick_label_colors[model_name])


def plot_ranks_intervals(
    ranks: pd.DataFrame,
    ax,
    ascending: bool = True,
    fixed_order: list | None = None,
    colors_map: dict | None = None,
    legend_dict: dict | None = None,
    xlabel: str = 'Rank',
    ylabel: str = 'Model',
    max_xtick: int | None = None,
    show_vlines: bool = True,
    delta: float = 0.2,
    show_axis_labels: bool = True,
    show_xticklabels: bool | None = None,
    show_yticklabels: bool | None = None,
    show_candidate_yticklabels: bool = True,
    show_range: bool = False,
    show_grid: bool = False,
    interval_width: float | None = None,
    labels_fontsize: int = 10,
    axis_fontsize: int = 10,
    legend_fontsize: int = 8,
    legend_title_fontsize: int | None = None,
    legend_loc: str = 'best',
    bbox_to_anchor: tuple[float, float] | None = None,
    legend_frameon: bool | None = None,
    linewidth: float = 2,
    default_color: tuple[float, float, float] = (0.8, 0.8, 0.8),
    align_labels: str | None = None,
    align_margin: float | None = None,
    color_labels: bool = False,
    label_colors_map: dict | None = None,
    prediction_interval: tuple[float, float] | None = None,
    prediction_color: str = 'purple',
    prediction_ylabel: str = 'Prediction interval',
):
    """Plot ranking intervals on the given axes.

    Parameters
    ----------
    ranks : DataFrame
        Index labels are candidates (models, tasks, etc.). Must have columns
        ``L`` and ``U`` (lower/upper rank bounds).
    ax : Axes
        Target axes.
    ascending : bool, default True
        When ``fixed_order`` is None, sort rows by median rank ascending
        (best rank at the bottom).
    fixed_order : sequence of labels or None
        Bottom-to-top row order. When set, ``ascending`` is ignored.
    colors_map : dict or None
        Maps index labels to line colors. Missing labels use ``default_color``.
    legend_dict : dict or None
        Optional legend with keys ``handles``, ``labels``, and ``title``.
    prediction_interval : tuple (L, U) or None
        When set, draws a prediction band at y=0 and expands the y-axis.
    prediction_ylabel : str, default 'Prediction interval'
        Y tick label for the prediction row when ``show_yticklabels`` is True.
    show_candidate_yticklabels : bool, default True
        When False, candidate rows keep tick positions but have blank labels.
    align_labels : 'left', 'right', or None
        Shift y-tick labels horizontally (see ``align_margin``).
    align_margin : float or None
        Horizontal offset for aligned labels; negative for left, positive for
        right. Defaults to 0.18 when None.
    color_labels : bool
        Color y-tick labels from ``label_colors_map`` or ``colors_map``.
    show_xticklabels, show_yticklabels : bool or None
        Control rank numbers on the x-axis and candidate names on the y-axis.
        When None, both follow ``show_axis_labels``.
    show_grid : bool, default False
        When False, disable the axes grid (e.g. from a seaborn theme).
    interval_width : float or None
        If set, annotate the axes with the average interval width.
    max_xtick : int or None
        Upper x-axis rank tick; defaults to the number of candidates.
    delta : float
        Horizontal padding added to interval endpoints.
    linewidth : float
        Line width for rank intervals (prediction band uses a thicker width).

    Returns
    -------
    Axes
        The same ``ax`` object, for chaining.
    """
    ranks = _order_ranks(ranks, ascending=ascending, fixed_order=fixed_order)
    n_candidates = len(ranks)
    max_xtick = n_candidates if max_xtick is None else max_xtick
    colors_map = _resolve_colors_map(ranks.index, colors_map, default_color)
    has_prediction = prediction_interval is not None
    show_xticklabels = show_axis_labels if show_xticklabels is None else show_xticklabels
    show_yticklabels = show_axis_labels if show_yticklabels is None else show_yticklabels

    _draw_rank_intervals(ax, ranks, colors_map, delta=delta, linewidth=linewidth)
    if show_vlines:
        _draw_rank_guides(ax, max_xtick)

    _configure_axes(
        ax,
        ranks,
        n_candidates,
        max_xtick,
        xlabel=xlabel,
        ylabel=ylabel,
        axis_fontsize=axis_fontsize,
        show_xticklabels=show_xticklabels,
        show_yticklabels=show_yticklabels,
        show_range=show_range,
        labels_fontsize=labels_fontsize,
        has_prediction=has_prediction,
    )
    _annotate_interval_width(ax, interval_width, labels_fontsize)

    if legend_dict is not None:
        _add_legend(
            ax,
            legend_dict,
            legend_fontsize=legend_fontsize,
            legend_title_fontsize=legend_title_fontsize,
            legend_loc=legend_loc,
            bbox_to_anchor=bbox_to_anchor,
            legend_frameon=legend_frameon,
        )

    if has_prediction:
        _draw_prediction_interval(
            ax,
            prediction_interval,
            ranks,
            n_candidates,
            delta=delta,
            prediction_color=prediction_color,
            labels_fontsize=labels_fontsize,
            show_yticklabels=show_yticklabels,
            show_candidate_yticklabels=show_candidate_yticklabels,
            prediction_ylabel=prediction_ylabel,
        )

    if not show_grid:
        ax.grid(False)

    if show_yticklabels and (align_labels is not None or color_labels):
        _style_y_tick_labels(
            ax,
            ranks,
            has_prediction=has_prediction,
            align_labels=align_labels,
            align_margin=align_margin,
            color_labels=color_labels,
            colors_map=colors_map,
            label_colors_map=label_colors_map,
        )

    return ax
