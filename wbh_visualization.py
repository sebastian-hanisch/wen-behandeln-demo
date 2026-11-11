"""Plotly-Abbildungen. Alle Achsen sind gesperrt (fixedrange), damit Touch-Geräte beim Scrollen nicht am Diagramm hängen bleiben."""

import numpy as np
import plotly.graph_objects as go

import wbh_constants as C


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.22), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_scatter(a):
    """Wahrer gegen geschätzten Effekt je Standort; die Gewählten sind farbig. Die Diagonale ist die fehlerfreie Schätzung."""
    tau, th = a.pop.tau, a.th
    picked = np.zeros(len(tau), dtype=bool)
    picked[a.chosen] = True
    fig = go.Figure()
    lo = float(min(tau.min(), np.nanmin(th) if np.any(~np.isnan(th)) else tau.min()))
    hi = float(max(tau.max(), np.nanmax(th) if np.any(~np.isnan(th)) else tau.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    lo, hi = lo - pad, hi + pad
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color="rgba(120,120,120,0.7)", dash="dash"), name="Schätzung = Wahrheit", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=tau[~picked], y=th[~picked], mode="markers", marker=dict(color=C.COLOR_OTHER, size=7), name="nicht gewählt",
                             hovertemplate="wahr %{x:.2f}, geschätzt %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=tau[picked], y=th[picked], mode="markers", marker=dict(color=C.COLOR_CHOSEN, size=9, line=dict(width=1, color="white")),
                             name=f"gewählt (k = {a.k})", hovertemplate="wahr %{x:.2f}, geschätzt %{y:.2f}<extra></extra>"))
    if a.defined:
        fig.add_hline(y=a.est_mean, line=dict(color=C.COLOR_WARN, width=1), annotation_text="Mittel der Gewählten im Piloten", annotation_position="top left")
        fig.add_vline(x=a.true_mean, line=dict(color=C.COLOR_TRUTH, width=1), annotation_text="wahres Mittel der Gewählten", annotation_position="bottom right")
    fig.update_xaxes(title_text="wahrer Effekt (Prozentpunkte Servicegrad)", range=[lo, hi])
    fig.update_yaxes(title_text="geschätzter Effekt (Prozentpunkte)", range=[lo, hi])
    return _base(fig, 420)


def build_curve(m_grid, rows, ytitle, live_x=None, live_y=None, live_name="Live-Wiederholungen", hline=None, name="Median über 20 Populationen"):
    """Kurve über den Pilotumfang m mit Band (Minimum bis Maximum über die Populationen). `rows`: Liste von [Median, Minimum, Maximum] oder None."""
    xs = [str(m) for m, r in zip(m_grid, rows) if r is not None]
    rr = [r for r in rows if r is not None]
    fig = go.Figure()
    if rr:
        fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[r[2] for r in rr] + [r[1] for r in rr][::-1], fill="toself", fillcolor="rgba(76,120,168,0.18)",
                                 line=dict(width=0), hoverinfo="skip", name="Minimum bis Maximum"))
        fig.add_trace(go.Scatter(x=xs, y=[r[0] for r in rr], mode="lines+markers", line=dict(color=C.COLOR_LINE, width=3), name=name,
                                 hovertemplate="m = %{x}: %{y:.2f}<extra></extra>"))
    if live_x is not None and live_y is not None and not np.isnan(live_y):
        fig.add_trace(go.Scatter(x=[str(live_x)], y=[live_y], mode="markers", marker=dict(color=C.COLOR_TRUTH, size=13, symbol="diamond"), name=live_name,
                                 hovertemplate="m = %{x}: %{y:.2f}<extra></extra>"))
    if hline is not None:
        fig.add_hline(y=hline, line=dict(color="rgba(120,120,120,0.8)", dash="dash"))
    fig.update_xaxes(title_text="Pilotumfang m (Fenster je Standort und Arm)", type="category", categoryorder="array", categoryarray=[str(m) for m in m_grid])
    fig.update_yaxes(title_text=ytitle)
    return _base(fig)


def build_bars(rows, live=None):
    """Fang-Anteil je Schätzer (Median über die Populationen mit Fehlerbalken Minimum bis Maximum). rows: [(est, median, min, max)]."""
    labels = [C.EST_SHORT[r[0]] for r in rows]
    fig = go.Figure(go.Bar(x=labels, y=[r[1] for r in rows], marker_color=C.COLOR_LINE,
                           error_y=dict(type="data", symmetric=False, array=[r[3] - r[1] for r in rows], arrayminus=[r[1] - r[2] for r in rows]),
                           hovertemplate="%{x}: %{y:.0f} %<extra></extra>", name="Median und Spanne"))
    if live is not None:
        fig.add_trace(go.Scatter(x=[C.EST_SHORT[live[0]]], y=[live[1]], mode="markers", marker=dict(color=C.COLOR_TRUTH, size=13, symbol="diamond"), name="Anzeige oben (ein Pilot)"))
    fig.update_yaxes(title_text="Fang-Anteil (%): Orakel = 100, Zufall = 0", range=[min(0, min(r[2] for r in rows) - 5), 105])
    fig.update_xaxes(title_text="")
    return _base(fig)


def build_rho(pop):
    """Streuung der Einzeldifferenz gegen den wahren Effekt je Standort."""
    fig = go.Figure(go.Scatter(x=pop.tau, y=pop.sd, mode="markers", marker=dict(color=C.COLOR_LINE, size=7), hovertemplate="Effekt %{x:.2f}, Streuung %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title_text="wahrer Effekt (Prozentpunkte)")
    fig.update_yaxes(title_text="Streuung der Einzeldifferenz (Prozentpunkte)")
    return _base(fig, 320)
