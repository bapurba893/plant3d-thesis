"""
physics_loss.py

Shared machinery for Block D rows D5 (Logistic ODE physics loss) and D6
(Gompertz ODE physics loss), both "+ L_mono" per CLAUDE.md. Both rows
reuse D4's exact architecture (PerTermFusionHeadD4) and dataset
(D4FusionDataset/build_scan_table_d4) UNMODIFIED -- D5/D6 change ONLY
the loss function, not the fusion input, per the user's own framing
("D1 through D4 add one component at a time, D5/D6 add the
physics-informed growth constraint").

Design decisions, verified against the primary source docs
(docs/Domain_Invariance_Strategy_Table.docx,
docs/plant_3d_pipeline_v6_corrected.drawio) via a dedicated research pass
BEFORE implementing, per explicit user instruction -- full sourced
findings and design reasoning in step_notes/D5_D6_Physics_Loss.md:

1. **dy/dt: REVISED from autodiff to finite-difference, a recorded and
   documented DEPARTURE from the docs (2026-09-29)**. The docs state
   verbatim (drawio node id="19"): "derivative via autodiff". This was
   implemented first (compute_physics_residual/compute_l_phys_autodiff_
   ISOLATED_PARTIAL below, KEPT, not deleted, as a recorded negative
   finding) and verified correct in isolation to machine precision
   BEFORE being wired in. It was then found, via a synthetic recovery
   test with a KNOWN Logistic solution, to constrain the WRONG quantity:
   autodiff differentiates y w.r.t. the elapsed_days INPUT CHANNEL while
   holding every other input (including prev_height, D4's own lag-1
   feature) FIXED/detached -- an "isolated partial derivative". But
   prev_height is ITSELF a function of time along the real trajectory
   (moving from scan t_k to t_k+1 changes prev_height too), so the
   isolated partial derivative and the TOTAL trajectory derivative
   (finite-difference between the model's own predictions at REAL
   consecutive scans) are DIFFERENT quantities. Measured directly on the
   synthetic recovery test: isolated-partial RMS residual 0.12 (height)
   vs. TOTAL trajectory RMS residual 4.72 (height) -- an ~40x gap. The
   optimizer could trivially minimize the isolated-partial residual
   (via a softplus-bounded rho collapsing toward its floor) without the
   model's REAL predicted growth rate ever becoming physically
   consistent. This is why L_phys now uses the TOTAL trajectory
   derivative (finite-difference on real consecutive scans, same
   once-per-epoch computation style as L_mono) instead of autodiff.
2. **"V" in Gompertz's "y = ln V" is height/stem_diameter, not literal
   volume** -- L_mono is explicitly scoped to height/stem_diameter in
   both docs, and D0's growth_curves.py already only ever fits Logistic/
   Gompertz to those two traits (volume deliberately excluded there,
   since it can legitimately decrease via senescence). Both D5 and D6
   apply their ODE constraint to PHYSICS_TERMS = {height, stem_diameter}
   only, never volume.
3. **rho,K (Logistic) / alpha,beta (Gompertz) are TRAINABLE** (drawio,
   verbatim: "rho,K / alpha,beta trainable") -- not fixed from D0's
   per-plant curve fits. Implemented as GLOBAL nn.Parameter scalars
   (matching this codebase's own Kendall-uncertainty-weighting precedent
   of a small number of global learned scalars), NOT per-plant (would
   need an unstated prediction mechanism to generalize to held-out test
   plants -- a bigger, unspecified architecture change) and NOT
   per-species (a categorical structure nowhere mentioned in either
   doc). SEPARATE scalars per trait (not shared between height and
   stem_diameter), matching D0's own convention of always fitting them
   as two independent curves per plant. Initialized from D0's
   population-average fitted values (a reasonable starting point, not a
   fixed target -- they remain fully trainable throughout D5/D6
   training) rather than an arbitrary guess. The rate parameter (rho for
   Logistic, beta for Gompertz) is softplus-bounded (effective_rate_param)
   to keep it strictly positive -- found necessary (not sufficient on its
   own, see point 1) via the same synthetic recovery test: an
   unconstrained rho collapsed toward/through zero, making the physics
   law vacuous rather than genuinely constraining anything.
4. **L_phys/L_mono are fixed-weight, never Kendall-weighted** -- matches
   CLAUDE.md's established two-regime loss rule (L_dom/L_ent/L_phys/
   L_mono/L_KD are all "fixed/scheduled, NEVER learned").
5. **L_phys and L_mono are BOTH computed once per epoch**, not per
   mini-batch, sharing a SINGLE full-table forward pass
   (run_physics_mono_step) -- both need ordered same-plant scan pairs,
   which D1-D4's per-scan-independent DataLoader/batching can't provide.
   Uses the SAME already-filtered D4 scan table (same rows D4's own
   exclusion logic kept), grouped by plant_id and sorted by elapsed_days.
   L_mono uses consecutive PAIRS within that filtered set as-is (may
   skip an excluded scan -- an already-documented simplification, kept
   unchanged here since a skipped scan doesn't invalidate a
   monotonicity check the way it would a rate estimate). L_phys is
   stricter: a pair is DROPPED if any real scan of that plant (checked
   against the full, unfiltered timeseries table) falls strictly between
   the two elapsed_days values -- an excluded scan straddling the pair
   would silently inflate dt and distort the finite-difference rate
   estimate, which L_mono's monotonicity check doesn't need to worry
   about but L_phys's actual rate computation does.

L_trait/L_growth's per-batch training loop is UNCHANGED from D1-D4 --
D5/D6 reuse train_d1_fusion_baseline.run_epoch directly (no physics-
specific per-batch wrapper needed now that L_phys moved to the
once-per-epoch pass alongside L_mono).
"""

import math

import torch
import torch.nn.functional as F

PHYSICS_TERMS = ["height", "stem_diameter"]
RATE_PARAM_FLOOR = 1e-4


def effective_rate_param(raw_param):
    """softplus-bounded rate parameter (rho for Logistic, beta for
    Gompertz) -- prevents a degenerate rho/beta -> 0 collapse found via
    a synthetic recovery test (2026-09-28, see step_notes/
    D5_D6_Physics_Loss.md): with an UNCONSTRAINED trainable rho, a
    lambda_phys sweep (1e-4 to 10.0) showed the ODE residual dropping
    toward 0 (0.20->0.0008) while rho ALSO dropped toward 0
    (0.028->0.001) instead of the true generating value (0.30) -- the
    optimizer was trivially minimizing L_phys by making the physics law
    itself vacuous (dydt - rho*y*(1-y/K) collapses to just dydt as
    rho->0), not by genuinely recovering the growth law. softplus(raw)
    + floor keeps the effective rate strictly positive, making that
    specific collapse structurally unreachable.

    Necessary but NOT sufficient on its own -- see module docstring
    point 1: even with this bound, the autodiff-based dydt constrained
    the wrong quantity (isolated partial, not total trajectory
    derivative). Kept in the finite-difference redesign too, since a
    degenerate near-zero rate is a risk regardless of how dydt is
    computed."""
    return F.softplus(raw_param) + RATE_PARAM_FLOOR


def inverse_softplus_init(target_value):
    """Raw parameter value x such that effective_rate_param(x) ~=
    target_value -- for initializing a raw rate parameter near a
    desired starting point (e.g. D0's population-fit average) while
    keeping the actual trainable tensor in the UNCONSTRAINED softplus
    domain."""
    y = max(target_value - RATE_PARAM_FLOOR, 1e-6)
    return math.log(math.expm1(y))


def run_l_mono_step(model, table, dataset_class, target_norm_stats, feature_norm_stats,
                     temporal_norm_stats, prev_stage_norm_stats, device, lambda_mono, optimizer=None):
    """Once-per-epoch L_mono-only optimizer step. Retained standalone
    (not folded into run_physics_mono_step) for D4-comparison callers
    and for the synthetic verification scripts that test L_mono in
    isolation; D5/D6's own main() uses run_physics_mono_step instead,
    which computes L_phys and L_mono from a SINGLE forward pass."""
    is_train = optimizer is not None
    model.train(is_train)
    with torch.set_grad_enabled(is_train):
        l_mono = compute_l_mono(model, table, dataset_class, target_norm_stats, feature_norm_stats,
                                 temporal_norm_stats, prev_stage_norm_stats, device)
        if is_train and l_mono.requires_grad:
            optimizer.zero_grad()
            (lambda_mono * l_mono).backward()
            optimizer.step()
    return float(l_mono.item())


def run_physics_mono_step(model, table, dataset_class, target_norm_stats, feature_norm_stats,
                           temporal_norm_stats, prev_stage_norm_stats, full_timeseries_table,
                           ode_family, ode_params, device, lambda_phys, lambda_mono, optimizer=None):
    """Once-per-epoch L_phys (finite-difference, see module docstring
    point 1) + L_mono optimizer step, from a SINGLE shared forward pass
    over the whole (small, ~59-row) train table. optimizer=None -> eval
    mode, diagnostic-only (no update), used for val/test reporting.
    Returns (l_phys_value, l_mono_value)."""
    is_train = optimizer is not None
    model.train(is_train)
    with torch.set_grad_enabled(is_train):
        l_phys = compute_l_phys_finite_diff(model, table, dataset_class, target_norm_stats, feature_norm_stats,
                                             temporal_norm_stats, prev_stage_norm_stats, ode_family, ode_params,
                                             full_timeseries_table, device)
        l_mono = compute_l_mono(model, table, dataset_class, target_norm_stats, feature_norm_stats,
                                 temporal_norm_stats, prev_stage_norm_stats, device)
        total = lambda_phys * l_phys + lambda_mono * l_mono
        if is_train and total.requires_grad:
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
    return float(l_phys.item()), float(l_mono.item())


def compute_l_phys_finite_diff(model, table, dataset_class, target_norm_stats, feature_norm_stats,
                                temporal_norm_stats, prev_stage_norm_stats, ode_family, ode_params,
                                full_timeseries_table, device):
    """ACTIVE L_phys implementation (see module docstring point 1 for
    why this replaced the autodiff version). For each PHYSICS_TERM,
    predicts every remaining scan's value via a NORMAL (static-feature)
    forward pass -- same as compute_l_mono, no differentiable-leaf
    machinery needed since dy/dt is now a finite difference between two
    real predictions, not an autodiff derivative. Groups by plant_id,
    sorts by elapsed_days, and for each CONSECUTIVE pair in that
    (D4-filtered) ordering, checks against `full_timeseries_table`
    (the complete, unfiltered per-scan table) whether any real scan of
    that plant falls strictly between the two elapsed_days values -- if
    so, the pair straddles an excluded scan and is DROPPED (a stricter
    rule than L_mono's, see module docstring point 5). ode_params:
    same {"height": (raw_rate, other), "stem_diameter": (raw_rate,
    other)} convention as the retired autodiff version below."""
    from train_d1_fusion_baseline import ALL_TERMS

    table = table.reset_index(drop=True)
    if len(table) == 0:
        return torch.tensor(0.0, device=device)

    ds = dataset_class(table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    all_feat = torch.stack([torch.tensor(ds[i][0]) for i in range(len(ds))]).to(device)
    preds = model(all_feat)

    residuals = []
    for term in PHYSICS_TERMS:
        idx = ALL_TERMS.index(term)
        mean, std = target_norm_stats[term]
        y_raw = preds[:, idx] * std + mean
        y_for_ode = torch.log(y_raw.clamp(min=1e-6)) if ode_family == "gompertz" else y_raw

        raw_rate, other = ode_params[term]
        rate = effective_rate_param(raw_rate)

        for pid, g in table.groupby("plant_id"):
            g_sorted = g.sort_values("elapsed_days")
            order = g_sorted.index.to_numpy()
            t_vals = g_sorted["elapsed_days"].to_numpy()
            if len(order) < 2:
                continue
            plant_days = full_timeseries_table.loc[full_timeseries_table["plant_id"] == pid, "elapsed_days"].to_numpy()

            y_ordered = y_for_ode[order]
            for k in range(len(order) - 1):
                t_k, t_k1 = t_vals[k], t_vals[k + 1]
                straddled = bool(((plant_days > t_k) & (plant_days < t_k1)).any())
                if straddled:
                    continue
                dt = t_k1 - t_k
                if dt <= 0:
                    continue
                dydt = (y_ordered[k + 1] - y_ordered[k]) / dt
                y_mid = 0.5 * (y_ordered[k] + y_ordered[k + 1])
                if ode_family == "logistic":
                    eps = dydt - rate * y_mid * (1.0 - y_mid / other)
                else:
                    eps = dydt - (other - rate * y_mid)  # alpha - beta*y_log
                residuals.append(eps.unsqueeze(0))

    if not residuals:
        return torch.tensor(0.0, device=device)
    all_eps = torch.cat(residuals)
    return (all_eps ** 2).mean()


def compute_l_mono(model, table, dataset_class, target_norm_stats, feature_norm_stats,
                    temporal_norm_stats, prev_stage_norm_stats, device):
    """Once-per-epoch computation (NOT batched via DataLoader, see module
    docstring point 5): for each plant in `table` (already D4-filtered),
    sorted by elapsed_days, predicts height/stem_diameter for every
    remaining scan via a NORMAL (non-physics, static-feature) forward
    pass -- L_mono only needs predicted VALUES at each timestep, not a
    time-derivative. Penalizes any decrease between CONSECUTIVE scans
    within that plant's remaining (filtered) set, combining both traits
    into one L_mono, matching L_phys's single-term convention."""
    from train_d1_fusion_baseline import ALL_TERMS

    table = table.reset_index(drop=True)
    if len(table) == 0:
        return torch.tensor(0.0, device=device)

    ds = dataset_class(table, target_norm_stats, feature_norm_stats, temporal_norm_stats, prev_stage_norm_stats)
    all_feat = torch.stack([torch.tensor(ds[i][0]) for i in range(len(ds))]).to(device)
    preds = model(all_feat)

    penalties = []
    for term in PHYSICS_TERMS:
        idx = ALL_TERMS.index(term)
        mean, std = target_norm_stats[term]
        y_raw = preds[:, idx] * std + mean
        for pid, g in table.groupby("plant_id"):
            order = g.sort_values("elapsed_days").index.to_numpy()
            if len(order) < 2:
                continue
            y_ordered = y_raw[order]
            diffs = y_ordered[1:] - y_ordered[:-1]
            penalties.append(torch.relu(-diffs))
    if not penalties:
        return torch.tensor(0.0, device=device)
    return torch.cat(penalties).mean()


# =============================================================================
# RETIRED: autodiff-based L_phys (the docs' literal "derivative via
# autodiff" instruction). KEPT, not deleted -- module docstring point 1
# records this as a verified negative finding: the derivative computation
# itself was checked correct to machine precision in isolation, but it
# constrains an ISOLATED PARTIAL derivative (w.r.t. elapsed_days, holding
# prev_height and everything else fixed) that a synthetic recovery test
# showed disagrees with the TOTAL trajectory derivative by ~11-40x RMS.
# Not used by run_physics_mono_step or any active D5/D6 training path --
# kept importable for step_notes/D5_D6_Physics_Loss.md's documented
# comparison and for anyone wanting to reproduce the negative finding.
# =============================================================================

def compute_physics_residual(model, feat_batch, t_mean, t_std, target_norm_stats, term, ode_family):
    """RETIRED (see section banner above) -- autodiff-based dy/dt,
    isolated partial derivative w.r.t. elapsed_days, everything else
    detached. Rebuilds the SAME per-term input slice
    PerTermFusionHeadD4.forward uses internally, but with elapsed_days
    replaced by a FRESH requires_grad=True leaf."""
    from train_d4_fusion_previous_stage import (
        _N_B3B, _GROWTH_RELEVANT_SLICE, _TEMPORAL_COL_IDX, _TERM_TO_PREV_IDX,
    )

    with torch.enable_grad():
        b3b = feat_batch[:, :_N_B3B].detach()
        t_z_batch = feat_batch[:, _TEMPORAL_COL_IDX].detach()
        t_raw = (t_z_batch * t_std + t_mean).clone().requires_grad_(True)
        t_z_fresh = (t_raw - t_mean) / t_std

        prev_idx = _TERM_TO_PREV_IDX[term]
        prev = feat_batch[:, prev_idx:prev_idx + 1].detach()

        if term in _GROWTH_RELEVANT_SLICE:
            growth_slice = feat_batch[:, _GROWTH_RELEVANT_SLICE[term]].detach()
            x = torch.cat([b3b, growth_slice, t_z_fresh.unsqueeze(1), prev], dim=1)
        else:
            x = torch.cat([b3b, t_z_fresh.unsqueeze(1), prev], dim=1)

        y_z = model.heads[term](x).squeeze(1)
        y_mean, y_std = target_norm_stats[term]
        y_raw = y_z * y_std + y_mean

        if ode_family == "logistic":
            y_for_deriv = y_raw
        elif ode_family == "gompertz":
            y_for_deriv = torch.log(y_raw.clamp(min=1e-6))
        else:
            raise ValueError(ode_family)

        dydt = torch.autograd.grad(y_for_deriv.sum(), t_raw, create_graph=True)[0]
    return y_for_deriv, dydt


def logistic_residual(y_raw, dydt, rho, K):
    """eps(t) = dy/dt - rho*y*(1 - y/K), CLAUDE.md's Logistic ODE
    (raw trait space). Used by both the retired autodiff path and the
    active finite-difference path (compute_l_phys_finite_diff inlines
    the same formula for the finite-diff case)."""
    return dydt - rho * y_raw * (1.0 - y_raw / K)


def gompertz_residual(y_log, dydt, alpha, beta):
    """eps(t) = dy/dt - (alpha - beta*y_log), CLAUDE.md's Gompertz ODE
    (log space, y_log = ln(raw trait value))."""
    return dydt - (alpha - beta * y_log)


def compute_l_phys_autodiff_ISOLATED_PARTIAL_retired(model, feat_batch, t_mean, t_std, target_norm_stats,
                                                       ode_family, ode_params):
    """RETIRED (see section banner above) -- kept only for the
    documented isolated-partial-vs-total-trajectory comparison. NOT
    called by run_physics_mono_step or any active training path."""
    residuals = []
    for term in PHYSICS_TERMS:
        y_val, dydt = compute_physics_residual(model, feat_batch, t_mean, t_std, target_norm_stats, term, ode_family)
        raw_rate, other = ode_params[term]
        rate = effective_rate_param(raw_rate)
        if ode_family == "logistic":
            eps = logistic_residual(y_val, dydt, rate, other)
        else:
            eps = gompertz_residual(y_val, dydt, other, rate)
        residuals.append(eps)
    all_eps = torch.cat(residuals)
    return (all_eps ** 2).mean()
