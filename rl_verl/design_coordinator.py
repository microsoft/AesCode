"""DesignValidator coordinator — Generate-only Swiss tournament.

This module owns the **cross-rollout** synchronization layer for the
Generate sub-step, the only path that needs an N-rollout group barrier.

Design goals:

1.  Three deadlines, **not** one:
      - assemble_deadline  : how long to wait for the Nth member to register.
      - score_deadline     : how long the Swiss tournament itself may take.
      - total_deadline     : absolute waiter cap (assemble + score + slack).

2.  Per-group state machine — never per-member timeout:
      OPEN  →  READY  →  SCORING  →  FINALIZED  →  CLOSED.
    Early arrivers and late arrivers share the same group-level deadline,
    so we cannot have "early arrivals timed out before late arrivals scored".

3.  Orphan-group safe.  After a group is FINALIZED its key is kept in a
    short TTL cache.  Any straggler hitting the same key inside the TTL
    is *absorbed* with a degraded result + ``infra_late_group`` reason
    code.  We never re-create a group for the same key while the cache
    is hot.

4.  m < N policy with confidence shrinkage:
      - m = 0  → all scores 0.5, ``infra_no_members``  (no information).
      - m = 1  → single member gets prior 0.5, ``low_m_prior``.
      - m >= 2 → real Swiss; the *missing* slots get None (mean fill).
    Each result carries ``confidence = m / N``.  Failed members are
    **not** mean-filled — they get the configured failure value with
    a model/infra reason code, so we never silently average failures
    into the surviving rollouts.

5.  Every behaviour is traceable: each waiter receives a structured
    ``DesignResult`` with ``score``, ``confidence``, ``reason_code``,
    ``fire_reason``, ``applicable``.

The coordinator is a *process-local* singleton, just like the legacy
``DesignScoreBarrier`` was — verl's AgentLoopWorker batches all rollouts
of one sample into the same worker via ``asyncio.gather``, so a
per-worker coordinator is enough.  If we ever need cross-worker
coordination we can promote it to a named detached Ray actor without
touching the call sites.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reason codes — keep these short, stable enums.  They are logged into
# wandb via reward_extra_info["design_reason_code"], so renaming will
# break dashboards.
# ---------------------------------------------------------------------------

class DesignReason:
    OK                  = "ok"
    LOW_M_PRIOR         = "low_m_prior"          # m<2 but result is still usable
    MODEL_EMPTY_HTML    = "model_empty_html"     # this rollout produced no HTML
    INFRA_RENDER_FAILED = "infra_render_failed"  # playwright failed
    INFRA_NO_MEMBERS    = "infra_no_members"     # m == 0 in this group
    INFRA_INCOMPLETE_GROUP = "infra_incomplete_group"  # assemble_timeout with m < N
    INFRA_LATE_GROUP    = "infra_late_group"     # arrived after group closed
    INFRA_VLM_TIMEOUT   = "infra_vlm_timeout"    # swiss tournament timed out
    INFRA_VLM_ERROR     = "infra_vlm_error"      # tournament raised
    INFRA_TOTAL_TIMEOUT = "infra_total_timeout"  # waiter hit absolute deadline


@dataclass
class DesignResult:
    """Structured result handed to every waiter.

    ``score`` is in [0,1] (Generate ranking).  For infra failures we
    return a neutral 0.5 with applicable=0 so the GDPO advantage for
    this dim collapses (group mean → zero advantage), instead of
    silently feeding noise into training.  For model failures
    (empty HTML) we return 0.0 with applicable=1 — the model *should*
    be penalised for producing nothing.
    """
    score: float = 0.5
    confidence: float = 0.0
    reason_code: str = DesignReason.OK
    fire_reason: str = "unknown"      # "full" | "assemble_timeout" | "late_absorb"
    applicable: int = 1               # 0 → mask in training (zero advantage)


# ---------------------------------------------------------------------------
# Tunables (env-overridable, kept on the coordinator instance)
# ---------------------------------------------------------------------------

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

class GroupState:
    OPEN       = "OPEN"
    READY      = "READY"
    SCORING    = "SCORING"
    FINALIZED  = "FINALIZED"
    CLOSED     = "CLOSED"


@dataclass
class _Group:
    key: tuple
    expected_n: int
    ref_path: Optional[str]
    created_at: float = 0.0
    members: list[dict] = field(default_factory=list)
    state: str = GroupState.OPEN
    assemble_deadline: float = 0.0
    score_deadline: float = 0.0
    total_deadline: float = 0.0
    fire_reason: str = "unknown"
    fired_event: asyncio.Event = field(default_factory=asyncio.Event)
    results: list[DesignResult] = field(default_factory=list)
    deadline_task: Optional[asyncio.Task] = None


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

class GenerateSwissCoordinator:
    """Process-local Generate-mode coordinator.

    Public API is intentionally tiny:

        coord = get_generate_coordinator()
        result = await coord.submit_and_wait(
            sample_id="abc", sub_step_idx=0,
            ref_path="/p/ref.png", render_path="/p/r3.png",
            model_failed=False,
        )
        # result is a DesignResult.
    """

    def __init__(self):
        self._lock = asyncio.Lock()
        self._groups: dict[tuple, _Group] = {}
        self._recently_closed: dict[tuple, float] = {}
        # Per-prefix epoch counter — the coordinator allocates a fresh
        # epoch every time a prior group for the same (sample, sub_step)
        # has transitioned out of OPEN (READY / SCORING / FINALIZED /
        # recently-closed).  This makes cross-step namespacing automatic:
        # callers never need to thread a ``global_step`` token through
        # the reward pipeline.  See ``_resolve_active_key``.
        self._epoch_by_prefix: dict[tuple, int] = {}
        # Lazy-started heartbeat task — see ``_ensure_heartbeat``.
        self._heartbeat_task: Optional[asyncio.Task] = None

        self.expected_n         = _env_int("ROLLOUT_N", 8)
        # Assemble timeout is a *watchdog fallback* only.
        # Primary trigger remains explicit full-N completion; this timeout
        # exists to break rare deadlocks (worker crash / host fault / lost slot).
        # <=0 disables assemble-timeout fire.
        self.assemble_timeout   = _env_float("DESIGN_GEN_ASSEMBLE_TIMEOUT", 3600.0)
        self.score_timeout      = _env_float("DESIGN_GEN_SCORE_TIMEOUT", 360.0)
        # total = assemble + score + slack by default
        self.total_timeout      = _env_float(
            "DESIGN_GEN_TOTAL_TIMEOUT", self.assemble_timeout + self.score_timeout + 600.0
        )
        self.closed_ttl         = _env_float("DESIGN_GEN_CLOSED_TTL", 120.0)
        # Confidence shrinkage prior (m < min_members → blend with prior).
        self.prior              = _env_float("DESIGN_PRIOR", 0.5)
        self.min_members        = _env_int("DESIGN_MIN_MEMBERS", 2)

        # Per-mode failure values (model-failure must hurt, infra-failure must not).
        self.model_fail_value   = _env_float("DESIGN_MODEL_FAIL_VALUE", 0.0)
        self.infra_fail_value   = _env_float("DESIGN_INFRA_FAIL_VALUE", 0.5)

        # Heartbeat cadence (set <=0 to disable).  Logs a one-line snapshot of
        # group-state histogram + oldest-OPEN age + closed-TTL cache size so
        # an operator watching stdout can immediately tell whether the
        # coordinator itself is the source of an apparent training hang
        # (vs. vLLM, FSDP, Playwright, the judge endpoint, etc.).
        self.heartbeat_interval = _env_float("DESIGN_GEN_HEARTBEAT_INTERVAL", 60.0)

        logger.info(
            "[GenerateSwissCoordinator] init expected_n=%d "
            "assemble=%.0fs score=%.0fs total=%.0fs closed_ttl=%.0fs "
            "prior=%.2f min_members=%d heartbeat=%.0fs",
            self.expected_n, self.assemble_timeout, self.score_timeout,
            self.total_timeout, self.closed_ttl, self.prior, self.min_members,
            self.heartbeat_interval,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _prefix(sample_id: str, sub_step_idx: int) -> tuple:
        """Stable (sample, sub_step) prefix shared across training steps."""
        return (str(sample_id), int(sub_step_idx))

    def _resolve_active_key(self, prefix: tuple, global_step: Optional[int] = None) -> tuple:
        """Return the group key for ``prefix``.

        MUST be called under ``self._lock``.

        When ``global_step`` is provided (preferred), the key is simply
        ``prefix + (global_step,)`` — all rollouts from the same training
        step deterministically land in the same group, regardless of arrival
        timing.  This eliminates the epoch-split bug where early arrivals
        trigger assemble_timeout, and late arrivals from the *same* step
        accidentally create a new group.

        When ``global_step`` is None (legacy fallback), the old epoch-bump
        behaviour is used: a fresh epoch is allocated whenever no OPEN group
        exists for this prefix.
        """
        if global_step is not None:
            # Deterministic key — no epoch counter needed.
            key = prefix + (int(global_step),)
            return key

        # Legacy fallback: epoch-bump (kept for backward compat).
        epoch = self._epoch_by_prefix.get(prefix, 0)
        key = prefix + (epoch,)
        g = self._groups.get(key)
        # Happy path: an OPEN group exists for this prefix — join it.
        if g is not None and g.state == GroupState.OPEN:
            return key
        # The previous group is already non-OPEN (firing or finalised)
        # or its key is still in the recently-closed TTL cache.  Bump.
        if g is not None or key in self._recently_closed:
            epoch += 1
            self._epoch_by_prefix[prefix] = epoch
            key = prefix + (epoch,)
        return key

    @staticmethod
    def _key(sample_id: str, sub_step_idx: int, group_token: Optional[str] = None) -> tuple:
        """DEPRECATED — kept only for tests / external callers that still
        build keys manually.  The live code path uses
        ``_resolve_active_key`` instead, which manages epochs internally.
        """
        token = "_" if group_token is None else str(group_token)
        return (str(sample_id), int(sub_step_idx), token)

    def _gc_closed(self, now: float) -> None:
        if not self._recently_closed:
            return
        for k in [k for k, exp in self._recently_closed.items() if exp <= now]:
            self._recently_closed.pop(k, None)

    # ------------------------------------------------------------------
    # Heartbeat — operator-visible health snapshot.  Logged at INFO level
    # so it shows up in the standard training stdout.  All snapshot work
    # is read-only over the live dicts; we don't take ``self._lock`` to
    # keep the heartbeat resilient even if a future bug makes the lock
    # itself the source of stalls (we'd still see the heartbeat fire).
    # ------------------------------------------------------------------
    def _ensure_heartbeat(self) -> None:
        """Lazily spawn the heartbeat task on the live event loop.

        Called from ``submit_and_wait`` so we can be sure a loop exists.
        Idempotent — only one task per coordinator instance.  If the task
        ever crashes we let it die silently rather than break training.
        """
        if self.heartbeat_interval <= 0:
            return
        if self._heartbeat_task is not None and not self._heartbeat_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return  # no loop yet; will retry on next submit
        self._heartbeat_task = loop.create_task(self._heartbeat_loop())

    async def _heartbeat_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                self._emit_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception:
                # Heartbeat must never crash the coordinator.
                logger.exception("[GenCoord HEARTBEAT] snapshot failed")

    def _emit_heartbeat(self) -> None:
        now = time.monotonic()
        # Snapshot the dicts under a single Python statement (CPython GIL
        # makes dict.values() iteration atomic enough for a histogram).
        states_count: dict[str, int] = {}
        oldest_open_age = 0.0
        oldest_open_key: Optional[tuple] = None
        members_inflight = 0
        for g in list(self._groups.values()):
            states_count[g.state] = states_count.get(g.state, 0) + 1
            members_inflight += len(g.members)
            if g.state == GroupState.OPEN:
                age = now - g.created_at
                if age > oldest_open_age:
                    oldest_open_age = age
                    oldest_open_key = g.key

        closed_cache = len(self._recently_closed)
        active_groups = len(self._groups)

        if active_groups == 0 and closed_cache == 0:
            # Nothing to report — stay quiet to avoid log spam during idle
            # windows (e.g. between training ticks).
            return

        # Stable, grep-friendly format.
        parts = [
            f"active_groups={active_groups}",
            f"members_inflight={members_inflight}",
            f"closed_cache={closed_cache}",
        ]
        for st in (GroupState.OPEN, GroupState.READY, GroupState.SCORING,
                   GroupState.FINALIZED, GroupState.CLOSED):
            parts.append(f"{st.lower()}={states_count.get(st, 0)}")
        if oldest_open_key is not None:
            parts.append(
                f"oldest_open_age={oldest_open_age:.1f}s "
                f"oldest_open_key={oldest_open_key}"
            )
        logger.info("[GenCoord HEARTBEAT] %s", "  ".join(parts))

    # ------------------------------------------------------------------
    async def submit_and_wait(
        self,
        sample_id: str,
        sub_step_idx: int,
        ref_path: Optional[str],
        render_path: Optional[str],
        model_failed: bool = False,
        group_token: Optional[str] = None,  # IGNORED — kept for back-compat
        global_step: Optional[int] = None,
    ) -> DesignResult:
        """Register a Generate rollout; block until the group finishes.

        ``global_step`` (preferred): when provided, the group key is
        deterministically ``(sample_id, sub_step_idx, global_step)``.
        All rollouts from the same training step land in the same group
        regardless of arrival timing, eliminating the epoch-split bug.

        ``group_token`` is accepted for back-compat but **ignored**.
        """
        prefix = self._prefix(sample_id, sub_step_idx)
        now = time.monotonic()

        # Lazily start the operator-visible heartbeat on the live loop.
        self._ensure_heartbeat()

        async with self._lock:
            self._gc_closed(now)
            key = self._resolve_active_key(prefix, global_step=global_step)

            # When global_step is provided, the group key is deterministic.
            # A non-OPEN group means this rollout arrived after assemble_timeout
            # already fired — it's a genuine late arrival, not an epoch-split.
            # The state-guarded append below handles this via LATE absorb.

            # Orphan-safe guard: after a group is finalized we keep its key in
            # _recently_closed for a short TTL. A late arrival during this
            # window must be absorbed instead of re-creating a fresh group with
            # the same deterministic key.
            if key not in self._groups and key in self._recently_closed:
                logger.warning(
                    "[GenCoord] LATE absorb key=%s — key in recently_closed cache",
                    key,
                )
                return DesignResult(
                    score=self.infra_fail_value,
                    confidence=0.0,
                    reason_code=DesignReason.INFRA_LATE_GROUP,
                    fire_reason="late_absorb",
                    applicable=0,
                )

            if key not in self._groups:
                assemble_deadline = (now + self.assemble_timeout) if self.assemble_timeout > 0 else float("inf")
                total_deadline    = (now + self.total_timeout) if self.total_timeout > 0 else float("inf")
                g = _Group(
                    key=key,
                    expected_n=self.expected_n,
                    ref_path=ref_path,
                    created_at=now,
                    assemble_deadline=assemble_deadline,
                    score_deadline=0.0,        # set on fire
                    total_deadline=total_deadline,
                )
                if self.assemble_timeout > 0:
                    g.deadline_task = asyncio.create_task(
                        self._force_fire_at_deadline(key, assemble_deadline)
                    )
                self._groups[key] = g

            g = self._groups[key]

            # State-guarded append.  Only OPEN groups may accept new members.
            # If we observe a non-OPEN group here it means the group has
            # already started scoring (assemble_timeout fired between this
            # waiter creating the group and acquiring the lock, or a
            # previous fire is still in-flight before _groups.pop).  Absorb
            # the rollout as a late arrival instead of corrupting member
            # count vs results length (HIGH-2 in the review).
            if g.state != GroupState.OPEN:
                logger.warning(
                    "[GenCoord] LATE absorb key=%s — group state=%s (not OPEN)",
                    key, g.state,
                )
                return DesignResult(
                    score=self.infra_fail_value,
                    confidence=0.0,
                    reason_code=DesignReason.INFRA_LATE_GROUP,
                    fire_reason="late_absorb",
                    applicable=0,
                )

            slot = len(g.members)
            g.members.append({
                "render_path":   render_path,
                "model_failed":  bool(model_failed),
                "slot":          slot,
            })

            should_fire = len(g.members) >= g.expected_n
            if should_fire:
                g.state = GroupState.READY
                g.fire_reason = "full"
                if g.deadline_task and not g.deadline_task.done():
                    g.deadline_task.cancel()
                asyncio.create_task(self._fire_group(key))

            wait_until = g.total_deadline

        # ─── Wait on the group-level deadline.  All members of the same
        # group share this deadline, so early arrivals never time out
        # before late arrivals score.
        if self.total_timeout > 0:
            remaining = max(0.5, wait_until - time.monotonic())
            try:
                await asyncio.wait_for(g.fired_event.wait(), timeout=remaining)
            except asyncio.TimeoutError:
                logger.error(
                    "[GenCoord] TOTAL_TIMEOUT key=%s slot=%d/%d after %.0fs",
                    key, slot + 1, self.expected_n, self.total_timeout,
                )
                return DesignResult(
                    score=self.infra_fail_value,
                    confidence=0.0,
                    reason_code=DesignReason.INFRA_TOTAL_TIMEOUT,
                    fire_reason=g.fire_reason or "total_timeout",
                    applicable=0,
                )
        else:
            await g.fired_event.wait()

        # Group has been finalized.  Return our slot's result.
        if slot < len(g.results):
            return g.results[slot]

        # Defensive: results list shorter than members.  Should never happen
        # because _fire_group always pads to len(members).  Treat as infra.
        logger.error("[GenCoord] inconsistent results len for key=%s", key)
        return DesignResult(
            score=self.infra_fail_value,
            confidence=0.0,
            reason_code=DesignReason.INFRA_VLM_ERROR,
            fire_reason=g.fire_reason,
            applicable=0,
        )

    # ------------------------------------------------------------------
    async def _force_fire_at_deadline(self, key: tuple, deadline_ts: float) -> None:
        # ── Wait for full assemble deadline ──
        # NOTE: previously had an "early_fire" branch (DESIGN_GEN_EARLY_FIRE_TIMEOUT)
        # that fired after 30s if len < expected_n, intended to unblock val (n=1).
        # That branch also killed training groups where 8 rollouts arrived over
        # several minutes (vLLM queuing + verifier render latency), causing ~90%
        # of design scores to be masked.  Val now skips DesignValidator via
        # ikw["_is_val"] (see rl_verl/reward_fn.py + multi_step_agent.py), so the
        # early_fire workaround is no longer needed.
        delay = max(0.0, deadline_ts - time.monotonic())
        if delay > 0:
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

        async with self._lock:
            g = self._groups.get(key)
            if g is None or g.state != GroupState.OPEN:
                return
            g.state = GroupState.READY
            g.fire_reason = "assemble_timeout"
            logger.warning(
                "[GenCoord] assemble_timeout fire key=%s m=%d/%d",
                key, len(g.members), g.expected_n,
            )

        # Fire outside the lock (it will reacquire as needed).
        asyncio.create_task(self._fire_group(key))

    # ------------------------------------------------------------------
    async def _fire_group(self, key: tuple) -> None:
        """Run the Swiss tournament and finalize the group."""
        async with self._lock:
            g = self._groups.get(key)
            if g is None or g.state not in (GroupState.READY,):
                return
            g.state = GroupState.SCORING
            g.score_deadline = time.monotonic() + self.score_timeout

        try:
            results = await self._score_group(g)
        except asyncio.TimeoutError:
            logger.error(
                "[GenCoord] swiss SCORE_TIMEOUT key=%s m=%d/%d after %.0fs",
                key, len(g.members), g.expected_n, self.score_timeout,
            )
            results = self._fallback_results(g, DesignReason.INFRA_VLM_TIMEOUT)
        except Exception:
            logger.exception(
                "[GenCoord] swiss FAILED key=%s m=%d/%d",
                key, len(g.members), g.expected_n,
            )
            results = self._fallback_results(g, DesignReason.INFRA_VLM_ERROR)

        async with self._lock:
            g.results = results
            g.state = GroupState.FINALIZED
            self._groups.pop(key, None)
            self._recently_closed[key] = time.monotonic() + self.closed_ttl

        g.fired_event.set()

        # One-line summary line for log scraping.
        scores_str = ",".join(f"{r.score:.3f}" for r in results)
        reason_summary: dict[str, int] = {}
        for r in results:
            reason_summary[r.reason_code] = reason_summary.get(r.reason_code, 0) + 1
        logger.info(
            "[GenCoord] FINAL key=%s fire=%s m=%d/%d scores=[%s] reasons=%s",
            key, g.fire_reason, len(g.members), g.expected_n,
            scores_str, reason_summary,
        )

    # ------------------------------------------------------------------
    async def _score_group(self, g: _Group) -> list[DesignResult]:
        """m<N-aware Swiss invocation.  Returns one result per member."""
        n_members = len(g.members)
        confidence = n_members / float(g.expected_n) if g.expected_n > 0 else 0.0

        # Tag per-member reason for "model_failed" up-front.
        baseline_reasons: list[Optional[str]] = []
        for m in g.members:
            if m["model_failed"]:
                baseline_reasons.append(DesignReason.MODEL_EMPTY_HTML)
            elif not m["render_path"]:
                baseline_reasons.append(DesignReason.INFRA_RENDER_FAILED)
            else:
                baseline_reasons.append(None)

        # Members that CAN take part in pairwise comparison.
        scorable_idx = [i for i, r in enumerate(baseline_reasons) if r is None]

        if n_members == 0 or not g.ref_path:
            return self._fallback_results(g, DesignReason.INFRA_NO_MEMBERS)

        # Fixed policy: if the group is incomplete (not all rollouts assembled),
        # skip the VLM API entirely and mask all members from design updates.
        if n_members < g.expected_n:
            results: list[DesignResult] = []
            for i, _m in enumerate(g.members):
                reason = baseline_reasons[i]
                if reason == DesignReason.MODEL_EMPTY_HTML:
                    results.append(DesignResult(
                        score=self.model_fail_value, confidence=confidence,
                        reason_code=reason, fire_reason=g.fire_reason,
                        applicable=0,
                    ))
                elif reason == DesignReason.INFRA_RENDER_FAILED:
                    results.append(DesignResult(
                        score=self.infra_fail_value, confidence=confidence,
                        reason_code=reason, fire_reason=g.fire_reason,
                        applicable=0,
                    ))
                else:
                    results.append(DesignResult(
                        score=self.infra_fail_value, confidence=confidence,
                        reason_code=DesignReason.INFRA_INCOMPLETE_GROUP,
                        fire_reason=g.fire_reason, applicable=0,
                    ))
            return results

        # m_scorable < min_members → shrinkage prior path.
        if len(scorable_idx) < self.min_members:
            results: list[DesignResult] = []
            for i, m in enumerate(g.members):
                reason = baseline_reasons[i]
                if reason == DesignReason.MODEL_EMPTY_HTML:
                    results.append(DesignResult(
                        score=self.model_fail_value, confidence=confidence,
                        reason_code=reason, fire_reason=g.fire_reason,
                        applicable=0,
                    ))
                elif reason == DesignReason.INFRA_RENDER_FAILED:
                    results.append(DesignResult(
                        score=self.infra_fail_value, confidence=confidence,
                        reason_code=reason, fire_reason=g.fire_reason,
                        applicable=0,
                    ))
                else:
                    # Sole survivor — give prior with low confidence.
                    results.append(DesignResult(
                        score=self.prior, confidence=confidence,
                        reason_code=DesignReason.LOW_M_PRIOR,
                        fire_reason=g.fire_reason, applicable=0,
                    ))
            return results

        # Real Swiss path — invoke the judge over the *scorable* subset.
        from rl_verl.vlm_reward import swiss_tournament_judge
        render_paths = [g.members[i]["render_path"] for i in scorable_idx]

        scored = await asyncio.wait_for(
            swiss_tournament_judge(g.ref_path, render_paths),
            timeout=self.score_timeout,
        )

        # Map scorable subset back into the member layout.
        results = [None] * n_members  # type: ignore[list-item]
        succ_scores = [s for s in scored if s is not None]
        mean_succ = sum(succ_scores) / len(succ_scores) if succ_scores else self.prior

        for j, i in enumerate(scorable_idx):
            s = scored[j] if j < len(scored) else None
            if s is None:
                # Judge couldn't score this member (e.g. file vanished mid-flight).
                results[i] = DesignResult(
                    score=self.infra_fail_value, confidence=confidence,
                    reason_code=DesignReason.INFRA_VLM_ERROR,
                    fire_reason=g.fire_reason, applicable=0,
                )
            else:
                # Shrink toward prior when confidence is low.
                shrunk = confidence * float(s) + (1.0 - confidence) * self.prior
                results[i] = DesignResult(
                    score=float(shrunk), confidence=confidence,
                    reason_code=DesignReason.OK,
                    fire_reason=g.fire_reason, applicable=1,
                )

        # Now the failed / non-scorable members.
        for i, reason in enumerate(baseline_reasons):
            if results[i] is not None:
                continue
            if reason == DesignReason.MODEL_EMPTY_HTML:
                results[i] = DesignResult(
                    score=self.model_fail_value, confidence=confidence,
                    reason_code=reason, fire_reason=g.fire_reason,
                    applicable=0,
                )
            elif reason == DesignReason.INFRA_RENDER_FAILED:
                results[i] = DesignResult(
                    score=self.infra_fail_value, confidence=confidence,
                    reason_code=reason, fire_reason=g.fire_reason,
                    applicable=0,
                )
            else:  # should not happen, defensive
                results[i] = DesignResult(
                    score=mean_succ, confidence=confidence,
                    reason_code=DesignReason.INFRA_VLM_ERROR,
                    fire_reason=g.fire_reason, applicable=0,
                )
        return results  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def _fallback_results(self, g: _Group, reason: str) -> list[DesignResult]:
        """Build a uniform fallback for catastrophic failures."""
        n_members = max(1, len(g.members))
        confidence = n_members / float(g.expected_n) if g.expected_n > 0 else 0.0
        out: list[DesignResult] = []
        for m in g.members:
            if m["model_failed"]:
                out.append(DesignResult(
                    score=self.model_fail_value, confidence=confidence,
                    reason_code=DesignReason.MODEL_EMPTY_HTML,
                    fire_reason=g.fire_reason, applicable=0,
                ))
            else:
                out.append(DesignResult(
                    score=self.infra_fail_value, confidence=confidence,
                    reason_code=reason, fire_reason=g.fire_reason,
                    applicable=0,
                ))
        return out


# ---------------------------------------------------------------------------
# Process-local singleton
# ---------------------------------------------------------------------------

_coord: Optional[GenerateSwissCoordinator] = None


def get_generate_coordinator() -> GenerateSwissCoordinator:
    """Lazy-init singleton.  Must be called from inside a running event loop."""
    global _coord
    if _coord is None:
        _coord = GenerateSwissCoordinator()
    return _coord


__all__ = [
    "DesignReason",
    "DesignResult",
    "GenerateSwissCoordinator",
    "get_generate_coordinator",
]
