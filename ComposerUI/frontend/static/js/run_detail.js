/**
 * Logic for Run Detail screen: live status polling, incremental
 * trial streaming, stage timing breakdown, failure clusters, and
 * auto-termination on terminal run status.
 */

let runId = null;
let detailPollTimer = null;
let trialsPollTimer = null;
let elapsedTimer = null;
let errorsPollTimer = null;
let nextAfter = 0;
let nextErrorAfter = 0;
let isTerminal = false;
let allTrials = new Map();
let activeTrialIds = new Set();
let trialSort = { key: "trial_id", direction: "asc" };
let lastChartTrials = [];
let currentRunDetail = null;
let inspectorData = null;
let inspectorMode = null;
let activeInspectorTab = null;
let errorSummary = null;
let errorObservabilityAvailable = null;
let errorEvents = new Map();
let selectedErrorStage = "all";

document.addEventListener("DOMContentLoaded", () => {
    const params = new URLSearchParams(window.location.search);
    runId = params.get("run_id");

    if (!runId) {
        showToast("No run_id specified in URL", "error");
        return;
    }

    const runIdDisplay = document.getElementById("detail-run-id");
    if (runIdDisplay) runIdDisplay.textContent = runId;

    initInspectorHandlers();
    initTrialSorting();
    initCompletionPopupHandlers();

    // Initial fetch
    loadRunDetail();
    pollTrials();
    pollErrors();

    // Polling timers
    detailPollTimer = setInterval(loadRunDetail, 3000);
    trialsPollTimer = setInterval(pollTrials, 2000);
    elapsedTimer = setInterval(updateElapsedCells, 1000);
    errorsPollTimer = setInterval(pollErrors, 2000);
});

let chartResizeScheduled = false;
window.addEventListener("resize", () => {
    if (chartResizeScheduled) return;
    chartResizeScheduled = true;
    requestAnimationFrame(() => {
        chartResizeScheduled = false;
        updateCostChart(lastChartTrials);
        updateStageTimingChart(lastChartTrials);
    });
});

async function loadRunDetail() {
    if (!runId) return;
    try {
        const detail = await apiGet(`/api/runs/${runId}`);
        currentRunDetail = detail;
        renderRunHeaderAndKpis(detail);
        const hasRecordedTiming = detail.best_result?.mean_pipeline_latency_ms != null
            || Object.values(detail.stage_timings || {}).some(value => value != null);
        renderStageTimings(detail.stage_timings, hasRecordedTiming);
        renderFailureClusters(detail.failure_clusters);

        if (detail.error_message) {
            renderErrorBanner(detail.error_message);
        }

        // Terminal status check
        const terminalStatuses = ["completed", "failed", "stopped_early"];
        if (terminalStatuses.includes(detail.status)) {
            // Fetch the final records before stopping both polling loops.
            await Promise.all([pollTrials(), pollErrors()]);
            stopPolling(detail);
        }
    } catch (err) {
        console.error("Failed to load run detail:", err);
    }
}

async function pollErrors() {
    if (!runId) return;
    try {
        const res = await apiGet(`/api/runs/${runId}/errors?after=${nextErrorAfter}`);
        errorObservabilityAvailable = Boolean(res.available);
        errorSummary = res.summary || null;
        (res.errors || []).forEach(event => {
            errorEvents.set(String(event.event_id), event);
        });
        nextErrorAfter = res.next_after ?? nextErrorAfter;
        renderErrorObservability();
    } catch (err) {
        console.error("Failed to poll error observability:", err);
    }
}

async function pollTrials() {
    if (!runId) return;
    try {
        const res = await apiGet(`/api/runs/${runId}/trials?after=${nextAfter}`);
        (res.trials || []).forEach(trial => {
            allTrials.set(Number(trial.trial_id), trial);
        });
        nextAfter = res.next_after ?? nextAfter;

        const nextActiveIds = new Set(
            (res.active_trials || []).map(trial => Number(trial.trial_id)),
        );
        activeTrialIds.forEach(trialId => {
            const existing = allTrials.get(trialId);
            if (!nextActiveIds.has(trialId) && existing?.status === "running") {
                allTrials.delete(trialId);
            }
        });
        (res.active_trials || []).forEach(trial => {
            allTrials.set(Number(trial.trial_id), trial);
        });
        activeTrialIds = nextActiveIds;

        renderTrialRows();
        const completedTrials = [...allTrials.values()].filter(trial => trial.status !== "running");
        lastChartTrials = completedTrials;
        updateCostChart(completedTrials);
        updateStageTimingChart(completedTrials);
    } catch (err) {
        console.error("Failed to poll trials:", err);
    }
}

function stopPolling(detail) {
    if (isTerminal) return;
    isTerminal = true;
    const status = detail.status;
    if (detailPollTimer) clearInterval(detailPollTimer);
    if (trialsPollTimer) clearInterval(trialsPollTimer);
    if (elapsedTimer) clearInterval(elapsedTimer);
    if (errorsPollTimer) clearInterval(errorsPollTimer);

    const liveIndicator = document.getElementById("live-indicator");
    if (liveIndicator) {
        liveIndicator.innerHTML = `
            <span class="w-2 h-2 rounded-full bg-outline"></span>
            <span class="text-outline">Run Finished (${status})</span>
        `;
    }

    showRunCompletionPopup(detail);
}

function formatErrorRate(value) {
    const rate = Number(value);
    if (!Number.isFinite(rate)) return "—";
    return `${(rate * 100).toFixed(rate > 0 && rate < 0.01 ? 2 : 1)}%`;
}

function errorHealthBadge(health) {
    if (health === "HEALTHY") {
        return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-secondary-container/20 text-secondary font-semibold"><span class="w-1.5 h-1.5 rounded-full bg-secondary"></span>HEALTHY</span>`;
    }
    if (health === "DEGRADED") {
        return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-tertiary-container/20 text-tertiary font-semibold"><span class="w-1.5 h-1.5 rounded-full bg-tertiary"></span>DEGRADED</span>`;
    }
    return `<span class="inline-flex items-center gap-1 px-2 py-0.5 rounded bg-error-container/20 text-error font-semibold"><span class="w-1.5 h-1.5 rounded-full bg-error"></span>ERRORS DETECTED</span>`;
}

function humanizeStage(stage) {
    return String(stage || "unknown").replaceAll("_", " ").replace(/\b\w/g, char => char.toUpperCase());
}

function renderErrorObservability() {
    const rateEl = document.getElementById("error-rate-value");
    const recoveredEl = document.getElementById("error-recovered-value");
    const unrecoveredEl = document.getElementById("error-unrecovered-value");
    const badgeEl = document.getElementById("error-health-badge");
    const stagesEl = document.getElementById("error-stage-rates");
    const refusalCountEl = document.getElementById("answer-refusal-count-value");
    const refusalRateEl = document.getElementById("answer-refusal-rate-value");
    const refusalSamplesEl = document.getElementById("answer-refusal-samples-value");
    if (!rateEl || !recoveredEl || !unrecoveredEl || !badgeEl || !stagesEl || !refusalCountEl || !refusalRateEl || !refusalSamplesEl) return;

    if (errorObservabilityAvailable === false) {
        rateEl.textContent = "Not recorded";
        rateEl.className = "font-code-sm text-[12px] font-semibold text-outline mt-2";
        recoveredEl.textContent = "—";
        unrecoveredEl.textContent = "—";
        refusalCountEl.textContent = "—";
        refusalRateEl.textContent = "Not recorded";
        refusalSamplesEl.textContent = "Legacy run";
        badgeEl.innerHTML = `<span class="text-outline">LEGACY RUN</span>`;
        stagesEl.innerHTML = `<div class="px-space-sm py-3 text-outline text-body-sm bg-surface-container-low rounded">Stage error rates were not recorded for this run.</div>`;
        renderErrorEvents();
        return;
    }

    const summary = errorSummary || {};
    const attempts = Number(summary.attempts || 0);
    rateEl.className = "font-metric-display text-[22px] font-bold text-on-surface mt-0.5";
    rateEl.textContent = attempts > 0 ? formatErrorRate(summary.error_rate) : "Waiting";
    recoveredEl.textContent = String(summary.recovered_errors || 0);
    unrecoveredEl.textContent = String(summary.unrecovered_errors || 0);
    const generatedSamples = Number(summary.generation_samples || 0);
    refusalCountEl.textContent = String(summary.answer_refusals || 0);
    refusalRateEl.textContent = generatedSamples > 0
        ? formatErrorRate(summary.answer_refusal_rate)
        : "Waiting";
    refusalSamplesEl.textContent = generatedSamples > 0
        ? `${generatedSamples} generated answer${generatedSamples === 1 ? "" : "s"}`
        : "No generated answers yet";
    badgeEl.innerHTML = attempts > 0
        ? errorHealthBadge(summary.health)
        : `<span class="text-outline">NO ATTEMPTS YET</span>`;

    const stages = [...(summary.stages || [])].sort((left, right) =>
        Number(right.unrecovered_errors || 0) - Number(left.unrecovered_errors || 0)
        || Number(right.error_rate || 0) - Number(left.error_rate || 0)
        || String(left.stage).localeCompare(String(right.stage))
    );
    stagesEl.innerHTML = stages.length ? stages.map(stage => {
        const rate = Math.max(0, Math.min(1, Number(stage.error_rate || 0)));
        const hasFatal = Number(stage.unrecovered_errors || 0) > 0;
        const barColor = hasFatal ? "bg-error" : Number(stage.errors || 0) > 0 ? "bg-tertiary" : "bg-secondary";
        return `<button type="button" data-error-stage="${escapeHtml(stage.stage)}" class="text-left bg-surface-container-low hover:bg-surface-container-high rounded px-space-sm py-1.5 transition-colors">
            <div class="flex items-center justify-between gap-space-sm">
                <span class="font-code-sm text-[10px] text-on-surface font-semibold">${escapeHtml(humanizeStage(stage.stage))}</span>
                <span class="font-code-sm text-[10px] ${hasFatal ? "text-error" : "text-outline"}">${formatErrorRate(rate)} · ${stage.errors}/${stage.attempts}</span>
            </div>
            <div class="h-1 bg-surface-container-highest rounded-full overflow-hidden mt-1"><div class="h-full ${barColor}" style="width:${Math.max(rate * 100, rate > 0 ? 2 : 0)}%"></div></div>
        </button>`;
    }).join("") : `<div class="px-space-sm py-3 text-outline text-body-sm bg-surface-container-low rounded">Waiting for observed operations…</div>`;

    stagesEl.querySelectorAll("[data-error-stage]").forEach(button => {
        button.addEventListener("click", () => {
            selectedErrorStage = button.dataset.errorStage;
            document.getElementById("error-events-details")?.setAttribute("open", "");
            renderErrorEvents();
        });
    });
    renderErrorEvents();
}

function renderErrorEvents() {
    const container = document.getElementById("error-events-container");
    const filters = document.getElementById("error-event-filter");
    if (!container || !filters) return;
    const events = [...errorEvents.values()].sort((left, right) =>
        String(right.occurred_at).localeCompare(String(left.occurred_at))
    );
    const stages = [...new Set(events.map(event => event.stage))].sort();
    if (selectedErrorStage !== "all" && !stages.includes(selectedErrorStage)) selectedErrorStage = "all";
    filters.innerHTML = ["all", ...stages].map(stage => `<button type="button" data-error-filter="${escapeHtml(stage)}" class="px-1.5 py-0.5 rounded text-[9px] font-code-sm ${selectedErrorStage === stage ? "bg-primary text-on-primary" : "bg-surface-container-high text-outline hover:text-on-surface"}">${stage === "all" ? "All" : escapeHtml(humanizeStage(stage))}</button>`).join("");
    filters.querySelectorAll("[data-error-filter]").forEach(button => button.addEventListener("click", () => {
        selectedErrorStage = button.dataset.errorFilter;
        renderErrorEvents();
    }));

    const visible = events.filter(event => selectedErrorStage === "all" || event.stage === selectedErrorStage).slice(0, 50);
    if (!visible.length) {
        container.innerHTML = `<div class="text-outline text-[11px]">${errorObservabilityAvailable === false ? "No observability events were recorded." : "No errors recorded for this selection."}</div>`;
        return;
    }
    container.innerHTML = visible.map(event => {
        const recovered = event.recovery && event.recovery !== "unrecovered";
        const trial = event.trial_id == null ? "Run level" : `Trial #${event.trial_id}${event.sample_index == null ? "" : ` · Sample ${event.sample_index}`}`;
        const completedTrial = event.trial_id != null && allTrials.get(Number(event.trial_id))?.status !== "running";
        return `<div class="bg-surface-container-low border ${recovered ? "border-tertiary/30" : "border-error/30"} rounded p-space-xs">
            <div class="flex items-start justify-between gap-space-xs">
                <div class="min-w-0"><span class="font-code-sm text-[10px] font-semibold ${recovered ? "text-tertiary" : "text-error"}">${escapeHtml(event.error_code || event.error_type || "UNKNOWN")}</span><span class="text-[10px] text-outline"> · ${escapeHtml(humanizeStage(event.stage))}</span></div>
                <span class="text-[9px] font-code-sm ${recovered ? "text-tertiary" : "text-error"}">${recovered ? escapeHtml(String(event.recovery).toUpperCase()) : "UNRECOVERED"}</span>
            </div>
            <div class="text-[10px] text-on-surface-variant break-words mt-0.5">${escapeHtml(event.message || "No diagnostic message")}</div>
            <div class="flex items-center justify-between gap-space-xs mt-1 text-[9px] text-outline">
                <span>${escapeHtml(trial)}${event.attempt == null ? "" : ` · Attempt ${event.attempt}`}${event.provider ? ` · ${escapeHtml(event.provider)}` : ""}</span>
                ${completedTrial ? `<button type="button" data-open-error-trial="${Number(event.trial_id)}" class="text-primary hover:underline">Open trial</button>` : ""}
            </div>
        </div>`;
    }).join("");
    container.querySelectorAll("[data-open-error-trial]").forEach(button => {
        button.addEventListener("click", () => openTrialInspector(Number(button.dataset.openErrorTrial), "samples"));
    });
}

function initCompletionPopupHandlers() {
    document.getElementById("completion-popup-dismiss")?.addEventListener("click", hideRunCompletionPopup);
    document.getElementById("completion-popup-view-result")?.addEventListener("click", () => {
        const trialId = currentRunDetail?.best_result?.trial_id;
        hideRunCompletionPopup();
        if (trialId !== null && trialId !== undefined) openTrialInspector(Number(trialId), "overview");
    });
}

function showRunCompletionPopup(detail) {
    const popup = document.getElementById("run-completion-popup");
    if (!popup) return;

    const states = {
        completed: {
            title: "Run completed successfully",
            message: "The architecture search has finished. You can now inspect the winning configuration and its evaluation results.",
            icon: "task_alt",
            border: "#7bdb80",
            iconColor: "#7bdb80",
            iconBackground: "rgba(0, 113, 36, 0.2)",
        },
        failed: {
            title: "Run failed",
            message: detail.error_message || "The run stopped because an execution error occurred. Review the error and completed trials for details.",
            icon: "error",
            border: "#ffb4ab",
            iconColor: "#ffb4ab",
            iconBackground: "rgba(147, 0, 10, 0.2)",
        },
        stopped_early: {
            title: "Run stopped early",
            message: "The run reached its stopping condition. Results from all completed trials are available for inspection.",
            icon: "stop_circle",
            border: "#fabc45",
            iconColor: "#fabc45",
            iconBackground: "rgba(210, 153, 34, 0.2)",
        },
    };
    const state = states[detail.status] || states.stopped_early;
    const completedTrials = Number(detail.completed_trials || 0);
    const totalTrials = Number(detail.total_trials || 0);
    const bestTrialId = detail.best_result?.trial_id;

    popup.style.borderColor = state.border;
    document.getElementById("completion-popup-icon").textContent = state.icon;
    const iconWrap = document.getElementById("completion-popup-icon-wrap");
    iconWrap.style.color = state.iconColor;
    iconWrap.style.backgroundColor = state.iconBackground;
    document.getElementById("completion-popup-title").textContent = state.title;
    document.getElementById("completion-popup-message").textContent = state.message;
    document.getElementById("completion-popup-progress").textContent = totalTrials > 0
        ? `${completedTrials} / ${totalTrials}`
        : String(completedTrials);
    document.getElementById("completion-popup-score").textContent = detail.best_score == null
        ? "Not available"
        : formatScore(detail.best_score);

    const viewResultButton = document.getElementById("completion-popup-view-result");
    if (viewResultButton) viewResultButton.hidden = bestTrialId === null || bestTrialId === undefined;

    popup.hidden = false;
    document.title = `${detail.status === "completed" ? "✓" : "!"} ${state.title} // Muffakir Composer`;
}

function hideRunCompletionPopup() {
    const popup = document.getElementById("run-completion-popup");
    if (popup) popup.hidden = true;
}

function renderRunHeaderAndKpis(d) {
    const nameEl = document.getElementById("detail-run-name");
    if (nameEl) nameEl.textContent = d.run_name || runId;

    const statusBadge = document.getElementById("detail-status-badge");
    if (statusBadge) {
        statusBadge.innerHTML = getStatusBadgeHtml(d.status);
    }

    const bestScoreEl = document.getElementById("kpi-best-score");
    if (bestScoreEl) {
        bestScoreEl.textContent = d.best_score !== null && d.best_score !== undefined
            ? formatScore(d.best_score)
            : "—";
    }

    const progressEl = document.getElementById("kpi-progress");
    if (progressEl) {
        progressEl.textContent = `${d.completed_trials} / ${d.total_trials || "—"}`;
    }

    const progressBar = document.getElementById("kpi-progress-bar");
    if (progressBar && d.total_trials > 0) {
        const pct = Math.min(100, Math.round((d.completed_trials / d.total_trials) * 100));
        progressBar.style.width = `${pct}%`;
    }

    const costEl = document.getElementById("kpi-cost");
    if (costEl) {
        costEl.textContent = d.total_cost_usd !== null && d.total_cost_usd !== undefined
            ? `$${d.total_cost_usd.toFixed(4)}`
            : "Not recorded";
    }

    const wsEl = document.getElementById("kpi-web-search-fallbacks");
    if (wsEl) {
        wsEl.textContent = d.total_web_search_fallbacks !== null && d.total_web_search_fallbacks !== undefined
            ? String(d.total_web_search_fallbacks)
            : "0";
    }

    renderBestResultCard(d.best_result);
}

function renderBestResultCard(best) {
    const el = document.getElementById("kpi-best-result");
    if (!el) return;
    if (!best) {
        el.innerHTML = `<div class="text-outline font-body-sm">No successful trials completed yet.</div>`;
        return;
    }
    const pipelineLatency = best.mean_pipeline_latency_ms == null
        ? "Not recorded"
        : `${Number(best.mean_pipeline_latency_ms).toFixed(2)} ms/query`;
    el.innerHTML = `
        <div class="grid grid-cols-2 gap-space-xs mb-space-sm">
            <div><div class="text-[10px] text-outline font-code-sm">TRIAL</div><div class="text-primary font-semibold">#${best.trial_id}</div></div>
            <div><div class="text-[10px] text-outline font-code-sm">MEAN PIPELINE</div><div class="text-primary font-semibold">${pipelineLatency}</div></div>
        </div>
        ${renderMetricGrid(best.metrics || {})}
        <div class="mt-space-sm">${renderBestConfigChips(best.config || {})}</div>
    `;
}

function renderStageTimings(timings, hasRecordedTiming = false) {
    if (!timings) return;
    const stages = [
        { key: "query_transform", id: "time-query-transform" },
        { key: "query_embedding", id: "time-query-embedding" },
        { key: "vector_search", id: "time-vector-search" },
        { key: "rerank", id: "time-rerank" },
        { key: "relevance_check", id: "time-relevance-check" },
        { key: "web_search", id: "time-web-search" },
        { key: "generation", id: "time-generation" },
        { key: "hallucination_check", id: "time-hallucination-check" },
    ];

    stages.forEach(s => {
        const el = document.getElementById(s.id);
        if (el) {
            const val = timings[s.key];
            el.textContent = val !== null && val !== undefined
                ? `${Number(val).toFixed(2)} ms`
                : (hasRecordedTiming ? "Not applicable" : "Not recorded");
        }
    });
}

function renderFailureClusters(clusters) {
    const container = document.getElementById("failure-clusters-container");
    if (!container) return;

    const entries = Object.entries(clusters || {});
    if (entries.length === 0) {
        container.innerHTML = `
            <div class="px-space-base py-3 text-outline text-body-sm bg-surface-container-low rounded">
                No trial failures recorded so far.
            </div>
        `;
        return;
    }

    container.innerHTML = entries.map(([code, trialIds]) => `
        <div class="bg-surface-container-low border border-error/30 rounded p-space-sm flex flex-col gap-1">
            <div class="flex items-center justify-between">
                <span class="font-code-sm font-semibold text-error flex items-center gap-1">
                    <span class="material-symbols-outlined text-[15px]">warning</span>
                    <span>${escapeHtml(code)}</span>
                </span>
                <span class="px-2 py-0.5 rounded bg-error/20 text-error font-code-sm text-badge-label font-bold">
                    ${trialIds.length} trials
                </span>
            </div>
            <div class="text-body-xs text-outline flex items-center gap-1 flex-wrap">
                <span>Affected Trial IDs:</span>
                ${trialIds.slice(0, 10).map(id => `<button type="button" data-open-trial="${Number(id)}" class="px-1 bg-surface-container hover:bg-surface-container-high rounded font-code-sm text-primary transition-colors">#${id}</button>`).join("")}
                ${trialIds.length > 10 ? `<span class="text-outline">+${trialIds.length - 10} more</span>` : ""}
            </div>
        </div>
    `).join("");

    container.querySelectorAll("[data-open-trial]").forEach(button => {
        button.addEventListener("click", () => openTrialInspector(Number(button.dataset.openTrial)));
    });
}

function renderErrorBanner(msg) {
    let banner = document.getElementById("run-error-banner");
    if (!banner) {
        banner = document.createElement("div");
        banner.id = "run-error-banner";
        banner.className = "mb-space-base bg-error-container/20 border border-error rounded p-space-base flex items-start gap-space-sm";
        const contentContainer = document.getElementById("main-detail-content");
        if (contentContainer) {
            contentContainer.prepend(banner);
        }
    }
    banner.innerHTML = `
        <span class="material-symbols-outlined text-error text-[20px] shrink-0">report</span>
        <div class="flex flex-col gap-1 min-w-0">
            <span class="font-semibold text-error font-headline-sm">Execution Error Occurred</span>
            <pre class="font-code-sm text-code-sm text-on-surface whitespace-pre-wrap overflow-x-auto">${escapeHtml(msg)}</pre>
        </div>
    `;
}

function initTrialSorting() {
    document.querySelectorAll("[data-trial-sort]").forEach(button => {
        button.addEventListener("click", () => {
            const key = button.dataset.trialSort;
            if (trialSort.key === key) {
                trialSort.direction = trialSort.direction === "asc" ? "desc" : "asc";
            } else {
                trialSort = { key, direction: defaultTrialSortDirection(key) };
            }
            updateTrialSortHeaders();
            renderTrialRows();
        });
    });
    updateTrialSortHeaders();
}

function defaultTrialSortDirection(key) {
    return ["composite_score", "cost_usd", "web_search_fallback_count"].includes(key)
        ? "desc"
        : "asc";
}

function renderTrialRows() {
    const tbody = document.getElementById("trials-table-body");
    if (!tbody) return;

    const noDataRow = document.getElementById("no-trials-row");
    if (noDataRow) noDataRow.remove();

    const rows = [...allTrials.values()];
    const sortedTrials = [
        ...rows.filter(trial => trial.status === "running").sort(compareTrials),
        ...rows.filter(trial => trial.status !== "running").sort(compareTrials),
    ];
    tbody.innerHTML = "";

    if (sortedTrials.length === 0) {
        tbody.innerHTML = `
            <tr id="no-trials-row">
                <td colspan="10" class="py-8 text-center text-outline font-body-sm">Waiting for trials to execute...</td>
            </tr>
        `;
        return;
    }

    sortedTrials.forEach(t => {
        const tr = document.createElement("tr");
        const isRunning = t.status === "running";
        const isSuccess = t.status === "success";
        tr.className = isRunning
            ? "bg-primary/5 border-l-2 border-primary font-code-sm text-code-sm"
            : "hover:bg-surface-container-low transition-colors font-code-sm text-code-sm cursor-pointer group focus:outline-none focus:bg-surface-container-low focus:ring-1 focus:ring-inset focus:ring-primary/60";
        if (!isRunning) {
            tr.tabIndex = 0;
            tr.setAttribute("role", "button");
            tr.setAttribute("aria-label", `Inspect trial ${t.trial_id}`);
        }

        let statusBadge;
        if (isRunning) {
            statusBadge = `<span class="inline-flex items-center gap-1.5 text-primary font-semibold"><span class="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></span>RUNNING</span>`;
        } else if (isSuccess) {
            statusBadge = `<span class="text-secondary font-semibold">✓ SUCCESS</span>`;
        } else {
            statusBadge = `<span class="text-error font-semibold" title="${escapeHtml(t.error_code || t.error_type || "")}">✗ FAILED</span>`;
        }

        const scoreDisplay = isSuccess
            ? `<span class="text-on-surface font-semibold">${formatScore(t.composite_score)}</span>`
            : `<span class="text-outline">—</span>`;

        const costDisplay = t.cost_usd !== null && t.cost_usd !== undefined
            ? `$${t.cost_usd.toFixed(4)}`
            : "—";
        const webSearchDisplay = t.web_search_fallback_count
            ? `<span class="text-tertiary">${t.web_search_fallback_count}</span>`
            : "—";
        const completedSamples = Number(t.completed_samples || 0);
        const totalSamples = Number(t.total_samples || 0);
        const progressDisplay = totalSamples > 0
            ? `${completedSamples} / ${totalSamples}`
            : "—";
        const elapsedDisplay = isRunning
            ? `<span data-trial-elapsed data-started-at="${escapeHtml(t.started_at || "")}">${formatRunningElapsed(t.started_at)}</span>`
            : formatElapsedMs(t.latency_ms);

        tr.innerHTML = `
            <td class="py-1.5 px-3 text-outline">#${t.trial_id}</td>
            <td class="py-1.5 px-3">${statusBadge}</td>
            <td class="py-1.5 px-3 text-right ${isRunning ? "text-primary font-semibold" : "text-on-surface-variant"}">${progressDisplay}</td>
            <td class="py-1.5 px-3 text-right text-on-surface-variant">${elapsedDisplay}</td>
            <td class="py-1.5 px-3 text-right">${scoreDisplay}</td>
            <td class="py-1.5 px-3 text-right text-on-surface-variant">${t.mean_pipeline_latency_ms == null ? "—" : Number(t.mean_pipeline_latency_ms).toFixed(2) + "ms"}</td>
            <td class="py-1.5 px-3 text-right text-outline">${costDisplay}</td>
            <td class="py-1.5 px-3 text-right text-outline">${webSearchDisplay}</td>
            <td class="py-1.5 px-3">
                <div class="flex items-center gap-1 flex-wrap text-badge-label">
                    ${renderBestConfigChips(t.resolved_rag_config)}
                </div>
            </td>
            <td class="py-1.5 px-3 text-right">
                ${isRunning
                    ? `<span class="inline-flex items-center gap-1 text-outline"><span class="text-[10px]">In progress</span></span>`
                    : `<span class="inline-flex items-center gap-1 text-primary opacity-70 group-hover:opacity-100"><span class="text-[10px]">Open</span><span class="material-symbols-outlined text-[16px]">chevron_right</span></span>`}
            </td>
        `;
        if (!isRunning) {
            tr.addEventListener("click", () => openTrialInspector(t.trial_id));
            tr.addEventListener("keydown", event => {
                if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    openTrialInspector(t.trial_id);
                }
            });
        }
        tbody.appendChild(tr);
    });
}

function formatRunningElapsed(startedAt) {
    const startedMs = Date.parse(startedAt || "");
    if (!Number.isFinite(startedMs)) return "—";
    return formatElapsedMs(Math.max(0, Date.now() - startedMs));
}

function formatElapsedMs(value) {
    const elapsedMs = Number(value);
    if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return "—";
    const totalSeconds = Math.floor(elapsedMs / 1000);
    const hours = Math.floor(totalSeconds / 3600);
    const minutes = Math.floor((totalSeconds % 3600) / 60);
    const seconds = totalSeconds % 60;
    if (hours > 0) return `${hours}h ${String(minutes).padStart(2, "0")}m`;
    if (minutes > 0) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
    return `${seconds}s`;
}

function updateElapsedCells() {
    document.querySelectorAll("[data-trial-elapsed]").forEach(element => {
        element.textContent = formatRunningElapsed(element.dataset.startedAt);
    });
}

function compareTrials(left, right) {
    const leftValue = getTrialSortValue(left, trialSort.key);
    const rightValue = getTrialSortValue(right, trialSort.key);
    const leftMissing = leftValue === null || leftValue === undefined || leftValue === "";
    const rightMissing = rightValue === null || rightValue === undefined || rightValue === "";
    if (leftMissing || rightMissing) {
        if (leftMissing && rightMissing) return Number(left.trial_id || 0) - Number(right.trial_id || 0);
        // Keep unavailable telemetry at the bottom in either sort direction.
        return leftMissing ? 1 : -1;
    }

    const comparison = compareTrialValues(leftValue, rightValue);
    if (comparison !== 0) return trialSort.direction === "asc" ? comparison : -comparison;

    // Keep equal rows predictable while the stream receives more trials.
    return Number(left.trial_id || 0) - Number(right.trial_id || 0);
}

function getTrialSortValue(trial, key) {
    if (key === "resolved_rag_config") return stableConfigString(trial.resolved_rag_config);
    return trial[key];
}

function compareTrialValues(left, right) {
    if (typeof left === "number" && typeof right === "number") return left - right;
    return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
}

function stableConfigString(value) {
    if (value === null || value === undefined) return "";
    if (Array.isArray(value)) return value.map(stableConfigString).join("|");
    if (typeof value !== "object") return String(value);
    return Object.keys(value)
        .sort((left, right) => left.localeCompare(right))
        .map(key => `${key}:${stableConfigString(value[key])}`)
        .join("|");
}

function updateTrialSortHeaders() {
    document.querySelectorAll("[data-trial-sort-header]").forEach(header => {
        const key = header.dataset.trialSortHeader;
        const button = header.querySelector("[data-trial-sort]");
        const icon = header.querySelector("[data-sort-icon]");
        const isActive = key === trialSort.key;
        const directionLabel = trialSort.direction === "asc" ? "ascending" : "descending";

        header.setAttribute("aria-sort", isActive ? directionLabel : "none");
        if (icon) icon.textContent = isActive
            ? (trialSort.direction === "asc" ? "arrow_upward" : "arrow_downward")
            : "unfold_more";
        if (button) {
            button.setAttribute(
                "aria-label",
                `${isActive ? `Sorted by ${button.dataset.sortLabel}, ${directionLabel}` : `Sort by ${button.dataset.sortLabel}`}. Click to sort ${isActive && trialSort.direction === "asc" ? "descending" : "ascending"}.`,
            );
        }
    });
}

function getStatusBadgeHtml(status) {
    if (status === "running") {
        return `
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-container-high text-primary font-semibold text-badge-label">
                <span class="w-2 h-2 rounded-full bg-primary animate-ping"></span>
                <span>RUNNING</span>
            </span>
        `;
    }
    if (status === "completed") {
        return `
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-secondary-container/20 text-secondary font-semibold text-badge-label">
                <span class="w-2 h-2 rounded-full bg-secondary"></span>
                <span>COMPLETED</span>
            </span>
        `;
    }
    if (status === "failed") {
        return `
            <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-error-container/20 text-error font-semibold text-badge-label">
                <span class="w-2 h-2 rounded-full bg-error"></span>
                <span>FAILED</span>
            </span>
        `;
    }
    return `
        <span class="inline-flex items-center gap-1.5 px-2.5 py-1 rounded bg-surface-container text-tertiary font-semibold text-badge-label">
            <span>${(status || "UNKNOWN").toUpperCase()}</span>
        </span>
    `;
}

function renderBestConfigChips(cfg) {
    if (!cfg || typeof cfg !== "object") return "";
    const chips = [];
    if (cfg.llm_model) chips.push(`LLM:${cfg.llm_provider || ""}/${cfg.llm_model}`);
    if (cfg.llm_parameters) Object.entries(cfg.llm_parameters).forEach(([key, value]) => chips.push(`${key}:${value === null ? "omit" : JSON.stringify(value)}`));
    if (cfg.retrieval_method) chips.push(`retrieval:${cfg.retrieval_method}`);
    if (cfg.reranking_method && cfg.reranking) chips.push(`rerank:${cfg.reranking_method}`);
    if (cfg.reranking_model && ["cross_encoder", "pointwise"].includes(String(cfg.reranking_method))) chips.push(`reranker-model:${cfg.reranking_model}`);
    if (cfg.query_transformer_strategy && cfg.query_transformer) chips.push(`exp:${cfg.query_transformer_strategy}`);
    if (cfg.k) chips.push(`k:${cfg.k}`);

    return chips.map(c => `
        <span class="px-1.5 py-0.2 bg-surface-container rounded text-on-surface text-[10px]">${escapeHtml(c)}</span>
    `).join("");
}

function initInspectorHandlers() {
    document.getElementById("btn-run-metadata")?.addEventListener("click", () => openRunInspector("overview"));
    document.getElementById("btn-run-config")?.addEventListener("click", () => openRunInspector("configuration"));
    document.getElementById("btn-best-config")?.addEventListener("click", () => {
        const trialId = currentRunDetail?.best_result?.trial_id;
        if (trialId !== null && trialId !== undefined) openTrialInspector(trialId, "samples");
    });
    document.getElementById("inspector-close")?.addEventListener("click", closeInspector);
    document.getElementById("inspector-backdrop")?.addEventListener("click", closeInspector);
    document.getElementById("inspector-copy")?.addEventListener("click", copyInspectorJson);
    document.getElementById("inspector-content")?.addEventListener("click", event => {
        const button = event.target.closest("[data-best-samples]");
        if (button) openTrialInspector(Number(button.dataset.bestSamples), "samples");
    });
    document.addEventListener("keydown", event => {
        if (event.key === "Escape" && !document.getElementById("inspector-layer")?.hidden) {
            closeInspector();
        }
    });
}

async function openRunInspector(initialTab = "overview") {
    if (!runId) return;
    openInspectorShell("Run Inspector", currentRunDetail?.run_name || runId, "Loading run configuration and metadata…");
    try {
        const metadata = await apiGet(`/api/runs/${runId}/metadata`);
        inspectorMode = "run";
        inspectorData = {
            overview: currentRunDetail || {},
            configuration: metadata.config || {},
            manifest: metadata.manifest || {},
            pricing: metadata.pricing || {},
            best_config: currentRunDetail?.best_config || {},
            best_result: currentRunDetail?.best_result || null,
        };
        activeInspectorTab = initialTab;
        renderInspector();
    } catch (err) {
        renderInspectorError(err);
    }
}

async function openTrialInspector(trialId, initialTab = "overview") {
    if (!runId || !Number.isFinite(Number(trialId))) return;
    openInspectorShell("Trial Inspector", `Trial #${trialId}`, "Loading configuration, samples, answers, and chunks…");
    try {
        const data = await apiGet(`/api/runs/${runId}/trials/${trialId}`);
        inspectorMode = "trial";
        inspectorData = data;
        activeInspectorTab = initialTab;
        renderInspector();
    } catch (err) {
        renderInspectorError(err);
    }
}

function openInspectorShell(eyebrow, title, subtitle) {
    const layer = document.getElementById("inspector-layer");
    if (layer) layer.hidden = false;
    document.body.classList.add("overflow-hidden");
    document.getElementById("inspector-eyebrow").textContent = eyebrow;
    document.getElementById("inspector-title").textContent = title;
    document.getElementById("inspector-subtitle").textContent = subtitle;
    document.getElementById("inspector-tabs").innerHTML = "";
    document.getElementById("inspector-content").innerHTML = `
        <div class="h-full flex items-center justify-center text-outline font-code-sm text-[12px]">
            <span class="material-symbols-outlined text-[18px] animate-spin mr-space-xs">progress_activity</span>
            Loading engineering data…
        </div>
    `;
}

function closeInspector() {
    const layer = document.getElementById("inspector-layer");
    if (layer) layer.hidden = true;
    document.body.classList.remove("overflow-hidden");
}

function renderInspector() {
    if (!inspectorData || !inspectorMode) return;
    const trial = inspectorData.trial || {};
    const tabs = inspectorMode === "run"
        ? [
            { id: "overview", label: "Overview", icon: "dashboard" },
            { id: "configuration", label: "Run Config", icon: "tune" },
            { id: "prompts", label: "Prompts", icon: "edit_note" },
            { id: "pricing", label: "Pricing", icon: "payments" },
            { id: "best_result", label: "Best Result", icon: "military_tech" },
            { id: "manifest", label: "Manifest", icon: "data_object" },
            { id: "raw", label: "Raw JSON", icon: "code" },
        ]
        : [
            { id: "overview", label: "Overview", icon: "monitoring" },
            { id: "samples", label: `Samples (${inspectorData.sample_count || 0})`, icon: "quiz" },
            { id: "configuration", label: "Configuration", icon: "tune" },
            { id: "export", label: "Export Python", icon: "download" },
            { id: "raw", label: "Raw JSON", icon: "code" },
        ];

    if (!tabs.some(tab => tab.id === activeInspectorTab)) activeInspectorTab = tabs[0].id;
    const tabsEl = document.getElementById("inspector-tabs");
    tabsEl.innerHTML = tabs.map(tab => `
        <button type="button" data-inspector-tab="${tab.id}" class="flex items-center gap-1.5 px-3 py-2.5 border-b-2 font-code-sm text-[11px] whitespace-nowrap transition-colors ${tab.id === activeInspectorTab ? "border-primary text-primary" : "border-transparent text-outline hover:text-on-surface"}">
            <span class="material-symbols-outlined text-[15px]">${tab.icon}</span>
            <span>${escapeHtml(tab.label)}</span>
        </button>
    `).join("");
    tabsEl.querySelectorAll("[data-inspector-tab]").forEach(button => {
        button.addEventListener("click", () => {
            activeInspectorTab = button.dataset.inspectorTab;
            renderInspector();
        });
    });

    if (inspectorMode === "run") {
        document.getElementById("inspector-eyebrow").textContent = "Run Inspector";
        document.getElementById("inspector-title").textContent = inspectorData.configuration.run_name || currentRunDetail?.run_name || runId;
        document.getElementById("inspector-subtitle").textContent = `Run ${runId}`;
    } else {
        document.getElementById("inspector-eyebrow").textContent = "Trial Inspector";
        document.getElementById("inspector-title").textContent = `Trial #${trial.trial_id}`;
        document.getElementById("inspector-subtitle").textContent = `${trial.status || "unknown"} · ${inspectorData.sample_count || 0} evaluated samples`;
    }

    const content = document.getElementById("inspector-content");
    if (inspectorMode === "run") {
        content.innerHTML = renderRunInspectorContent(activeInspectorTab);
    } else {
        content.innerHTML = renderTrialInspectorContent(activeInspectorTab);
        if (activeInspectorTab === "export") loadTrialPythonExport();
    }
}

async function loadTrialPythonExport() {
    const selected = inspectorData;
    const selectedRun = runId;
    const trialId = selected.trial.trial_id;
    const container = document.getElementById("trial-python-export");
    if (!container) return;
    const current = () => inspectorData === selected && runId === selectedRun &&
        activeInspectorTab === "export" && document.getElementById("trial-python-export") === container;
    const endpoint = `/api/runs/${encodeURIComponent(selectedRun)}/trials/${Number(trialId)}/export`;
    try {
        const data = await apiGet(`${endpoint}?format=preview`);
        if (!current()) return;
        container.innerHTML = `
            <h3 class="text-on-surface font-semibold mb-3">Export Trial #${Number(trialId)}</h3>
            <p class="text-outline text-sm mb-3">${escapeHtml(data.metadata.pipeline_mode)} · ${escapeHtml(data.metadata.retrieval_source)} · Original status: ${escapeHtml(data.metadata.status)}</p>
            ${data.notices.map(notice => `<p class="text-tertiary text-sm mb-2">${escapeHtml(notice)}</p>`).join("")}
            <div class="flex flex-wrap gap-3 my-4">
                <button type="button" id="copy-trial-python" class="px-3 py-2 rounded bg-primary text-on-primary">Copy code</button>
                <a class="px-3 py-2 rounded border border-outline-variant text-primary" href="${endpoint}?format=python">Download Python</a>
                <a class="px-3 py-2 rounded border border-outline-variant text-primary" href="${endpoint}?format=zip">Download project ZIP</a>
            </div>
            <p class="text-sm text-outline mb-3">Set the environment variables below. For corpus pipelines, run <code>python rag_app.py --index</code> once, then <code>python rag_app.py --question "Your question"</code>.</p>
            <h4 class="font-semibold mt-4">Requirements</h4><pre class="overflow-auto text-xs p-3">${escapeHtml(data.requirements)}</pre>
            <h4 class="font-semibold mt-4">Environment</h4>
            <ul class="text-sm list-disc pl-5 my-3">${data.environment.map(item => `<li><code>${escapeHtml(item.name)}</code> (${item.required ? "required" : "optional"}): ${escapeHtml(item.description)}</li>`).join("")}</ul>
            <details class="my-4"><summary class="cursor-pointer text-primary">Setup instructions</summary><pre class="whitespace-pre-wrap text-xs p-3">${escapeHtml(data.readme)}</pre></details>
            <h4 class="font-semibold mt-4">Python preview</h4><pre class="overflow-auto text-xs p-3 bg-surface-container-low rounded"><code>${escapeHtml(data.code)}</code></pre>`;
        document.getElementById("copy-trial-python").addEventListener("click", async event => {
            const button = event.currentTarget;
            try {
                await navigator.clipboard.writeText(data.code);
                if (current()) button.textContent = "Copied";
            } catch (_) {
                if (current()) button.textContent = "Copy unavailable — download Python";
            }
        });
    } catch (err) {
        if (current()) container.innerHTML = `<p role="alert" class="text-tertiary">Export unavailable: ${escapeHtml(err.message || String(err))}</p><button disabled class="mt-3 opacity-50">Download Python</button>`;
    }
}

function renderRunInspectorContent(tab) {
    if (tab === "configuration") return renderJsonPanel("Persisted Run Configuration", inspectorData.configuration, "API keys are intentionally excluded.");
    if (tab === "prompts") return renderRunPrompts(inspectorData.configuration || {});
    if (tab === "pricing") return renderRunPricing(inspectorData.pricing || {});
    if (tab === "best_result") return renderBestResultInspector(inspectorData.best_result);
    if (tab === "manifest") return renderJsonPanel("Trace Manifest", inspectorData.manifest, "Run identity, trace schema, search space, and lifecycle state.");
    if (tab === "raw") return renderJsonPanel("Complete Run Inspector Payload", inspectorData);

    const overview = inspectorData.overview || {};
    const manifest = inspectorData.manifest || {};
    const config = inspectorData.configuration || {};
    return `
        <div class="grid grid-cols-2 lg:grid-cols-3 gap-space-sm mb-space-lg">
            ${renderStatCard("Status", overview.status || manifest.status || "—", "radio_button_checked", "text-primary")}
            ${renderStatCard("Progress", `${overview.completed_trials ?? 0} / ${overview.total_trials || "—"}`, "checklist", "text-secondary")}
            ${renderStatCard("Best Score", overview.best_score == null ? "—" : formatScore(overview.best_score), "verified", "text-secondary")}
            ${renderStatCard("Total Cost", overview.total_cost_usd == null ? "Not recorded" : `$${Number(overview.total_cost_usd).toFixed(4)}`, "payments", "text-tertiary")}
            ${renderStatCard("Web Fallbacks", overview.total_web_search_fallbacks ?? 0, "travel_explore", "text-tertiary")}
            ${renderStatCard("Config Hash", manifest.config_hash || "—", "fingerprint", "text-outline")}
        </div>
        ${renderSection("Identity & Inputs", renderKeyValueGrid({
            run_id: runId,
            created_at: overview.created_at || config.created_at,
            documents_path: config.documents_path,
            evaluation_dataset: config.eval_dataset_path,
            pipeline_mode: config.pipeline_mode,
            retrieval_source: config.retrieval_source,
        }))}
        ${renderSection("Models & Runtime", renderKeyValueGrid({
            llm_provider: config.llm_provider,
            llm_model: config.llm_model,
            embedding_model: config.embedding_model,
            vector_db_provider: config.vector_db_provider,
            device: config.device,
            workers: config.n_jobs,
            max_eval_samples: config.max_eval_samples,
        }))}
        ${renderSection("Best Configuration Stage Timings", renderTimingGrid(overview.stage_timings || {}))}
    `;
}

function renderRunPrompts(config) {
    const prompts = config.resolved_prompts || {};
    const overrides = config.prompt_overrides || {};
    const hashes = config.prompt_hashes || {};
    if (!Object.keys(prompts).length) {
        return `<div class="py-16 text-center text-outline">
            <span class="material-symbols-outlined text-[28px] mb-space-xs">history</span>
            <p class="font-body-sm">Prompt configuration was not recorded for this run.</p>
        </div>`;
    }
    return `
        <div class="grid grid-cols-2 gap-space-sm mb-space-lg">
            ${renderStatCard("Language", config.language === "en" ? "English" : "Arabic", "translate", "text-primary")}
            ${renderStatCard("Customized", Object.keys(overrides).length, "edit", "text-secondary")}
        </div>
        <div class="flex flex-col gap-space-sm">
            ${Object.entries(prompts).map(([key, template], index) => `
                <details class="bg-surface-container border border-outline-variant/30 rounded" ${index === 0 ? "open" : ""}>
                    <summary class="cursor-pointer px-space-base py-space-sm flex items-center justify-between gap-space-sm hover:bg-surface-container-high">
                        <span class="font-code-sm text-[11px] text-on-surface">${escapeHtml(key)}</span>
                        <span class="font-code-sm text-[10px] ${Object.prototype.hasOwnProperty.call(overrides, key) ? "text-primary" : "text-outline"}">${Object.prototype.hasOwnProperty.call(overrides, key) ? "Custom" : "Default"}</span>
                    </summary>
                    <div class="px-space-base pb-space-base">
                        <div class="font-code-sm text-[9px] text-outline mb-space-xs break-all">SHA-256 ${escapeHtml(hashes[key] || "Not recorded")}</div>
                        <pre dir="auto" class="p-space-sm rounded bg-surface-container-low overflow-auto whitespace-pre-wrap break-words font-code-sm text-[11px] text-on-surface-variant">${escapeHtml(template)}</pre>
                    </div>
                </details>
            `).join("")}
        </div>`;
}

function renderRunPricing(pricing) {
    const models = pricing.models || [];
    const modeLabel = pricing.mode === "custom" ? "Per-run overrides" : "Default catalog";
    const fetchLabel = pricing.fetch_failed === true
        ? "Catalog fetch failed"
        : (pricing.fetched_at ? `Fetched ${pricing.fetched_at}` : "Not recorded");
    const rate = value => value === null || value === undefined
        ? "Not recorded"
        : `$${Number(value).toLocaleString(undefined, { maximumFractionDigits: 6 })}`;
    const sourceBadge = source => source === "custom"
        ? '<span class="px-2 py-0.5 rounded bg-primary/10 text-primary border border-primary/30">Custom</span>'
        : '<span class="px-2 py-0.5 rounded bg-surface-container-high text-outline border border-outline-variant/30">Default</span>';

    const rows = models.map(item => `
        <tr class="hover:bg-surface-container-low">
            <td class="px-space-sm py-space-sm border-b border-outline-variant/20">
                <div class="font-code-sm text-[11px] text-on-surface">${escapeHtml(item.provider)} / ${escapeHtml(item.model)}</div>
                <div class="flex flex-wrap gap-1 mt-1">${(item.roles || []).map(role => `<span class="font-code-sm text-[9px] text-outline">${escapeHtml(role)}</span>`).join('<span class="text-outline/40">·</span>')}</div>
            </td>
            <td class="px-space-sm py-space-sm border-b border-outline-variant/20 font-code-sm text-[11px]">${sourceBadge(item.pricing_source)}</td>
            <td class="px-space-sm py-space-sm border-b border-outline-variant/20 text-right font-code-sm text-[11px] text-on-surface-variant">${rate(item.input_usd_per_million_tokens)}</td>
            <td class="px-space-sm py-space-sm border-b border-outline-variant/20 text-right font-code-sm text-[11px] text-on-surface-variant">${rate(item.output_usd_per_million_tokens)}</td>
        </tr>`).join("");

    return `
        <div class="grid grid-cols-2 lg:grid-cols-3 gap-space-sm mb-space-lg">
            ${renderStatCard("Pricing Mode", modeLabel, "payments", "text-tertiary")}
            ${renderStatCard("Tracked Models", models.length, "smart_toy", "text-primary")}
            ${renderStatCard("Catalog Status", fetchLabel, "cloud_sync", pricing.fetch_failed ? "text-error" : "text-secondary")}
        </div>
        ${models.length ? `
          <div class="overflow-auto border border-outline-variant/30 rounded">
            <table class="w-full border-collapse">
              <thead><tr class="bg-surface-container-high text-outline font-code-sm text-[10px]">
                <th class="text-left px-space-sm py-space-xs">Provider / Model</th>
                <th class="text-left px-space-sm py-space-xs">Source</th>
                <th class="text-right px-space-sm py-space-xs">Input USD / 1M</th>
                <th class="text-right px-space-sm py-space-xs">Output USD / 1M</th>
              </tr></thead>
              <tbody>${rows}</tbody>
            </table>
          </div>` : `
          <div class="py-16 text-center text-outline">
            <span class="material-symbols-outlined text-[28px] mb-space-xs">history</span>
            <p class="font-body-sm">Default catalog pricing was used. Model-level pricing was not recorded for this historical run.</p>
          </div>`}
        ${pricing.source_url ? `<div class="mt-space-sm font-code-sm text-[9px] text-outline break-all">Catalog: ${escapeHtml(pricing.source_url)}</div>` : ""}`;
}

function renderBestResultInspector(best) {
    if (!best) return `<div class="text-outline font-body-sm">No successful result is available yet.</div>`;
    return `
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-space-sm mb-space-lg">
            ${renderStatCard("Trial", `#${best.trial_id}`, "tag", "text-primary")}
            ${renderStatCard("Composite", formatScore(best.composite_score), "verified", "text-secondary")}
            ${renderStatCard("Mean Pipeline", best.mean_pipeline_latency_ms == null ? "Not recorded" : `${Number(best.mean_pipeline_latency_ms).toFixed(2)} ms`, "timer", "text-primary")}
            ${renderStatCard("Eval Overhead", best.mean_evaluation_overhead_ms == null ? "Not recorded" : `${Number(best.mean_evaluation_overhead_ms).toFixed(2)} ms`, "analytics", "text-tertiary")}
        </div>
        ${renderSection("Evaluation Metrics", renderMetricGrid(best.metrics || {}))}
        ${renderSection("Mean Stage Timings", renderTimingGrid(best.stage_timings || {}))}
        ${renderSection("Resolved Architecture", renderJsonBlock(best.config || {}))}
        <button type="button" data-best-samples="${best.trial_id}" class="mt-space-base inline-flex items-center gap-1.5 px-3 py-2 rounded bg-primary text-on-primary font-semibold text-[12px]">
            <span class="material-symbols-outlined text-[16px]">quiz</span>View evaluation samples
        </button>
    `;
}

function renderTrialInspectorContent(tab) {
    if (tab === "export") return '<div id="trial-python-export" aria-live="polite" class="p-4">Preparing Python export…</div>';
    const trial = inspectorData.trial || {};
    if (tab === "samples") return renderSamples(inspectorData.samples || []);
    if (tab === "configuration") return renderJsonPanel("Resolved Trial Configuration", trial.resolved_rag_config || {}, "The exact architecture and hyperparameters used by this trial.");
    if (tab === "raw") return renderJsonPanel("Complete Trial Inspector Payload", inspectorData);

    return `
        <div class="grid grid-cols-2 lg:grid-cols-4 gap-space-sm mb-space-lg">
            ${renderStatCard("Status", trial.status || "—", trial.status === "success" ? "check_circle" : "error", trial.status === "success" ? "text-secondary" : "text-error")}
            ${renderStatCard("Composite", trial.status === "success" ? formatScore(trial.composite_score) : "—", "verified", "text-secondary")}
            ${renderStatCard("Mean Pipeline", trial.mean_pipeline_latency_ms == null ? "Not recorded" : `${Number(trial.mean_pipeline_latency_ms).toFixed(2)} ms`, "timer", "text-primary")}
            ${renderStatCard("Cost", trial.cost_usd == null ? "—" : `$${Number(trial.cost_usd).toFixed(5)}`, "payments", "text-tertiary")}
        </div>
        ${trial.error ? renderSection("Error", `<pre class="text-error whitespace-pre-wrap font-code-sm text-[11px]">${escapeHtml(trial.error)}</pre>`) : ""}
        ${renderSection("Evaluation Metrics", renderMetricGrid(trial.metrics || {}))}
        ${trial.answer_refusal_rate != null ? renderSection("Answer Refusals", renderKeyValueGrid({
            answer_refusal_count: trial.answer_refusal_count || 0,
            answer_refusal_rate: formatErrorRate(trial.answer_refusal_rate),
        })) : ""}
        ${renderSection("Latency Summary", renderKeyValueGrid({
            mean_pipeline_latency_ms: trial.mean_pipeline_latency_ms,
            mean_evaluation_overhead_ms: trial.mean_evaluation_overhead_ms,
            total_trial_duration_ms: trial.latency_ms,
        }))}
        ${renderSection("Token Usage", renderKeyValueGrid(trial.token_usage || {}))}
        ${renderSection("Mean Stage Timings", renderTimingGrid({
            query_transform: trial.mean_query_transform_ms,
            query_embedding: trial.mean_query_embedding_ms,
            vector_search: trial.mean_vector_search_ms,
            rerank: trial.mean_rerank_ms,
            relevance_check: trial.mean_relevance_check_ms,
            web_search: trial.mean_web_search_ms,
            generation: trial.mean_generation_ms,
            hallucination_check: trial.mean_hallucination_check_ms,
        }))}
        ${renderSection("Resolved Architecture", renderJsonBlock(trial.resolved_rag_config || {}))}
    `;
}

function renderSamples(samples) {
    if (!samples.length) {
        return `<div class="py-16 text-center text-outline">
            <span class="material-symbols-outlined text-[28px] mb-space-xs">hourglass_empty</span>
            <p class="font-body-sm">No sample traces are available for this trial yet.</p>
        </div>`;
    }

    return `<div class="flex flex-col gap-space-sm">${samples.map((sample, index) => {
        const candidates = sample.retrieved_candidates || [];
        const source = sample.context_source || (sample.web_search_used ? "web_search" : "vector_db");
        const transformedQuery = formatTransformedQuery(sample.transformed_query);
        const transformLabel = queryTransformLabel(sample.query_transform_strategy);
        return `
            <details class="bg-surface-container border border-outline-variant/30 rounded group" ${index === 0 ? "open" : ""}>
                <summary class="list-none cursor-pointer px-space-base py-space-sm flex items-start justify-between gap-space-base hover:bg-surface-container-high transition-colors">
                    <div class="min-w-0">
                        <div class="font-code-sm text-[10px] uppercase tracking-wider text-primary mb-1">Sample #${sample.sample_index ?? index}</div>
                        <div class="font-body-sm text-on-surface line-clamp-2">${escapeHtml(sample.question || "Question unavailable")}</div>
                    </div>
                    <div class="flex items-center gap-space-xs shrink-0">
                        <span class="px-1.5 py-0.5 rounded bg-surface-container-highest text-[10px] font-code-sm ${source === "web_search" ? "text-tertiary" : "text-secondary"}">${escapeHtml(source)}</span>
                        ${sample.answer_refusal ? `<span class="px-1.5 py-0.5 rounded bg-tertiary-container/20 text-[10px] font-code-sm text-tertiary">answer refusal</span>` : ""}
                        <span class="font-code-sm text-[10px] text-outline">${candidates.length} chunks</span>
                        <span class="material-symbols-outlined text-[16px] text-outline group-open:rotate-180 transition-transform">expand_more</span>
                    </div>
                </summary>
                <div class="p-space-base border-t border-outline-variant/20 flex flex-col gap-space-base">
                    ${renderTextBlock("Question", sample.question, "help")}
                    ${transformedQuery ? renderTextBlock(`Transformed Query · ${transformLabel}`, transformedQuery, "auto_awesome", "text-tertiary") : ""}
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-space-sm">
                        ${renderTextBlock("Expected Answer", sample.gold_answer, "task_alt", "text-secondary")}
                        ${renderTextBlock("Predicted Answer", sample.predicted_answer, "smart_toy", "text-primary")}
                    </div>
                    ${sample.gold_context ? renderTextBlock("Reference Context", sample.gold_context, "fact_check", "text-tertiary") : ""}
                    ${renderSection("Sample Metrics", renderMetricGrid(sample.metrics || {}), true)}
                    ${sample.generation_attempted ? renderSection("Answer Observability", renderKeyValueGrid({
                        answer_refusal: sample.answer_refusal,
                        refusal_reason: sample.answer_refusal_reason,
                    }), true) : ""}
                    ${renderSection("Latency, Tokens & Cost", renderKeyValueGrid({
                        pipeline_latency_ms: sample.pipeline_latency_ms,
                        evaluation_overhead_ms: sample.evaluation_overhead_ms,
                        total_sample_latency_ms: sample.latency_ms,
                        query_transform_ms: sample.query_transform_ms,
                        query_embedding_ms: sample.query_embedding_ms,
                        vector_search_ms: sample.vector_search_ms,
                        rerank_ms: sample.rerank_ms,
                        relevance_check_ms: sample.relevance_check_ms,
                        web_search_ms: sample.web_search_ms,
                        generation_ms: sample.generation_ms,
                        hallucination_check_ms: sample.hallucination_check_ms,
                        prompt_tokens: sample.token_usage?.prompt_tokens,
                        completion_tokens: sample.token_usage?.completion_tokens,
                        total_tokens: sample.token_usage?.total_tokens,
                        cost_usd: sample.cost_usd,
                    }), true)}
                    <div>
                        <div class="flex items-center justify-between mb-space-xs">
                            <h4 class="font-headline-sm text-[12px] font-semibold text-on-surface flex items-center gap-1.5"><span class="material-symbols-outlined text-[15px] text-primary">segment</span>Retrieved Chunks</h4>
                            <span class="font-code-sm text-[10px] text-outline">Ranked retrieval context</span>
                        </div>
                        <div class="flex flex-col gap-space-xs">
                            ${candidates.length ? candidates.map((candidate, candidateIndex) => renderCandidate(candidate, candidateIndex)).join("") : `<div class="p-space-sm bg-surface-container-low text-outline font-body-sm rounded">No retrieved chunks recorded.</div>`}
                        </div>
                    </div>
                    ${sample.error ? renderTextBlock("Sample Error", sample.error, "error", "text-error") : ""}
                </div>
            </details>
        `;
    }).join("")}</div>`;
}

function formatTransformedQuery(value) {
    if (Array.isArray(value)) return value.filter(Boolean).map(String).join("\n");
    return value == null ? "" : String(value).trim();
}

function queryTransformLabel(strategy) {
    const labels = {
        rewrite: "Query Rewrite",
        multi_query: "Multi-query Expansion",
        decomposition: "Query Decomposition",
        hyde: "HyDE",
        step_back: "Step-back Query",
    };
    return labels[strategy] || humanizeStage(strategy || "query transformation");
}

function renderCandidate(candidate, index) {
    const rank = candidate.rank || index + 1;
    const text = candidate.page_content || candidate.page_content_preview || "Chunk text unavailable in this older trace.";
    const metadata = candidate.metadata || {};
    const sourceLabel = metadata.source || metadata.source_file || metadata.file_name || metadata.url || "unknown source";
    return `
        <details class="bg-surface-container-low border border-outline-variant/20 rounded">
            <summary class="list-none cursor-pointer px-space-sm py-2 flex items-center justify-between gap-space-sm hover:bg-surface-container-high transition-colors">
                <div class="flex items-center gap-space-xs min-w-0">
                    <span class="w-6 h-6 rounded bg-primary/10 text-primary flex items-center justify-center font-code-sm text-[10px] font-bold shrink-0">${rank}</span>
                    <span class="font-code-sm text-[11px] text-on-surface-variant truncate">${escapeHtml(sourceLabel)}</span>
                </div>
                <span class="font-code-sm text-[10px] text-outline">${text.length.toLocaleString()} chars</span>
            </summary>
            <div class="p-space-sm border-t border-outline-variant/20">
                <pre class="whitespace-pre-wrap break-words font-body-sm text-[12px] leading-5 text-on-surface max-h-80 overflow-y-auto">${escapeHtml(text)}</pre>
                <details class="mt-space-sm pt-space-xs border-t border-outline-variant/20">
                    <summary class="cursor-pointer font-code-sm text-[10px] text-primary">Chunk metadata</summary>
                    <div class="mt-space-xs">${renderJsonBlock(metadata)}</div>
                </details>
            </div>
        </details>
    `;
}

function renderStatCard(label, value, icon, valueClass = "text-on-surface") {
    return `<div class="bg-surface-container border border-outline-variant/30 rounded p-space-sm min-w-0">
        <div class="flex items-center justify-between gap-2 text-outline font-code-sm text-[10px] uppercase tracking-wider">
            <span class="truncate">${escapeHtml(label)}</span>
            <span class="material-symbols-outlined text-[15px]">${icon}</span>
        </div>
        <div class="mt-space-xs font-code-lg text-[14px] font-semibold ${valueClass} truncate" title="${escapeHtml(String(value))}">${escapeHtml(String(value))}</div>
    </div>`;
}

function renderSection(title, body, compact = false) {
    return `<section class="${compact ? "" : "mb-space-lg"}">
        <h3 class="font-code-sm text-[10px] uppercase tracking-wider text-outline mb-space-xs">${escapeHtml(title)}</h3>
        ${body}
    </section>`;
}

function renderKeyValueGrid(data) {
    const entries = Object.entries(data || {}).filter(([, value]) => value !== null && value !== undefined && value !== "");
    if (!entries.length) return `<div class="text-outline font-body-sm">No data recorded.</div>`;
    return `<dl class="grid grid-cols-1 sm:grid-cols-2 gap-px bg-outline-variant/20 border border-outline-variant/20 rounded overflow-hidden">${entries.map(([key, value]) => `
        <div class="bg-surface-container-low p-space-sm min-w-0">
            <dt class="font-code-sm text-[10px] text-outline mb-1">${escapeHtml(key)}</dt>
            <dd class="font-code-sm text-[11px] text-on-surface break-words">${escapeHtml(formatInspectorValue(value))}</dd>
        </div>
    `).join("")}</dl>`;
}

function renderMetricGrid(metrics) {
    const entries = Object.entries(metrics || {});
    if (!entries.length) return `<div class="text-outline font-body-sm">No metrics recorded.</div>`;
    return `<div class="grid grid-cols-2 sm:grid-cols-3 gap-space-xs">${entries.map(([name, value]) => `
        <div class="bg-surface-container-low border border-outline-variant/20 rounded p-space-sm">
            <div class="font-code-sm text-[10px] text-outline truncate" title="${escapeHtml(metricDisplayName(name))}">${escapeHtml(metricDisplayName(name))}</div>
            <div class="font-code-lg text-[14px] text-secondary font-semibold mt-1">${escapeHtml(formatMetricValue(name, value))}</div>
        </div>
    `).join("")}</div>`;
}

function renderTimingGrid(timings) {
    const normalized = {};
    Object.entries(timings || {}).forEach(([key, value]) => {
        normalized[key] = value == null ? "—" : `${Number(value).toFixed(2)} ms`;
    });
    return renderKeyValueGrid(normalized);
}

function renderTextBlock(label, value, icon, colorClass = "text-on-surface") {
    return `<section class="bg-surface-container-low border border-outline-variant/20 rounded p-space-sm">
        <h4 class="font-code-sm text-[10px] uppercase tracking-wider text-outline mb-space-xs flex items-center gap-1.5">
            <span class="material-symbols-outlined text-[14px] ${colorClass}">${icon}</span>${escapeHtml(label)}
        </h4>
        <div class="font-body-sm text-[12px] leading-5 ${colorClass} whitespace-pre-wrap break-words">${escapeHtml(value || "Not recorded")}</div>
    </section>`;
}

function renderJsonPanel(title, value, note = "") {
    return `<div>
        <div class="mb-space-base">
            <h3 class="font-headline-sm text-[14px] font-semibold text-on-surface">${escapeHtml(title)}</h3>
            ${note ? `<p class="font-body-xs text-outline mt-1">${escapeHtml(note)}</p>` : ""}
        </div>
        ${renderJsonBlock(value)}
    </div>`;
}

function renderJsonBlock(value) {
    const text = JSON.stringify(value ?? {}, null, 2);
    return `<pre class="bg-surface-container-low border border-outline-variant/20 rounded p-space-base font-code-sm text-[11px] leading-5 text-on-surface overflow-x-auto whitespace-pre-wrap break-words">${escapeHtml(text)}</pre>`;
}

function formatInspectorValue(value) {
    if (typeof value === "object") return JSON.stringify(value);
    if (typeof value === "boolean") return value ? "true" : "false";
    return String(value);
}

async function copyInspectorJson() {
    if (!inspectorData) return;
    try {
        await navigator.clipboard.writeText(JSON.stringify(inspectorData, null, 2));
        showToast("Inspector JSON copied", "success");
    } catch (_) {
        showToast("Clipboard access is unavailable", "warning");
    }
}

function renderInspectorError(err) {
    const content = document.getElementById("inspector-content");
    if (content) {
        content.innerHTML = `<div class="bg-error-container/20 border border-error/50 rounded p-space-base text-error flex items-start gap-space-sm">
            <span class="material-symbols-outlined text-[18px]">error</span>
            <div><div class="font-semibold">Unable to load inspector data</div><div class="font-code-sm text-[11px] mt-1">${escapeHtml(err.message || String(err))}</div></div>
        </div>`;
    }
}

const SVG_NS = "http://www.w3.org/2000/svg";

function appendSvg(parent, name, attributes = {}, value = null) {
    const element = document.createElementNS(SVG_NS, name);
    Object.entries(attributes).forEach(([key, attribute]) => {
        element.setAttribute(key, String(attribute));
    });
    if (value != null) element.textContent = String(value);
    parent.appendChild(element);
    return element;
}

function showChartMessage(container, message) {
    const placeholder = document.createElement("p");
    placeholder.className = "h-full flex items-center justify-center text-center px-4 text-outline font-code-sm text-[11px]";
    placeholder.textContent = message;
    container.replaceChildren(placeholder);
    delete container.dataset.chartSignature;
}

function niceChartMaximum(value, zeroMaximum = 1) {
    if (value <= 0) return zeroMaximum;
    const magnitude = 10 ** Math.floor(Math.log10(value));
    const normalized = value / magnitude;
    const step = [1, 2, 5, 10].find(candidate => candidate >= normalized) || 10;
    return step * magnitude;
}

function hideTimingTooltip(container) {
    container.querySelector("[data-timing-tooltip]")?.remove();
}

function showTimingTooltip(container, event, point, series, formatValue) {
    let tooltip = container.querySelector("[data-timing-tooltip]");
    if (!tooltip) {
        tooltip = document.createElement("div");
        tooltip.dataset.timingTooltip = "";
        tooltip.style.cssText = [
            "position:absolute", "z-index:95", "pointer-events:none",
            "min-width:176px", "max-width:230px", "padding:8px 10px",
            "border:1px solid #414752", "border-radius:6px",
            "background:#1c2026", "color:#dfe2eb",
            "font:11px/1.55 'JetBrains Mono',monospace",
            "box-shadow:0 8px 20px rgba(0,0,0,.35)",
        ].join(";");
        container.style.position = "relative";
        container.appendChild(tooltip);
    }
    const pipeline = Number(point.trial.mean_pipeline_latency_ms);
    const details = [
        `${series.label} · Trial #${point.trialId}`,
        `Stage mean: ${formatValue(point.value)}`,
        point.trial.mean_pipeline_latency_ms == null || !Number.isFinite(pipeline)
            ? null : `Pipeline mean: ${pipeline.toFixed(1)} ms`,
        `Samples: ${point.trial.completed_samples || 0} / ${point.trial.total_samples || 0}`,
        "Click for trial configuration and samples",
    ].filter(Boolean);
    tooltip.replaceChildren(...details.map((detail, index) => {
        const line = document.createElement("div");
        line.textContent = detail;
        if (index === 0) line.style.color = series.color;
        if (index === details.length - 1) {
            line.style.color = "#8b919d";
            line.style.marginTop = "4px";
        }
        return line;
    }));
    const bounds = container.getBoundingClientRect();
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    tooltip.style.left = `${Math.max(0, Math.min(x + 12, bounds.width - tooltip.offsetWidth - 4))}px`;
    tooltip.style.top = `${Math.max(0, Math.min(y - tooltip.offsetHeight - 10, bounds.height - tooltip.offsetHeight))}px`;
}

function addTimingPointInteraction(svg, container, point, series, marker, formatValue) {
    const hit = appendSvg(svg, "circle", {
        cx: point.x, cy: point.y, r: 9,
        fill: "transparent", "pointer-events": "all",
        role: "button", tabindex: 0,
        "aria-label": `Open trial #${point.trialId} details: ${series.label} ${formatValue(point.value)}`,
    });
    hit.style.cursor = "pointer";
    const baseRadius = marker.getAttribute("r");
    const highlight = () => marker.setAttribute("r", "4");
    const reset = () => marker.setAttribute("r", baseRadius);
    const keyboardLocation = () => {
        const bounds = hit.getBoundingClientRect();
        return { clientX: bounds.left + bounds.width / 2, clientY: bounds.top + bounds.height / 2 };
    };
    hit.addEventListener("mouseenter", event => {
        highlight();
        showTimingTooltip(container, event, point, series, formatValue);
    });
    hit.addEventListener("mousemove", event =>
        showTimingTooltip(container, event, point, series, formatValue));
    hit.addEventListener("mouseleave", () => {
        reset();
        hideTimingTooltip(container);
    });
    hit.addEventListener("focus", () => {
        highlight();
        showTimingTooltip(container, keyboardLocation(), point, series, formatValue);
    });
    hit.addEventListener("blur", () => {
        reset();
        hideTimingTooltip(container);
    });
    const openDetails = () => {
        hideTimingTooltip(container);
        openTrialInspector(point.trialId, "overview");
    };
    hit.addEventListener("click", openDetails);
    hit.addEventListener("keydown", event => {
        if (event.key === "Enter" || event.key === " ") {
            event.preventDefault();
            openDetails();
        }
    });
}

function renderTelemetryChart(container, trials, series, formatValue, emptyMessage, interactive = false) {
    if (!trials || trials.length === 0) {
        showChartMessage(container, "Waiting for completed trials…");
        return;
    }
    const recordedSeries = series.filter(item =>
        item.values.some(value => value != null && Number.isFinite(value) && value >= 0),
    );
    if (recordedSeries.length === 0) {
        showChartMessage(container, emptyMessage);
        return;
    }

    try {
        const width = Math.max(300, Math.round(container.clientWidth || 640));
        const chartSignature = JSON.stringify([
            width, interactive, trials.map(trial => trial.trial_id),
            recordedSeries.map(item => item.values),
        ]);
        if (container.dataset.chartSignature === chartSignature && container.querySelector("svg")) return;
        const height = 220;
        const columns = width >= 560 ? 4 : 2;
        const legendRows = recordedSeries.length > 1
            ? Math.ceil(recordedSeries.length / columns) : 0;
        const left = 72;
        const right = 18;
        const top = legendRows ? 12 + legendRows * 18 + 12 : 18;
        const bottom = 27;
        const plotWidth = width - left - right;
        const plotHeight = height - top - bottom;
        const maximum = niceChartMaximum(Math.max(
            ...recordedSeries.flatMap(item => item.values.filter(value =>
                value != null && Number.isFinite(value) && value >= 0)),
        ), recordedSeries[0].label === "Cumulative Cost" ? 0.001 : 1);

        const svg = document.createElementNS(SVG_NS, "svg");
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "100%");
        svg.setAttribute("aria-hidden", interactive ? "false" : "true");
        svg.style.display = "block";
        svg.style.fontFamily = "JetBrains Mono, monospace";
        svg.style.fontSize = "10px";

        recordedSeries.forEach((item, index) => {
            if (!legendRows) return;
            const col = index % columns;
            const row = Math.floor(index / columns);
            const x = left + col * (plotWidth / columns);
            const y = 12 + row * 18;
            appendSvg(svg, "line", {
                x1: x, y1: y, x2: x + 12, y2: y,
                stroke: item.color, "stroke-width": 2,
            });
            appendSvg(svg, "text", {
                x: x + 16, y: y + 3, fill: "#c0c7d4", "font-size": 9,
            }, item.label);
        });

        for (let tick = 0; tick <= 4; tick += 1) {
            const y = top + (tick / 4) * plotHeight;
            const value = maximum * (1 - tick / 4);
            appendSvg(svg, "line", {
                x1: left, y1: y, x2: width - right, y2: y,
                stroke: "#414752", "stroke-opacity": 0.45,
            });
            appendSvg(svg, "text", {
                x: left - 8, y: y + 3, fill: "#8b919d",
                "text-anchor": "end",
            }, formatValue(value));
        }

        const xAt = index => left + (trials.length === 1
            ? plotWidth / 2 : (index / (trials.length - 1)) * plotWidth);
        const labelStep = Math.max(1, Math.ceil(trials.length / Math.max(
            2, Math.floor(plotWidth / 65),
        )));
        trials.forEach((trial, index) => {
            if (index % labelStep !== 0 && index !== trials.length - 1) return;
            appendSvg(svg, "text", {
                x: xAt(index), y: height - 8, fill: "#8b919d",
                "text-anchor": "middle",
            }, `#${trial.trial_id}`);
        });

        recordedSeries.forEach(item => {
            const points = item.values.map((value, index) => {
                if (value == null || !Number.isFinite(value) || value < 0) return null;
                return {
                    x: xAt(index),
                    y: top + (1 - value / maximum) * plotHeight,
                    value,
                    trialId: trials[index].trial_id,
                    trial: trials[index],
                };
            }).filter(Boolean);
            if (points.length > 1) {
                appendSvg(svg, "polyline", {
                    points: points.map(point => `${point.x},${point.y}`).join(" "),
                    fill: "none", stroke: item.color,
                    "stroke-width": item.label === "Generation" ? 2.3 : 1.8,
                    "stroke-linejoin": "round", "stroke-linecap": "round",
                });
            }
            if (trials.length <= 30 || interactive) points.forEach(point => {
                const marker = appendSvg(svg, "circle", {
                    cx: point.x, cy: point.y, r: interactive && trials.length > 30 ? 1.8 : 2.6,
                    fill: item.color, stroke: "#1c2026", "stroke-width": 1,
                });
                appendSvg(marker, "title", {},
                    `Trial #${point.trialId} · ${item.label}: ${formatValue(point.value)}`);
                if (interactive) addTimingPointInteraction(
                    svg, container, point, item, marker, formatValue,
                );
            });
        });
        container.replaceChildren(svg);
        container.dataset.chartSignature = chartSignature;
    } catch (error) {
        console.error("Failed to render trial trend chart:", error);
        showChartMessage(container, "Chart data is available, but it could not be displayed.");
    }
}

function updateCostChart(trials) {
    const container = document.getElementById("chart-cost");
    if (!container) return;

    let cumulativeCost = 0;
    const values = (trials || []).map(trial => {
        const cost = Number(trial.cost_usd);
        if (trial.cost_usd != null && Number.isFinite(cost)) cumulativeCost += cost;
        return cumulativeCost;
    });
    const hasRecordedCost = (trials || []).some(
        trial => trial.cost_usd != null && Number.isFinite(Number(trial.cost_usd)),
    );
    renderTelemetryChart(container, trials, hasRecordedCost
        ? [{ label: "Cumulative Cost", color: "#fabc45", values }]
        : [], value => `$${value.toFixed(value < 0.01 ? 5 : 3)}`,
        "No trial cost has been recorded yet.");
}

function updateStageTimingChart(trials) {
    const container = document.getElementById("chart-stage-timings");
    if (!container) return;

    const stages = [
        ["Query Transform", "mean_query_transform_ms", "#a2c9ff"],
        ["Query Embedding", "mean_query_embedding_ms", "#7bdb80"],
        ["Vector Search", "mean_vector_search_ms", "#fabc45"],
        ["Rerank", "mean_rerank_ms", "#ffb4ab"],
        ["Relevance Check", "mean_relevance_check_ms", "#fb7185"],
        ["Web Search", "mean_web_search_ms", "#2dd4bf"],
        ["Generation", "mean_generation_ms", "#c084fc"],
        ["Hallucination Check", "mean_hallucination_check_ms", "#38bdf8"],
    ];
    const series = stages.map(([label, field, color]) => ({
        label,
        color,
        values: (trials || []).map(trial => trial[field] == null ? null : Number(trial[field])),
    }));
    renderTelemetryChart(container, trials, series, value => `${Math.round(value)} ms`,
        "No stage timing data has been recorded yet.", true);
}
