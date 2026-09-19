/**
 * Logic for Pipeline Search Runs list: polling runs, rendering
 * telemetry table, filtering by status/search query, and row navigation.
 */

let runsData = [];
let pollInterval = null;
let autoRefreshActive = true;
let currentFilter = "all";
let storageSettings = null;

document.addEventListener("DOMContentLoaded", () => {
    initializeStorageSettings();
    loadRuns();
    pollInterval = setInterval(loadRuns, 3000);

    const refreshToggle = document.getElementById("refresh-toggle");
    if (refreshToggle) {
        refreshToggle.addEventListener("click", () => {
            autoRefreshActive = !autoRefreshActive;
            if (autoRefreshActive) {
                pollInterval = setInterval(loadRuns, 3000);
                refreshToggle.classList.remove("opacity-50");
                showToast("Auto-refresh enabled (3s)", "info");
            } else {
                clearInterval(pollInterval);
                refreshToggle.classList.add("opacity-50");
                showToast("Auto-refresh paused", "info");
            }
        });
    }

    // Status filter tabs
    document.querySelectorAll("[data-filter-status]").forEach(tab => {
        tab.addEventListener("click", () => {
            document.querySelectorAll("[data-filter-status]").forEach(t => {
                t.classList.remove("bg-surface-container", "text-primary", "font-semibold");
                t.classList.add("hover:bg-surface-container", "text-on-surface-variant");
            });
            tab.classList.add("bg-surface-container", "text-primary", "font-semibold");
            tab.classList.remove("hover:bg-surface-container", "text-on-surface-variant");
            currentFilter = tab.getAttribute("data-filter-status");
            renderRunsTable();
        });
    });

    // Search filter
    const searchInput = document.getElementById("search-runs-input");
    if (searchInput) {
        searchInput.addEventListener("input", () => {
            renderRunsTable();
        });
    }

    document.getElementById("storage-settings-button")?.addEventListener("click", openStorageModal);
    document.getElementById("storage-modal-close")?.addEventListener("click", closeStorageModal);
    document.getElementById("storage-cancel-button")?.addEventListener("click", closeStorageModal);
    document.getElementById("storage-save-button")?.addEventListener("click", saveStorageWorkspace);
    document.getElementById("storage-modal")?.addEventListener("click", event => {
        if (event.target.id === "storage-modal") closeStorageModal();
    });
    document.addEventListener("keydown", event => {
        if (event.key === "Escape") closeStorageModal();
    });
});

let comparisonResizeScheduled = false;
window.addEventListener("resize", () => {
    if (comparisonResizeScheduled) return;
    comparisonResizeScheduled = true;
    requestAnimationFrame(() => {
        comparisonResizeScheduled = false;
        renderComparisonChart();
    });
});

async function initializeStorageSettings() {
    try {
        storageSettings = await apiGet("/api/settings/storage");
        renderStorageSettings();
    } catch (err) {
        console.error("Failed to load storage settings:", err);
        const label = document.getElementById("storage-root-label");
        if (label) label.textContent = "Workspace unavailable";
    }
}

function renderStorageSettings() {
    if (!storageSettings) return;
    const label = document.getElementById("storage-root-label");
    const currentPath = document.getElementById("storage-current-path");
    const sourceBadge = document.getElementById("storage-source-badge");
    const pathInput = document.getElementById("storage-path-input");
    const saveButton = document.getElementById("storage-save-button");

    if (label) {
        label.textContent = storageSettings.runs_root;
        label.parentElement.title = `Runs workspace: ${storageSettings.runs_root}`;
    }
    if (currentPath) currentPath.textContent = storageSettings.runs_root;
    if (sourceBadge) sourceBadge.textContent = storageSettings.locked ? "CLI / environment" : storageSettings.source;
    if (pathInput) {
        pathInput.value = storageSettings.runs_root;
        pathInput.disabled = storageSettings.locked;
    }
    if (saveButton) saveButton.disabled = storageSettings.locked;

    renderRecentWorkspaces();
    if (storageSettings.locked) {
        showStorageFeedback(
            "This workspace is locked for the current server session by --runs-dir or MUFFAKIR_RUNS_ROOT.",
            "warning"
        );
    } else {
        hideStorageFeedback();
    }
}

function renderRecentWorkspaces() {
    const section = document.getElementById("storage-recent-section");
    const list = document.getElementById("storage-recent-list");
    if (!section || !list || !storageSettings) return;

    const roots = storageSettings.recent_roots || [];
    section.classList.toggle("hidden", roots.length === 0);
    list.innerHTML = roots.map(root => `
        <button class="flex items-center gap-space-xs rounded border border-outline-variant/30 bg-surface-container-low px-space-sm py-1.5 text-left font-code-sm text-[11px] text-on-surface-variant hover:border-primary/60 hover:text-primary" data-storage-recent="${escapeHtml(root)}" type="button">
          <span class="material-symbols-outlined text-[14px] shrink-0">history</span>
          <span class="break-all">${escapeHtml(root)}</span>
        </button>
    `).join("");
    list.querySelectorAll("[data-storage-recent]").forEach(button => {
        button.addEventListener("click", () => {
            const input = document.getElementById("storage-path-input");
            if (input) input.value = button.getAttribute("data-storage-recent") || "";
            hideStorageFeedback();
        });
    });
}

function openStorageModal() {
    const modal = document.getElementById("storage-modal");
    if (!modal) return;
    renderStorageSettings();
    modal.classList.remove("hidden");
    modal.classList.add("flex");
    if (!storageSettings?.locked) document.getElementById("storage-path-input")?.focus();
}

function closeStorageModal() {
    const modal = document.getElementById("storage-modal");
    if (!modal) return;
    modal.classList.add("hidden");
    modal.classList.remove("flex");
}

function showStorageFeedback(message, type = "error") {
    const feedback = document.getElementById("storage-feedback");
    if (!feedback) return;
    feedback.textContent = message;
    feedback.className = `rounded border px-space-base py-space-sm font-body-sm text-[12px] ${
        type === "warning"
            ? "border-tertiary/40 bg-tertiary/5 text-tertiary"
            : "border-error/40 bg-error/5 text-error"
    }`;
}

function hideStorageFeedback() {
    const feedback = document.getElementById("storage-feedback");
    if (feedback) feedback.classList.add("hidden");
}

async function saveStorageWorkspace() {
    const input = document.getElementById("storage-path-input");
    const saveButton = document.getElementById("storage-save-button");
    const requestedRoot = input?.value.trim();
    if (!requestedRoot) {
        showStorageFeedback("Enter a workspace path before saving.");
        return;
    }

    hideStorageFeedback();
    if (saveButton) saveButton.disabled = true;
    try {
        storageSettings = await apiPut("/api/settings/storage", { runs_root: requestedRoot });
        renderStorageSettings();
        closeStorageModal();
        runsData = [];
        await loadRuns();
        showToast("Runs workspace changed successfully.", "success");
    } catch (err) {
        showStorageFeedback(err.message || "Unable to change the runs workspace.");
    } finally {
        if (saveButton) saveButton.disabled = Boolean(storageSettings?.locked);
    }
}

async function loadRuns() {
    try {
        const res = await apiGet("/api/runs");
        runsData = res.runs || [];
        updateKpiStats();
        renderComparisonChart();
        renderRunsTable();
    } catch (err) {
        console.error("Failed to load runs:", err);
    }
}

function updateKpiStats() {
    const activeCount = runsData.filter(r => r.status === "running").length;
    const completedCount = runsData.filter(r => r.status === "completed").length;
    const failedCount = runsData.filter(r => r.status === "failed").length;

    const countAllEl = document.getElementById("count-all");
    if (countAllEl) countAllEl.textContent = runsData.length;

    const countRunningEl = document.getElementById("count-running");
    if (countRunningEl) countRunningEl.textContent = activeCount;

    const countCompletedEl = document.getElementById("count-completed");
    if (countCompletedEl) countCompletedEl.textContent = completedCount;

    const countFailedEl = document.getElementById("count-failed");
    if (countFailedEl) countFailedEl.textContent = failedCount;

    // Header active summary
    const headerActiveEl = document.getElementById("header-active-runs");
    if (headerActiveEl) {
        headerActiveEl.textContent = `${activeCount} runs active`;
    }
}

function renderRunsTable() {
    const tbody = document.getElementById("runs-table-body");
    if (!tbody) return;

    const searchVal = (document.getElementById("search-runs-input")?.value || "").toLowerCase().trim();

    const filtered = runsData.filter(r => {
        if (currentFilter !== "all" && r.status !== currentFilter) return false;
        if (searchVal) {
            const matchesName = (r.run_name || "").toLowerCase().includes(searchVal);
            const matchesId = (r.run_id || "").toLowerCase().includes(searchVal);
            return matchesName || matchesId;
        }
        return true;
    });

    if (filtered.length === 0) {
        tbody.innerHTML = `
            <tr>
                <td colspan="8" class="py-8 text-center text-outline font-body-sm">
                    No runs found matching current filter.
                </td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = filtered.map(r => {
        const statusBadge = getStatusBadge(r.status, r.completed_trial_count, r.total_trials);
        const scoreDisplay = r.best_score !== null && r.best_score !== undefined
            ? `<span class="text-secondary font-bold text-code-lg">${formatScore(r.best_score)}</span>`
            : `<span class="text-outline">—</span>`;

        const progressPct = r.total_trials > 0
            ? Math.min(100, Math.round((r.completed_trial_count / r.total_trials) * 100))
            : 0;

        const bestChips = renderBestConfigChips(r.best_config);

        return `
            <tr class="hover:bg-surface-container-low transition-colors group cursor-pointer" onclick="window.location.href='/run_detail.html?run_id=${r.run_id}'">
                <td class="py-2 px-3">
                    <div class="flex flex-col min-w-0">
                        <a class="text-primary hover:underline font-semibold truncate flex items-center gap-1.5" href="/run_detail.html?run_id=${r.run_id}">
                            <span class="material-symbols-outlined text-[15px] text-primary-container">hub</span>
                            <span>${escapeHtml(r.run_name)}</span>
                        </a>
                        <div class="flex items-center gap-1.5 text-badge-label text-outline mt-0.5">
                            <span class="px-1 py-0.2 bg-surface-container rounded font-code-sm text-on-surface-variant">${r.run_id.substring(0, 8)}</span>
                            <span>•</span>
                            <span>${r.total_trials} trials</span>
                        </div>
                    </div>
                </td>
                <td class="py-2 px-3 whitespace-nowrap font-code-sm">
                    <div class="flex flex-col">
                        <span class="text-on-surface">${formatDate(r.created_at)}</span>
                        <span class="text-outline text-badge-label">${formatTimeAgo(r.created_at)}</span>
                    </div>
                </td>
                <td class="py-2 px-3 whitespace-nowrap">
                    ${statusBadge}
                </td>
                <td class="py-2 px-3 text-right whitespace-nowrap">
                    ${scoreDisplay}
                </td>
                <td class="py-2 px-3 whitespace-nowrap">
                    <div class="flex flex-col gap-1 w-32">
                        <div class="flex justify-between text-badge-label text-outline font-code-sm">
                            <span class="text-on-surface font-semibold">${r.completed_trial_count}</span>
                            <span>/ ${r.total_trials || "—"}</span>
                        </div>
                        <div class="w-full bg-surface-container-highest h-1 rounded-full overflow-hidden">
                            <div class="bg-primary h-full transition-all" style="width: ${progressPct}%"></div>
                        </div>
                    </div>
                </td>
                <td class="py-2 px-3 whitespace-nowrap text-on-surface-variant font-code-sm">
                    ${r.duration_seconds ? formatDuration(r.duration_seconds) : "—"}
                </td>
                <td class="py-2 px-3">
                    <div class="flex items-center gap-1 flex-wrap font-code-sm">
                        ${bestChips}
                    </div>
                </td>
                <td class="py-2 px-3 text-right whitespace-nowrap">
                    <a href="/run_detail.html?run_id=${r.run_id}" class="inline-flex items-center gap-1 px-2 py-1 bg-surface-container hover:bg-surface-container-high rounded text-on-surface text-badge-label transition-colors">
                        <span>Details</span>
                        <span class="material-symbols-outlined text-[13px]">arrow_forward</span>
                    </a>
                </td>
            </tr>
        `;
    }).join("");
}

function getStatusBadge(status, completed, total) {
    if (status === "running") {
        return `
            <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-surface-container-high text-primary font-semibold text-badge-label">
                <span class="w-1.5 h-1.5 rounded-full bg-primary animate-ping"></span>
                <span>running (${completed}/${total || "?"})</span>
            </span>
        `;
    }
    if (status === "completed") {
        return `
            <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-secondary-container/20 text-secondary font-semibold text-badge-label">
                <span class="w-1.5 h-1.5 rounded-full bg-secondary"></span>
                <span>completed</span>
            </span>
        `;
    }
    if (status === "failed") {
        return `
            <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-error-container/20 text-error font-semibold text-badge-label">
                <span class="w-1.5 h-1.5 rounded-full bg-error"></span>
                <span>failed</span>
            </span>
        `;
    }
    return `
        <span class="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-surface-container text-tertiary font-semibold text-badge-label">
            <span>${status}</span>
        </span>
    `;
}

function renderBestConfigChips(cfg) {
    if (!cfg || typeof cfg !== "object") {
        return `<span class="text-outline text-badge-label">None yet</span>`;
    }
    const chips = [];
    if (cfg.retrieval_method) chips.push(`retrieval:${cfg.retrieval_method}`);
    if (cfg.reranking_method && cfg.reranking) chips.push(`rerank:${cfg.reranking_method}`);
    if (cfg.reranking_model && ["cross_encoder", "pointwise"].includes(String(cfg.reranking_method))) chips.push(`reranker-model:${cfg.reranking_model}`);
    if (cfg.query_transformer_strategy && cfg.query_transformer) chips.push(`exp:${cfg.query_transformer_strategy}`);
    if (cfg.k) chips.push(`k:${cfg.k}`);
    if (cfg.chunking_method) chips.push(`chunk:${cfg.chunking_method}`);

    return chips.slice(0, 4).map(c => `
        <span class="px-1.5 py-0.5 bg-surface-container rounded text-on-surface text-[10px] border border-outline-variant/30">${escapeHtml(c)}</span>
    `).join("");
}

function formatDate(isoString) {
    if (!isoString) return "—";
    try {
        const d = new Date(isoString);
        return d.toLocaleDateString() + " " + d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch (_) {
        return isoString;
    }
}

function renderComparisonChart() {
    const card = document.getElementById("cross-run-card");
    const container = document.getElementById("chart-cross-run");
    if (!container) return;
    if (!runsData || runsData.length === 0) {
        if (card) card.classList.add("hidden");
        return;
    }
    if (card) card.classList.remove("hidden");

    // Chronological order: up to 12 most recent runs.
    const recentRuns = [...runsData].reverse().slice(-12);
    try {
        const width = Math.max(300, Math.round(container.clientWidth || 640));
        const height = 220;
        const left = 42;
        const right = 16;
        const top = 14;
        const bottom = 38;
        const plotWidth = width - left - right;
        const plotHeight = height - top - bottom;
        const step = plotWidth / recentRuns.length;
        const barWidth = Math.min(36, step * 0.65);
        const svgNs = "http://www.w3.org/2000/svg";
        const svg = document.createElementNS(svgNs, "svg");
        svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
        svg.setAttribute("width", "100%");
        svg.setAttribute("height", "100%");
        svg.setAttribute("aria-hidden", "true");
        svg.style.display = "block";
        svg.style.fontFamily = "JetBrains Mono, monospace";

        const add = (name, attributes, label = null, parent = svg) => {
            const element = document.createElementNS(svgNs, name);
            Object.entries(attributes).forEach(([key, value]) =>
                element.setAttribute(key, String(value)));
            if (label != null) element.textContent = String(label);
            parent.appendChild(element);
            return element;
        };
        for (let tick = 0; tick <= 4; tick += 1) {
            const y = top + tick * plotHeight / 4;
            add("line", {
                x1: left, y1: y, x2: width - right, y2: y,
                stroke: "#414752", "stroke-opacity": 0.45,
            });
            add("text", {
                x: left - 7, y: y + 3, fill: "#8b919d",
                "font-size": 10, "text-anchor": "end",
            }, `${100 - tick * 25}%`);
        }

        const labelStep = Math.max(1, Math.ceil(recentRuns.length /
            Math.max(2, Math.floor(plotWidth / 75))));
        recentRuns.forEach((run, index) => {
            const score = run.best_score == null ? 0 : Number(run.best_score) * 100;
            const boundedScore = Number.isFinite(score)
                ? Math.min(100, Math.max(0, score)) : 0;
            const color = {
                completed: "#7bdb80",
                running: "#a2c9ff",
                failed: "#ffb4ab",
            }[run.status] || "#fabc45";
            const x = left + step * (index + 0.5);
            const barHeight = boundedScore / 100 * plotHeight;
            const bar = add("rect", {
                x: x - barWidth / 2, y: top + plotHeight - Math.max(barHeight, 2),
                width: barWidth, height: Math.max(barHeight, 2),
                rx: 3, fill: color, "fill-opacity": 0.75,
            });
            add("title", {},
                `${run.run_name || run.run_id} · ${boundedScore.toFixed(1)}% · ${run.status} · ${run.completed_trial_count}/${run.total_trials} trials`,
                bar);
            if (index % labelStep !== 0 && index !== recentRuns.length - 1) return;
            const name = run.run_name || run.run_id.slice(0, 8);
            add("text", {
                x, y: height - 10, fill: "#8b919d",
                "font-size": 10, "text-anchor": "middle",
            }, name.length > 12 ? `${name.slice(0, 11)}…` : name);
        });
        container.replaceChildren(svg);
    } catch (error) {
        console.error("Failed to render cross-run comparison:", error);
        const message = document.createElement("p");
        message.className = "h-full flex items-center justify-center text-outline font-code-sm text-[11px]";
        message.textContent = "Run scores are available, but the comparison could not be displayed.";
        container.replaceChildren(message);
    }
}
