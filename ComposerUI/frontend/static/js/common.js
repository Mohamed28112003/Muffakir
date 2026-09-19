/**
 * Common utilities, API wrappers, and toast notifications for ComposerUI.
 */

async function apiGet(url) {
    const res = await fetch(url);
    if (!res.ok) {
        let errDetail = res.statusText;
        try {
            const data = await res.json();
            if (data.detail) errDetail = data.detail;
            else if (data.error) errDetail = data.error;
        } catch (_) {}
        throw new Error(errDetail || `Request failed with status ${res.status}`);
    }
    return await res.json();
}

async function apiPost(url, payload) {
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!res.ok) {
        let errDetail = res.statusText;
        if (data.detail) errDetail = data.detail;
        else if (data.error) errDetail = data.error;
        throw new Error(errDetail || `Request failed with status ${res.status}`);
    }
    return data;
}

async function apiPut(url, payload) {
    const res = await fetch(url, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    let data = {};
    try {
        data = await res.json();
    } catch (_) {}
    if (!res.ok) {
        let errDetail = res.statusText;
        if (data.detail) errDetail = data.detail;
        else if (data.error) errDetail = data.error;
        throw new Error(errDetail || `Request failed with status ${res.status}`);
    }
    return data;
}

function showToast(message, type = "info") {
    let container = document.getElementById("status-toast");
    if (!container) {
        container = document.createElement("div");
        container.id = "status-toast";
        container.className = "fixed bottom-4 right-4 z-50 flex flex-col gap-2 max-w-md pointer-events-none";
        document.body.appendChild(container);
    }

    const toast = document.createElement("div");
    const colors = {
        success: "bg-surface-container-high border-secondary text-secondary",
        error: "bg-surface-container-high border-error text-error",
        info: "bg-surface-container-high border-primary text-primary",
        warning: "bg-surface-container-high border-tertiary text-tertiary",
    };

    const icons = {
        success: "check_circle",
        error: "error",
        info: "info",
        warning: "warning",
    };

    const chosenColor = colors[type] || colors.info;
    const chosenIcon = icons[type] || icons.info;

    toast.className = `flex items-center gap-2 px-3 py-2 rounded border shadow-lg font-code-sm text-code-sm pointer-events-auto transition-all transform duration-200 translate-y-2 opacity-0 ${chosenColor}`;
    toast.innerHTML = `
        <span class="material-symbols-outlined text-[18px]">${chosenIcon}</span>
        <span class="flex-1 text-on-surface font-body-sm">${escapeHtml(message)}</span>
        <button class="text-outline hover:text-on-surface ml-2 leading-none" onclick="this.parentElement.remove()">
            <span class="material-symbols-outlined text-[14px]">close</span>
        </button>
    `;

    container.appendChild(toast);
    requestAnimationFrame(() => {
        toast.classList.remove("translate-y-2", "opacity-0");
    });

    setTimeout(() => {
        toast.classList.add("opacity-0", "translate-y-2");
        setTimeout(() => toast.remove(), 300);
    }, 4500);
}

function escapeHtml(str) {
    if (!str) return "";
    return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
}

function formatScore(score) {
    if (score === null || score === undefined) return "—";
    const pct = (score * 100).toFixed(1);
    return `${pct}%`;
}

function metricDisplayName(name) {
    return name === "llm_judge_rating" ? "LLM Judge Rating" : name;
}

function formatMetricValue(name, value) {
    if (value === null || value === undefined) return "—";
    if (name === "llm_judge_rating" && typeof value === "number") {
        const rating = Number(value).toFixed(2).replace(/\.?0+$/, "");
        return `${rating} / 5`;
    }
    return typeof value === "number" ? formatScore(value) : String(value);
}

function formatDuration(seconds) {
    if (seconds === null || seconds === undefined) return "—";
    const s = Math.floor(seconds);
    const hrs = Math.floor(s / 3600);
    const mins = Math.floor((s % 3600) / 60);
    const secs = s % 60;
    if (hrs > 0) {
        return `${String(hrs).padStart(2, "0")}:${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
    }
    return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

function formatTimeAgo(isoString) {
    if (!isoString) return "—";
    try {
        const date = new Date(isoString);
        const diffSecs = Math.floor((new Date() - date) / 1000);
        if (diffSecs < 60) return `${diffSecs}s ago`;
        if (diffSecs < 3600) return `${Math.floor(diffSecs / 60)}m ago`;
        if (diffSecs < 86400) return `${Math.floor(diffSecs / 3600)}h ago`;
        return `${Math.floor(diffSecs / 86400)}d ago`;
    } catch (_) {
        return isoString;
    }
}
