/* Shared, provider-aware parameter editors and per-model generation variants. */
const LLMSettings = (() => {
    const defaults = () => ({ temperature: 0, max_tokens: 4096 });
    const roles = ["generation", "judge", "query_transform", "reranker", "dataset"];
    const drafts = Object.fromEntries(roles.map(role => [role, defaults()]));
    const panels = {};
    let capabilities = {};
    let variants = [];
    let editing = null;
    let dialog;
    const clone = value => JSON.parse(JSON.stringify(value));
    const value = id => document.getElementById(id)?.value || "";
    const checked = id => Boolean(document.getElementById(id)?.checked);
    const esc = value => escapeHtml(String(value));
    const inputClass = "bg-surface-container-low border border-outline-variant/50 rounded px-space-sm py-1.5 text-on-surface font-code-sm text-[12px] w-full";
    const buttonClass = "px-space-sm py-1 rounded border border-outline-variant/40 text-primary text-[12px] hover:bg-surface-container-high";
    const hints = {
        temperature: "Lower values reduce sampling variation. Omit for models that do not accept temperature.",
        max_tokens: "Upper bound on output tokens, not a target length. Models may impose smaller limits.",
        top_p: "Nucleus sampling. Usually adjust temperature or top-p, rather than both.",
        stop: "One stop sequence per line; matching text can end generation early.",
        seed: "Best-effort repeatability only; identical answers are not guaranteed.",
        top_k: "Limit sampling to the highest-ranked candidate tokens.",
        frequency_penalty: "Discourage repeated tokens based on frequency.",
        presence_penalty: "Discourage tokens already present in the output.",
        timeout_seconds: "Timeout for a provider request, in seconds.",
        max_retries: "Provider request retries only; independent of trial and dataset validation retries.",
    };

    function provider(role) {
        const state = deriveWorkflowState();
        if (role === "variant") return value("llm-variant-provider");
        if (role === "generation") return value("llm-provider-select");
        if (role === "judge" && checked("judge-llm-override-checkbox")) return value("judge-llm-provider-select");
        if (role === "query_transform" && state.showQueryTransformFields) return value("query-transform-llm-provider-select");
        if (role === "dataset" && state.showDatasetLlmFields) return value("dataset-llm-provider-select");
        if (role === "reranker") {
            if (checked("reranker-llm-override-checkbox")) return value("reranker-llm-provider-select");
            if (state.retrievalOnly) return value("query-transform-llm-provider-select");
        }
        return value("llm-provider-select");
    }

    function enabled(role, state = deriveWorkflowState()) {
        if (role === "generation") return state.needsAnswerLlm;
        if (role === "dataset") return state.needsDatasetLlm;
        if (role === "query_transform") return state.queryExpansionActive;
        if (role === "reranker") return state.llmRerankingActive && !state.webSearchOnly;
        return state.allowGenerationMetrics && collectSelectedMetrics().some(m => ["faithfulness", "answer_correctness", "llm_judge_rating"].includes(m));
    }

    function summary(parameters) {
        return Object.entries(parameters || {}).map(([key, val]) => `${key}=${val === null ? "omit" : JSON.stringify(val)}`).join(", ");
    }

    function problems(parameters, providerName) {
        const specs = capabilities[providerName] || {};
        return Object.entries(parameters).flatMap(([key, val]) => {
            const spec = specs[key];
            if (!spec) return [`${providerName} does not support ${key}; clear it or select another provider.`];
            if (val === null && spec.nullable) return [];
            if (spec.type === "strings") return Array.isArray(val) && val.every(s => typeof s === "string" && s.length) ? [] : [`${key} must contain non-empty strings.`];
            if (typeof val !== "number" || !Number.isFinite(val) || (spec.integer && !Number.isInteger(val))) return [`${key} must be a ${spec.integer ? "whole" : "finite"} number.`];
            if ((spec.min !== undefined && val < spec.min) || (spec.max !== undefined && val > spec.max) || (spec.exclusive_min !== undefined && val <= spec.exclusive_min)) return [`${key} is outside the supported range for ${providerName}.`];
            return [];
        });
    }

    function changed() {
        if (typeof invalidateDryRun === "function") invalidateDryRun();
        updateComboCount();
    }

    function drawEditor(role, host) {
        const name = provider(role);
        const specs = capabilities[name] || {};
        const parameters = drafts[role];
        host.dataset.provider = name;
        host.innerHTML = `<p class="text-body-xs text-outline my-space-sm">Adapter-supported settings for ${esc(name)}. Individual models or custom endpoints may reject options. Blank optional fields use provider defaults.</p>
            <div data-parameter-grid class="grid grid-cols-1 sm:grid-cols-2 gap-space-sm"></div>
            <p data-parameter-error role="alert" class="text-error text-body-xs my-space-sm"></p>
            <p data-parameter-summary class="text-outline font-code-sm text-[11px] break-words my-space-sm"></p>
            <button type="button" data-parameter-reset class="${buttonClass}">Reset settings</button>`;
        const grid = host.querySelector("[data-parameter-grid]");
        [...new Set([...Object.keys(specs), ...Object.keys(parameters)])].forEach(key => {
            const spec = specs[key];
            const label = document.createElement("label");
            label.className = "flex flex-col gap-1 text-[12px] text-on-surface";
            if (!spec) {
                label.innerHTML = `<span class="text-error">Unsupported: ${esc(key)} = ${esc(JSON.stringify(parameters[key]))}</span><button type="button" class="${buttonClass}">Clear ${esc(key)}</button>`;
                label.querySelector("button").onclick = () => { delete parameters[key]; drawEditor(role, host); changed(); };
                grid.appendChild(label);
                return;
            }
            label.innerHTML = `<span>${esc(spec.label)}</span><small class="text-outline">${esc(hints[key])}</small>`;
            const input = document.createElement(spec.type === "strings" ? "textarea" : "input");
            input.className = inputClass;
            input.dataset.parameter = key;
            input.setAttribute("aria-label", `${role}: ${spec.label}`);
            if (spec.type !== "strings") {
                input.type = "number";
                input.step = spec.integer ? "1" : "any";
                if (spec.min !== undefined) input.min = spec.min;
                if (spec.max !== undefined) input.max = spec.max;
            } else input.rows = 2;
            input.value = parameters[key] === null || parameters[key] === undefined ? "" : (Array.isArray(parameters[key]) ? parameters[key].join("\n") : parameters[key]);
            input.placeholder = key in defaults() ? `Library default: ${defaults()[key]}` : "Provider default";
            input.disabled = parameters[key] === null;
            input.oninput = () => {
                if (input.value === "") delete parameters[key];
                else parameters[key] = spec.type === "strings" ? input.value.split("\n").filter(s => s.length) : Number(input.value);
                if (input.validity.badInput) parameters[key] = NaN;
                refreshStatus(role, host);
                changed();
            };
            label.appendChild(input);
            if (spec.nullable) {
                const omit = document.createElement("label");
                omit.innerHTML = `<input type="checkbox" ${parameters[key] === null ? "checked" : ""}/> Omit temperature from request`;
                omit.querySelector("input").onchange = event => {
                    if (event.target.checked) parameters[key] = null;
                    else parameters[key] = 0;
                    drawEditor(role, host); changed();
                };
                // Keep labels separate: nested labels are invalid HTML.
                grid.appendChild(label); grid.appendChild(omit);
            } else grid.appendChild(label);
        });
        host.querySelector("[data-parameter-reset]").onclick = () => { drafts[role] = defaults(); drawEditor(role, host); changed(); };
        refreshStatus(role, host);
    }

    function refreshStatus(role, host) {
        host.querySelector("[data-parameter-error]").textContent = problems(drafts[role], provider(role)).join(" ");
        host.querySelector("[data-parameter-summary]").textContent = `Effective settings: ${summary(effective(role))}`;
    }

    function effective(role) {
        return { ...defaults(), ...drafts[role] };
    }

    function parameterFields() {
        const result = {};
        roles.filter(role => enabled(role)).forEach(role => {
            result[role === "generation" ? "llm_parameters" : `${role}_llm_parameters`] = effective(role);
        });
        return result;
    }

    function variantIdentity(entry) {
        const p = Object.fromEntries(Object.entries(entry.parameters || {}).sort(([a], [b]) => a.localeCompare(b)));
        return JSON.stringify([entry.provider.trim().toLowerCase(), entry.model.trim(), p]);
    }

    function renderVariants() {
        const list = document.getElementById("llm-variant-list");
        list.innerHTML = "";
        variants.forEach((entry, index) => {
            const row = document.createElement("div");
            row.className = "rounded border border-outline-variant/30 p-space-sm my-space-sm";
            row.innerHTML = `<p class="text-on-surface text-[13px]">${esc(entry.provider)} / ${esc(entry.model)}</p><p class="text-outline text-[11px] break-words">${esc(summary(entry.parameters))}</p><div class="flex gap-space-sm mt-space-xs"><button type="button" class="${buttonClass}" data-edit>Edit</button><button type="button" class="${buttonClass}" data-duplicate>Duplicate</button><button type="button" class="${buttonClass}" data-remove>Remove</button></div>`;
            row.querySelector("[data-edit]").onclick = () => openVariant(entry, index);
            row.querySelector("[data-duplicate]").onclick = () => openVariant(entry, null);
            row.querySelector("[data-remove]").onclick = () => { variants.splice(index, 1); renderVariants(); changed(); };
            list.appendChild(row);
        });
    }

    function openVariant(entry, index) {
        editing = index;
        drafts.variant = clone(entry.parameters || defaults());
        dialog.innerHTML = `<h2 class="text-on-surface font-bold mb-space-sm">${index === null ? "Add" : "Edit"} generation variant</h2><label>Provider<select id="llm-variant-provider" class="${inputClass}">${Object.keys(capabilities).map(p => `<option value="${esc(p)}">${esc(p)}</option>`).join("")}</select></label><label>Model<input id="llm-variant-model" class="${inputClass}" value="${esc(entry.model)}"/></label><div id="llm-variant-editor"></div><p id="llm-variant-error" class="text-error" role="alert"></p><div class="flex gap-space-sm mt-space-base"><button type="button" id="llm-variant-save" class="${buttonClass}">Save variant</button><button type="button" id="llm-variant-cancel" class="${buttonClass}">Cancel</button></div>`;
        document.getElementById("llm-variant-provider").value = entry.provider;
        const editor = document.getElementById("llm-variant-editor");
        drawEditor("variant", editor);
        document.getElementById("llm-variant-provider").onchange = () => drawEditor("variant", editor);
        document.getElementById("llm-variant-cancel").onclick = () => dialog.close();
        document.getElementById("llm-variant-save").onclick = () => {
            const candidate = { provider: provider("variant"), model: value("llm-variant-model").trim(), parameters: effective("variant") };
            const errors = problems(candidate.parameters, candidate.provider);
            if (!candidate.model) errors.push("Enter a model identifier.");
            if (variants.some((v, i) => i !== editing && variantIdentity(v) === variantIdentity(candidate))) errors.push("This exact configuration already exists. Change a parameter before saving the duplicate.");
            if (errors.length) { document.getElementById("llm-variant-error").textContent = errors.join(" "); return; }
            if (editing === null) variants.push(candidate); else variants[editing] = candidate;
            dialog.close(); renderVariants(); changed();
        };
        dialog.showModal();
    }

    function init(catalog) {
        capabilities = catalog.llm_parameter_capabilities || {};
        const targets = { generation: "provider-data-card", judge: "judge-override-controls", query_transform: "query-transform-provider-card", reranker: "reranker-llm-card", dataset: "dataset-provider-card" };
        roles.forEach(role => {
            const parent = document.getElementById(targets[role]);
            if (!parent) return;
            const details = document.createElement("details");
            details.className = "mt-space-base border-t border-outline-variant/30 pt-space-sm";
            details.innerHTML = '<summary class="cursor-pointer text-primary text-[13px]">Advanced LLM settings</summary><div></div>';
            parent.appendChild(details);
            panels[role] = details;
            drawEditor(role, details.querySelector("div"));
        });
        const panel = document.createElement("section");
        panel.className = "mt-space-base border-t border-outline-variant/30 pt-space-sm";
        panel.innerHTML = `<h3 class="text-on-surface font-bold">Generation model variants</h3><p class="text-body-xs text-outline my-space-sm">No variants: use the answer model above once. With variants: compare only the listed configurations. Each variant multiplies the other search dimensions. Auxiliary settings stay fixed. Credentials and endpoint are shared with the answer configuration.</p><button type="button" class="${buttonClass}" id="llm-add-variant">Add model variant</button><div id="llm-variant-list"></div>`;
        document.getElementById("provider-data-card")?.appendChild(panel);
        document.getElementById("llm-add-variant").onclick = () => openVariant({ provider: provider("generation"), model: value("llm-model-input"), parameters: effective("generation") }, null);
        dialog = document.createElement("dialog");
        dialog.className = "bg-surface-container text-on-surface rounded border border-outline-variant/50 p-space-lg";
        dialog.style.cssText = "max-width:720px;width:92vw;max-height:85vh;overflow:auto";
        document.body.appendChild(dialog);
        document.addEventListener("change", () => refresh());
        refresh();
    }

    function refresh() {
        Object.entries(panels).forEach(([role, panel]) => {
            panel.hidden = !enabled(role);
            const host = panel.querySelector("div");
            if (host.dataset.provider !== provider(role)) drawEditor(role, host);
        });
    }

    function errors() {
        const issues = roles.filter(role => enabled(role)).flatMap(role => problems(effective(role), provider(role)).map(s => `${role}: ${s}`));
        if (enabled("generation")) variants.forEach(entry => issues.push(...problems(entry.parameters, entry.provider)));
        return issues;
    }

    function reset() {
        variants = [];
        roles.forEach(role => { drafts[role] = defaults(); if (panels[role]) drawEditor(role, panels[role].querySelector("div")); });
        renderVariants();
    }

    return { init, refresh, reset, errors, parameterFields, summary, problems, variantIdentity,
        variants: () => enabled("generation") ? clone(variants) : [] };
})();
