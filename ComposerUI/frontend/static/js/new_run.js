/**
 * Logic for New Search Run page: search space chip toggling, real-time combo
 * counting, provider/vector-db/parser availability, custom-value stages
 * (k / chunking / embedding), eval-dataset mode, train/test split, metrics
 * selection, preflight dry-run, and run launch.
 */

let catalogData = null;
let highestUnlockedStep = 0;
let workflowRenderInProgress = false;
let validatedConfigSignature = null;
let dryRunInFlight = false;
let promptDefinitions = [];
let promptResolutionSignature = null;
let promptResolutionError = null;
let promptResolveSequence = 0;
let renderedPromptLanguage = "ar";
const promptDraftsByLanguage = { ar: {}, en: {} };
const customPricingAutoDrafts = {};
const customPricingManualDrafts = [];
let customPricingManualSequence = 0;
let activeDetectedPricingKeys = new Set();

const WORKFLOW_STEPS = [
    "tab-panel-data",
    "tab-panel-parser",
    "tab-panel-providers",
    "tab-panel-eval",
    "tab-panel-search-space",
    "tab-panel-prompts",
    "tab-panel-review",
];

const PARSER_FIELD_LABELS = {
    endpoint: { label: "Endpoint", type: "text" },
    api_key: { label: "API Key", type: "password" },
    language: { label: "Language", type: "text" },
    export_type: { label: "Export Type", type: "text" },
};

const SEARCH_PROVIDER_FIELD_LABELS = {
    max_depth: { label: "Max Depth", type: "number" },
    time_limit: { label: "Time Limit (s)", type: "number" },
    max_urls: { label: "Max URLs", type: "number" },
    max_results: { label: "Max Results", type: "number" },
};

document.addEventListener("DOMContentLoaded", async () => {
    organizeWorkflowPanels();
    initNotebookTabs();
    await initCatalog();
    LLMSettings.init(catalogData || {});
    initChipHandlers();
    initCustomAddHandlers();
    initParserHandlers();
    initJudgeLlmHandlers();
    initTaskLlmHandlers();
    initCustomPricingHandlers();
    initRetrievalSourceHandlers();
    initPipelineModeHandlers();
    initDatasetModeHandlers();
    initPromptManagerHandlers();
    initFormHandlers();
    updateComboCount();
    initGuidedWorkflow();
});

function organizeWorkflowPanels() {
    const moves = [
        ["documents-config", "data-corpus-slot"],
        ["dataset-strategy-card", "data-dataset-slot"],
        ["split-policy-card", "data-dataset-slot"],
        ["provider-data-card", "answer-provider-slot"],
        ["query-transform-provider-card", "answer-provider-slot"],
    ];
    moves.forEach(([elementId, targetId]) => {
        const element = document.getElementById(elementId);
        const target = document.getElementById(targetId);
        if (element && target) target.appendChild(element);
    });
}

async function initCatalog() {
    try {
        catalogData = await apiGet("/api/catalog");
        populateProviderSelect();
        populateRerankerChips();
        populateRerankerModelSuggestions();
        populateRerankerLlmProviderSelect();
        populateVectorDbChips();
        populateChunkingMethodSelect();
        populateChunkingQuickfill();
        populateEmbeddingSuggestions();
        populateParserSelect();
        populateMetricsCheckboxes();
        populateDeviceSelect();
        populateJudgeLlmProviderSelect();
        populateDatasetLlmProviderSelect();
        populateQueryTransformLlmProviderSelect();
        populateSearchProviderSelect();
        applyCapabilityAvailability();
    } catch (err) {
        showToast("Failed to load provider catalog: " + err.message, "error");
    }
}

function unavailableLabel(entry) {
    return entry.install_command || `pip install "Muffakir[${entry.extra || entry.name}]"`;
}

function populateProviderSelect() {
    const select = document.getElementById("llm-provider-select");
    if (!select || !catalogData.providers) return;
    select.innerHTML = "";
    catalogData.providers.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.available ? p.name : `${p.name} (${unavailableLabel(p)})`;
        opt.disabled = !p.available;
        if (p.name === "openai") opt.selected = true;
        select.appendChild(opt);
    });
}

function populateDatasetLlmProviderSelect() {
    const select = document.getElementById("dataset-llm-provider-select");
    if (!select || !catalogData?.providers) return;
    select.innerHTML = "";
    catalogData.providers.forEach(provider => {
        const option = document.createElement("option");
        option.value = provider.name;
        option.textContent = provider.available
            ? provider.name
            : `${provider.name} (${unavailableLabel(provider)})`;
        option.disabled = !provider.available;
        if (provider.name === "openai") option.selected = true;
        select.appendChild(option);
    });
}

function populateDeviceSelect() {
    const select = document.getElementById("device-select");
    if (!select || !catalogData.devices) return;
    select.innerHTML = "";
    catalogData.devices.forEach(d => {
        const opt = document.createElement("option");
        opt.value = d.name;
        opt.textContent = d.available ? d.name : `${d.name} (${d.requires})`;
        opt.disabled = !d.available;
        if (d.name === "auto") opt.selected = true;
        select.appendChild(opt);
    });
}

function populateSearchProviderSelect() {
    const select = document.getElementById("search-provider-select");
    if (!select || !catalogData.web_search_providers) return;
    select.innerHTML = "";
    catalogData.web_search_providers.forEach(w => {
        const opt = document.createElement("option");
        opt.value = w.name;
        opt.textContent = w.available ? w.name : `${w.name} (${unavailableLabel(w)})`;
        opt.disabled = !w.available;
        select.appendChild(opt);
    });
    renderSearchProviderExtraFields(select.value);
}

function renderSearchProviderExtraFields(providerName) {
    const container = document.getElementById("search-provider-extra-fields");
    if (!container) return;
    container.innerHTML = "";
    if (!providerName || !catalogData || !catalogData.web_search_providers) return;
    const provider = catalogData.web_search_providers.find(w => w.name === providerName);
    if (!provider) return;

    provider.fields.forEach(fieldName => {
        const meta = SEARCH_PROVIDER_FIELD_LABELS[fieldName] || { label: fieldName, type: "text" };
        const wrap = document.createElement("div");
        wrap.className = "flex flex-col gap-1";
        const label = document.createElement("label");
        label.className = "font-code-sm text-[11px] text-outline";
        label.textContent = meta.label;
        label.setAttribute("for", `search-provider-field-${fieldName}`);
        const input = document.createElement("input");
        input.className = "bg-surface-container-low border border-outline-variant/50 rounded px-space-sm py-1 font-code-sm text-[12px] text-on-surface focus:outline-none focus:border-primary w-24";
        input.id = `search-provider-field-${fieldName}`;
        input.type = meta.type;
        wrap.appendChild(label);
        wrap.appendChild(input);
        container.appendChild(wrap);
    });
}

function collectSearchProviderFieldConfig() {
    const providerName = document.getElementById("search-provider-select")?.value;
    if (!providerName || !catalogData) return {};
    const provider = catalogData.web_search_providers.find(w => w.name === providerName);
    if (!provider) return {};
    const config = {};
    provider.fields.forEach(fieldName => {
        const el = document.getElementById(`search-provider-field-${fieldName}`);
        if (el && el.value.trim()) config[fieldName] = Number(el.value);
    });
    return config;
}

function populateJudgeLlmProviderSelect() {
    const select = document.getElementById("judge-llm-provider-select");
    if (!select || !catalogData.providers) return;
    select.innerHTML = "";
    catalogData.providers.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.available ? p.name : `${p.name} (${unavailableLabel(p)})`;
        opt.disabled = !p.available;
        if (p.name === "openai") opt.selected = true;
        select.appendChild(opt);
    });
}

function populateQueryTransformLlmProviderSelect() {
    const select = document.getElementById("query-transform-llm-provider-select");
    if (!select || !catalogData.providers) return;
    select.innerHTML = "";
    catalogData.providers.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.available ? p.name : `${p.name} (${unavailableLabel(p)})`;
        opt.disabled = !p.available;
        if (p.name === "openai") opt.selected = true;
        select.appendChild(opt);
    });
}

function makeToggleChip(value, label, opts = {}) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip-toggle flex items-center gap-space-xs px-space-sm py-space-xs rounded bg-surface-container-low border border-outline-variant/40 font-code-sm text-code-sm text-outline hover:border-primary/50 transition-all";
    chip.setAttribute("data-val", value);
    chip.setAttribute("data-selected", "false");
    if (opts.disabled) {
        chip.disabled = true;
        chip.classList.add("opacity-40", "cursor-not-allowed");
        chip.title = opts.title || "Not available";
    }
    const checkbox = document.createElement("span");
    checkbox.className = "chip-checkbox w-3.5 h-3.5 rounded border border-outline-variant flex items-center justify-center text-[10px]";
    const text = document.createElement("span");
    text.textContent = label;
    chip.appendChild(checkbox);
    chip.appendChild(text);

    return chip;
}

function makeRemovableChip(value, label) {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "chip-toggle active chip-custom flex items-center gap-space-xs px-space-sm py-space-xs rounded bg-surface-container-high border border-primary text-primary font-code-sm text-code-sm transition-all shadow-[0_0_8px_rgba(88,166,255,0.25)]";
    chip.setAttribute("data-val", value);
    chip.setAttribute("data-selected", "true");
    chip.title = "Click to remove";
    const checkbox = document.createElement("span");
    checkbox.className = "chip-checkbox w-3.5 h-3.5 rounded bg-primary text-on-primary flex items-center justify-center text-[10px] font-bold";
    checkbox.textContent = "✕";
    const text = document.createElement("span");
    text.textContent = label;
    chip.appendChild(checkbox);
    chip.appendChild(text);

    chip.addEventListener("click", () => {
        chip.remove();
        updateComboCount();
    });
    return chip;
}

function populateVectorDbChips() {
    const group = document.querySelector('[data-stage-group="vector_db_provider"]');
    if (!group || !catalogData.vector_dbs) return;
    group.innerHTML = "";
    catalogData.vector_dbs.forEach(v => {
        const title = v.available ? null : unavailableLabel(v);
        group.appendChild(makeToggleChip(v.name, v.name, { disabled: !v.available, title }));
    });
}

function populateRerankerChips() {
    const group = document.querySelector('[data-stage-group="reranking"]');
    if (!group || !catalogData?.rerankers) return;
    group.innerHTML = "";
    catalogData.rerankers.forEach(reranker => {
        const title = reranker.available
            ? reranker.description
            : unavailableLabel(reranker);
        const chip = makeToggleChip(
            reranker.name,
            reranker.label || reranker.name,
            { disabled: !reranker.available, title },
        );
        if (reranker.name === "semantic_similarity" && reranker.available) {
            setChipSelected(chip, true);
        }
        group.appendChild(chip);
    });
    if (!group.querySelector('.chip-toggle[data-selected="true"]')) {
        const noneChip = group.querySelector('.chip-toggle[data-val="none"]');
        if (noneChip) setChipSelected(noneChip, true);
    }
}

function populateRerankerLlmProviderSelect() {
    const select = document.getElementById("reranker-llm-provider-select");
    if (!select || !catalogData?.providers) return;
    select.innerHTML = "";
    catalogData.providers.forEach(provider => {
        const option = document.createElement("option");
        option.value = provider.name;
        option.textContent = provider.available
            ? provider.name
            : `${provider.name} (${unavailableLabel(provider)})`;
        option.disabled = !provider.available;
        if (provider.name === "openai") option.selected = true;
        select.appendChild(option);
    });
}

function populateRerankerModelSuggestions() {
    const datalist = document.getElementById("reranking-model-suggestions");
    if (!datalist || !catalogData?.reranker_models) return;
    datalist.innerHTML = "";
    catalogData.reranker_models.forEach(model => {
        const option = document.createElement("option");
        option.value = model;
        datalist.appendChild(option);
    });
}

function selectedRerankers() {
    return new Set(
        Array.from(document.querySelectorAll(
            '[data-stage-group="reranking"] .chip-toggle[data-selected="true"]'
        )).map(chip => chip.getAttribute("data-val"))
    );
}

function renderRerankerConfiguration() {
    const selected = selectedRerankers();
    setElementVisible(
        "reranking-model-fields",
        selected.has("cross_encoder") || selected.has("pointwise"),
    );
    setElementVisible("reranker-llm-card", selected.has("llm"));
    setElementVisible(
        "reranker-llm-fields",
        selected.has("llm")
            && Boolean(document.getElementById("reranker-llm-override-checkbox")?.checked),
    );
    setElementVisible("custom-reranker-fields", selected.has("custom"));
}

function populateChunkingMethodSelect() {
    const select = document.getElementById("chunking-method-select");
    if (!select || !catalogData.chunking_methods) return;
    select.innerHTML = "";
    catalogData.chunking_methods.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        opt.textContent = m;
        select.appendChild(opt);
    });
}

function populateChunkingQuickfill() {
    const container = document.getElementById("chunking-quickfill");
    if (!container || !catalogData.chunking_presets) return;
    container.innerHTML = "";
    catalogData.chunking_presets.forEach(preset => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.className = "px-space-sm py-space-xs rounded bg-surface-container-low border border-outline-variant/40 font-code-sm text-[11px] text-outline hover:border-primary/50 hover:text-on-surface transition-all";
        btn.textContent = preset.label;
        btn.addEventListener("click", () => {
            document.getElementById("chunking-method-select").value = preset.value.method;
            document.getElementById("chunking-size-input").value = preset.value.size;
            document.getElementById("chunking-overlap-input").value = preset.value.overlap;
        });
        container.appendChild(btn);
    });
}

function populateEmbeddingSuggestions() {
    const datalist = document.getElementById("embedding-suggestions");
    if (!datalist || !catalogData.embedding_models) return;
    datalist.innerHTML = "";
    catalogData.embedding_models.forEach(m => {
        const opt = document.createElement("option");
        opt.value = m;
        datalist.appendChild(opt);
    });
}

function applyCapabilityAvailability() {
    const capabilities = catalogData?.capabilities || {};
    const disableChip = (stage, value, capability) => {
        const status = capabilities[capability];
        if (!status || status.available) return;
        const chip = document.querySelector(`[data-stage-group="${stage}"] .chip-toggle[data-val="${value}"]`);
        if (!chip) return;
        setChipSelected(chip, false);
        chip.disabled = true;
        chip.classList.add("opacity-40", "cursor-not-allowed");
        chip.title = unavailableLabel(status);
    };

    disableChip("retrieval", "hybrid", "bm25");
    disableChip("reranking", "bm25", "bm25");
    ["semantic_similarity", "cross_encoder", "pointwise"].forEach(value => {
        disableChip("reranking", value, "local");
    });

    const tokenStatus = capabilities.token;
    const tokenOption = document.querySelector('#chunking-method-select option[value="token"]');
    if (tokenOption && tokenStatus && !tokenStatus.available) {
        tokenOption.disabled = true;
        tokenOption.textContent = `token (${unavailableLabel(tokenStatus)})`;
    }

    const localStatus = capabilities.local;
    if (localStatus && !localStatus.available) {
        const input = document.getElementById("embedding-custom-input");
        const button = document.getElementById("btn-add-embedding");
        [input, button].filter(Boolean).forEach(element => {
            element.disabled = true;
            element.title = unavailableLabel(localStatus);
            element.classList.add("opacity-40", "cursor-not-allowed");
        });
        if (input) input.placeholder = unavailableLabel(localStatus);
    }
}

function populateParserSelect() {
    const select = document.getElementById("parser-select");
    if (!select || !catalogData.parsers) return;
    while (select.options.length > 1) select.remove(1);
    catalogData.parsers.forEach(p => {
        const opt = document.createElement("option");
        opt.value = p.name;
        opt.textContent = p.available ? p.name : `${p.name} (${unavailableLabel(p)})`;
        opt.disabled = !p.available;
        select.appendChild(opt);
    });
    renderParserFields("");
}

function renderParserFields(parserName) {
    const container = document.getElementById("parser-fields");
    if (!container) return;
    container.innerHTML = "";
    if (!parserName || !catalogData || !catalogData.parsers) return;
    const parser = catalogData.parsers.find(p => p.name === parserName);
    if (!parser) return;

    parser.fields.forEach(fieldName => {
        const meta = PARSER_FIELD_LABELS[fieldName] || { label: fieldName, type: "text" };
        const wrap = document.createElement("div");
        wrap.className = "flex flex-col gap-1";
        const label = document.createElement("label");
        label.className = "font-code-sm text-[11px] text-outline";
        label.textContent = meta.label;
        label.setAttribute("for", `parser-field-${fieldName}`);
        const input = document.createElement("input");
        input.className = "bg-surface-container-low border border-outline-variant/50 rounded px-space-sm py-1 font-code-sm text-[12px] text-on-surface focus:outline-none focus:border-primary";
        input.id = `parser-field-${fieldName}`;
        input.type = meta.type;
        wrap.appendChild(label);
        wrap.appendChild(input);
        container.appendChild(wrap);
    });
}

function populateMetricsCheckboxes() {
    const retrievalGroup = document.getElementById("metrics-retrieval-group");
    const generationGroup = document.getElementById("metrics-generation-group");
    if (!catalogData || !catalogData.metrics) return;

    const defaults = new Set(catalogData.metrics.default || []);

    function makeMetricChip(name) {
        const isChecked = defaults.has(name);
        const label = document.createElement("label");
        label.className = `chip-toggle metric-chip flex items-center gap-space-xs px-space-sm py-space-xs rounded font-code-sm text-code-sm transition-all cursor-pointer select-none ${
            isChecked
                ? "active bg-surface-container-high border border-primary text-primary shadow-[0_0_8px_rgba(88,166,255,0.25)]"
                : "bg-surface-container-low border border-outline-variant/40 text-outline hover:border-primary/50"
        }`;

        const input = document.createElement("input");
        input.type = "checkbox";
        input.className = "metric-checkbox hidden";
        input.value = name;
        input.checked = isChecked;

        const box = document.createElement("span");
        box.className = `chip-checkbox w-3.5 h-3.5 rounded flex items-center justify-center text-[10px] ${
            isChecked ? "bg-primary text-on-primary font-bold" : "border border-outline-variant"
        }`;
        box.textContent = isChecked ? "✓" : "";

        const text = document.createElement("span");
        text.textContent = metricDisplayName(name);
        if (name === "llm_judge_rating") {
            label.title = "Semantic correctness against the reference answer, rated from 1 to 5.";
        }

        label.appendChild(input);
        label.appendChild(box);
        label.appendChild(text);

        input.addEventListener("change", () => {
            if (input.checked) {
                label.classList.add("active", "bg-surface-container-high", "border-primary", "text-primary", "shadow-[0_0_8px_rgba(88,166,255,0.25)]");
                label.classList.remove("bg-surface-container-low", "text-outline");
                box.classList.add("bg-primary", "text-on-primary", "font-bold");
                box.classList.remove("border", "border-outline-variant");
                box.textContent = "✓";
            } else {
                label.classList.remove("active", "bg-surface-container-high", "border-primary", "text-primary", "shadow-[0_0_8px_rgba(88,166,255,0.25)]");
                label.classList.add("bg-surface-container-low", "text-outline");
                box.classList.remove("bg-primary", "text-on-primary", "font-bold");
                box.classList.add("border", "border-outline-variant");
                box.textContent = "";
            }
            updateMetricsCounters();
        });

        return label;
    }

    if (retrievalGroup) {
        retrievalGroup.innerHTML = "";
        (catalogData.metrics.retrieval || []).forEach(m => retrievalGroup.appendChild(makeMetricChip(m)));
    }
    if (generationGroup) {
        generationGroup.innerHTML = "";
        (catalogData.metrics.generation || []).forEach(m => generationGroup.appendChild(makeMetricChip(m)));
    }

    updateMetricsCounters();
}

function updateMetricsCounters() {
    const rChecked = document.querySelectorAll("#metrics-retrieval-group .metric-checkbox:checked").length;
    const gChecked = document.querySelectorAll("#metrics-generation-group .metric-checkbox:checked").length;
    const total = rChecked + gChecked;

    const rCounter = document.getElementById("retrieval-metrics-counter");
    if (rCounter) rCounter.textContent = `${rChecked} active`;

    const gCounter = document.getElementById("generation-metrics-counter");
    if (gCounter) gCounter.textContent = `${gChecked} active`;

    const totalDisplay = document.getElementById("eval-metrics-total-display");
    if (totalDisplay) totalDisplay.textContent = `${total} active`;
}

function initChipHandlers() {
    document.querySelectorAll(".chip-toggle").forEach(chip => {
        if (chip.disabled) return;
        chip.addEventListener("click", () => {
            const isSelected = chip.getAttribute("data-selected") === "true";
            setChipSelected(chip, !isSelected);
            updateComboCount();
        });
    });

    const resetBtn = document.getElementById("btn-reset");
    if (resetBtn) {
        resetBtn.addEventListener("click", resetToDefaults);
    }
}

function setChipSelected(chip, selected) {
    chip.setAttribute("data-selected", selected ? "true" : "false");
    const checkbox = chip.querySelector(".chip-checkbox");

    if (selected) {
        chip.classList.add("active", "bg-surface-container-high", "border-primary", "text-primary", "shadow-[0_0_8px_rgba(88,166,255,0.25)]");
        chip.classList.remove("bg-surface-container-low", "text-outline");
        if (checkbox) {
            checkbox.classList.add("bg-primary", "text-on-primary");
            checkbox.classList.remove("border-outline-variant");
            checkbox.textContent = "✓";
        }
    } else {
        chip.classList.remove("active", "bg-surface-container-high", "border-primary", "text-primary", "shadow-[0_0_8px_rgba(88,166,255,0.25)]");
        chip.classList.add("bg-surface-container-low", "text-outline");
        if (checkbox) {
            checkbox.classList.remove("bg-primary", "text-on-primary");
            checkbox.classList.add("border-outline-variant");
            checkbox.textContent = "";
        }
    }
}

function initCustomAddHandlers() {
    const addKBtn = document.getElementById("btn-add-k");
    if (addKBtn) {
        addKBtn.addEventListener("click", () => {
            const input = document.getElementById("k-custom-input");
            const val = parseInt(input.value, 10);
            if (!val || val < 1) {
                showToast("Enter a valid positive integer for k", "error");
                return;
            }
            const group = document.querySelector('[data-stage-group="k"]');
            if (group.querySelector(`.chip-toggle[data-val="${val}"]`)) {
                showToast(`k: ${val} is already in the list`, "warning");
                return;
            }
            group.appendChild(makeRemovableChip(String(val), `k: ${val}`));
            input.value = "";
            updateComboCount();
        });
    }

    const addChunkingBtn = document.getElementById("btn-add-chunking");
    if (addChunkingBtn) {
        addChunkingBtn.addEventListener("click", () => {
            const method = document.getElementById("chunking-method-select").value;
            const size = parseInt(document.getElementById("chunking-size-input").value, 10);
            const overlap = parseInt(document.getElementById("chunking-overlap-input").value, 10);
            if (!method || !size || isNaN(overlap) || size < 1 || overlap < 0) {
                showToast("Enter valid chunking method/size/overlap values", "error");
                return;
            }
            const value = { method, size, overlap };
            const encoded = JSON.stringify(value);
            const group = document.querySelector('[data-stage-group="chunking"]');
            if (Array.from(group.children).some(c => c.getAttribute("data-val") === encoded)) {
                showToast("That chunking combination is already in the list", "warning");
                return;
            }
            group.appendChild(makeRemovableChip(encoded, `${method} ${size}/${overlap}`));
            updateComboCount();
        });
    }

    const addEmbeddingBtn = document.getElementById("btn-add-embedding");
    if (addEmbeddingBtn) {
        addEmbeddingBtn.addEventListener("click", () => {
            const input = document.getElementById("embedding-custom-input");
            const val = input.value.trim();
            if (!val) {
                showToast("Enter a HuggingFace model id", "error");
                return;
            }
            const group = document.querySelector('[data-stage-group="embedding_model"]');
            if (Array.from(group.children).some(c => c.getAttribute("data-val") === val)) {
                showToast("That embedding model is already in the list", "warning");
                return;
            }
            group.appendChild(makeRemovableChip(val, val));
            input.value = "";
            updateComboCount();
        });
    }

    const addRerankingModelBtn = document.getElementById("btn-add-reranking-model");
    if (addRerankingModelBtn) {
        addRerankingModelBtn.addEventListener("click", () => {
            const input = document.getElementById("reranking-model-input");
            const value = input?.value.trim() || "";
            if (!value) {
                showToast("Enter a Hugging Face reranker model id", "error");
                return;
            }
            const group = document.querySelector('[data-stage-group="reranking_model"]');
            if (Array.from(group.children).some(chip => chip.getAttribute("data-val") === value)) {
                showToast("That reranker model is already in the list", "warning");
                return;
            }
            group.appendChild(makeRemovableChip(value, value));
            input.value = "";
            updateComboCount();
        });
    }
}

function initParserHandlers() {
    const select = document.getElementById("parser-select");
    if (select) {
        select.addEventListener("change", () => renderParserFields(select.value));
    }
}

function initJudgeLlmHandlers() {
    const checkbox = document.getElementById("judge-llm-override-checkbox");
    const fields = document.getElementById("judge-llm-fields");
    if (checkbox && fields) {
        checkbox.addEventListener("change", () => {
            fields.hidden = !checkbox.checked;
        });
    }
}

function initTaskLlmHandlers() {
    ["dataset-llm-reuse-checkbox", "query-transform-override-checkbox"].forEach(id => {
        const control = document.getElementById(id);
        if (control) control.addEventListener("change", renderWorkflow);
    });
}

function initRetrievalSourceHandlers() {
    const sourceRadios = document.querySelectorAll('input[name="retrieval-knowledge-source"]');
    const searchProviderSelect = document.getElementById("search-provider-select");

    function refresh() {
        const source = document.querySelector('input[name="retrieval-knowledge-source"]:checked')?.value || "documents";
        document.querySelectorAll(".retrieval-source-tile").forEach(tile => {
            const isCurrent = tile.getAttribute("data-source") === source;
            if (isCurrent) {
                tile.classList.add("bg-surface-container-high", "border-primary", "text-primary");
                tile.classList.remove("bg-surface-container-low", "border-outline-variant/40", "text-outline");
            } else {
                tile.classList.remove("bg-surface-container-high", "border-primary", "text-primary");
                tile.classList.add("bg-surface-container-low", "border-outline-variant/40", "text-outline");
            }
        });
    }

    sourceRadios.forEach(radio => radio.addEventListener("change", refresh));
    if (searchProviderSelect) {
        searchProviderSelect.addEventListener("change", () => renderSearchProviderExtraFields(searchProviderSelect.value));
    }
    refresh();
}

function initPipelineModeHandlers() {
    const radios = document.querySelectorAll('input[name="pipeline-mode"]');
    radios.forEach(r => r.addEventListener("change", refreshPipelineModePanel));
    refreshPipelineModePanel();
}

// Also called from updateComboCount() -- the query-transform LLM fields'
// visibility depends on whether a non-"none" query_expansion chip is
// currently active, and chip toggling doesn't otherwise know about this panel.
function refreshPipelineModePanel() {
    const state = deriveWorkflowState();

    document.querySelectorAll(".pipeline-mode-tile").forEach(tile => {
        const isCurrent = tile.getAttribute("data-mode") === state.pipelineMode;
        if (isCurrent) {
            tile.classList.add("bg-surface-container-high", "border-primary", "text-primary");
            tile.classList.remove("bg-surface-container-low", "border-outline-variant/40", "text-outline");
        } else {
            tile.classList.remove("bg-surface-container-high", "border-primary", "text-primary");
            tile.classList.add("bg-surface-container-low", "border-outline-variant/40", "text-outline");
        }
    });

    const generationMetricsCard = document.getElementById("generation-metrics-card");
    if (generationMetricsCard) generationMetricsCard.hidden = state.retrievalOnly;

    // Hiding the card isn't enough: collectSelectedMetrics() reads
    // ".metric-checkbox:checked" regardless of visibility, so a still-checked
    // A checked generation-metric chip would be submitted and rejected
    // server-side. Uncheck them (dispatching "change" so the chip visuals and
    // the counters stay in sync with their own handlers).
    if (state.retrievalOnly) {
        document.querySelectorAll("#metrics-generation-group .metric-checkbox:checked").forEach(cb => {
            cb.checked = false;
            cb.dispatchEvent(new Event("change"));
        });
    }

    renderQueryTransformControls(state);

    // Synthetic eval-dataset generation always needs an LLM, whatever the
    // pipeline mode -- warn when retrieval-only is combined with "auto".
    refreshRetrievalOnlyDatasetNotice();
}

function renderQueryTransformControls(state = deriveWorkflowState()) {
    const card = document.getElementById("query-transform-provider-card");
    const override = document.getElementById("query-transform-override-checkbox");
    const control = document.getElementById("query-transform-override-control");
    const fields = document.getElementById("query-transform-llm-fields");

    if (card) card.hidden = !state.queryExpansionActive || state.webSearchOnly;
    if (!state.queryExpansionActive || state.webSearchOnly) {
        if (fields) fields.hidden = true;
        return;
    }

    if (state.retrievalOnly && override) override.checked = true;
    if (override) override.disabled = state.retrievalOnly;
    if (control) {
        const labelText = control.querySelector("span");
        if (labelText) {
            labelText.textContent = state.retrievalOnly
                ? "Required in Retrieval-Only mode"
                : "Use a different LLM for query transforms";
        }
    }
    if (fields) fields.hidden = !(state.retrievalOnly || Boolean(override?.checked));
}

// Informational only: retrieval-only mode never calls a main LLM for the
// pipeline itself, but eval_dataset_mode="auto" still generates synthetic Q&A
// pairs with one. Driven from both refreshPipelineModePanel() and the
// dataset-mode refresh() below so either trigger keeps it accurate.
function refreshRetrievalOnlyDatasetNotice() {
    const notice = document.getElementById("retrieval-only-auto-dataset-notice");
    if (!notice) return;
    const mode = document.querySelector('input[name="pipeline-mode"]:checked')?.value || "full_rag";
    const datasetMode = document.querySelector('input[name="eval-dataset-mode"]:checked')?.value || "auto";
    notice.hidden = !(mode === "retrieval_only" && datasetMode === "auto");
}

function initDatasetModeHandlers() {
    const radios = document.querySelectorAll('input[name="eval-dataset-mode"]');
    const existingFields = document.getElementById("dataset-existing-fields");
    const splitCheckbox = document.getElementById("train-test-split-checkbox");
    const testSizeInput = document.getElementById("test-size-input");

    function refresh() {
        const mode = document.querySelector('input[name="eval-dataset-mode"]:checked')?.value || "auto";
        if (existingFields) existingFields.hidden = mode !== "existing";
        const canSplit = mode === "existing";
        if (splitCheckbox) {
            splitCheckbox.disabled = !canSplit;
            if (!canSplit) splitCheckbox.checked = false;
        }
        if (testSizeInput) {
            testSizeInput.disabled = !canSplit || !splitCheckbox.checked;
        }

        // Highlight selected dataset mode tile
        document.querySelectorAll(".dataset-mode-tile").forEach(tile => {
            const isCurrent = tile.getAttribute("data-mode") === mode;
            if (isCurrent) {
                tile.classList.add("active", "bg-surface-container-high", "border-primary", "text-primary", "shadow-[0_0_8px_rgba(88,166,255,0.15)]");
                tile.classList.remove("bg-surface-container-low", "border-outline-variant/40", "text-outline");
            } else {
                tile.classList.remove("active", "bg-surface-container-high", "border-primary", "text-primary", "shadow-[0_0_8px_rgba(88,166,255,0.15)]");
                tile.classList.add("bg-surface-container-low", "border-outline-variant/40", "text-outline");
            }
        });

        // Update Card 01 header mode badge
        const modeBadge = document.getElementById("eval-mode-badge");
        if (modeBadge) {
            modeBadge.textContent = mode === "auto" ? "auto-generate" : "existing file";
        }

        refreshRetrievalOnlyDatasetNotice();
        refreshPipelineModePanel();

        updateSplitDisplay();
    }

    function updateSplitDisplay() {
        const mode = document.querySelector('input[name="eval-dataset-mode"]:checked')?.value || "auto";
        const canSplit = mode === "existing";
        const isChecked = splitCheckbox && splitCheckbox.checked && canSplit;
        const testPct = Math.round(parseFloat(testSizeInput?.value || 0.2) * 100);
        const trainPct = 100 - testPct;

        const splitBadge = document.getElementById("eval-split-badge");
        if (splitBadge) {
            splitBadge.textContent = isChecked ? `${testPct}% test split` : "Full Set (No Split)";
        }
        const pill = document.getElementById("split-status-pill");
        if (pill) {
            if (!canSplit) {
                pill.textContent = "Disabled in Auto Mode";
                pill.className = "px-2 py-0.5 rounded font-code-sm text-[11px] bg-surface-container text-outline";
            } else if (isChecked) {
                pill.textContent = `Active (${testPct}% Test)`;
                pill.className = "px-2 py-0.5 rounded font-code-sm text-[11px] bg-primary/20 text-primary border border-primary/40 font-semibold";
            } else {
                pill.textContent = "Disabled (100% Data)";
                pill.className = "px-2 py-0.5 rounded font-code-sm text-[11px] bg-surface-container text-outline";
            }
        }
        const trainLabel = document.getElementById("split-train-label");
        if (trainLabel) trainLabel.textContent = `Reserved: ${trainPct}% (not scored)`;
        const testLabel = document.getElementById("split-test-label");
        if (testLabel) testLabel.textContent = `Test: ${testPct}% (scored)`;
        const trainBar = document.getElementById("split-train-bar");
        if (trainBar) trainBar.style.width = `${trainPct}%`;
        const testBar = document.getElementById("split-test-bar");
        if (testBar) testBar.style.width = `${testPct}%`;
    }

    radios.forEach(r => r.addEventListener("change", refresh));
    if (splitCheckbox) {
        splitCheckbox.addEventListener("change", () => {
            if (testSizeInput) testSizeInput.disabled = !splitCheckbox.checked;
            updateSplitDisplay();
        });
    }
    if (testSizeInput) {
        testSizeInput.addEventListener("input", updateSplitDisplay);
    }
    refresh();
}

function collectParserFieldConfig() {
    const parserName = document.getElementById("parser-select")?.value;
    if (!parserName || !catalogData) return {};
    const parser = catalogData.parsers.find(p => p.name === parserName);
    if (!parser) return {};
    const config = {};
    parser.fields.forEach(fieldName => {
        const el = document.getElementById(`parser-field-${fieldName}`);
        if (el && el.value.trim()) config[fieldName] = el.value.trim();
    });
    return config;
}

function updateComboCount() {
    LLMSettings.refresh();
    let total = 1;
    let anyStageActive = LLMSettings.variants().length > 0;
    const webSearchOnly = deriveWorkflowState().webSearchOnly;

    const stages = ["query_expansion", "retrieval", "reranking", "reranking_model", "k", "chunking", "embedding_model", "vector_db_provider"];

    stages.forEach(stage => {
        const group = document.querySelector(`[data-stage-group="${stage}"]`);
        const counter = document.querySelector(`.stage-counter[data-stage="${stage}"]`);
        if (!group) return;

        const activeChips = group.querySelectorAll('.chip-toggle[data-selected="true"]');
        const count = activeChips.length;

        if (counter) {
            counter.textContent = `${count} active`;
        }

        if (count > 0) anyStageActive = true;
    });

    if (webSearchOnly) {
        // Local architecture dimensions are ignored by MuffakirSearch. Submit
        // one inert combination instead of running duplicate web-only trials.
        total = Math.max(1, LLMSettings.variants().length);
    } else if (!anyStageActive) {
        total = 0;
    } else {
        total = buildSearchSpaceCombinations(serializeSearchSpace()).length;
    }

    const comboDisplay = document.getElementById("combo-total-display");
    if (comboDisplay) comboDisplay.textContent = total.toLocaleString();

    const tabBadge = document.getElementById("tab-combo-badge");
    if (tabBadge) tabBadge.textContent = `${total.toLocaleString()} combo${total === 1 ? '' : 's'}`;

    const comboExplanation = document.getElementById("combo-explanation");
    if (comboExplanation) {
        comboExplanation.textContent = webSearchOnly
            ? "One web-backed trial per generation variant; local architecture stages do not apply"
            : "Combinations are computed across all active stages below";
    }

    renderRerankerConfiguration();
    refreshPipelineModePanel();
    renderCustomPricingModels();
    renderConfigurationSummary();
    renderDryRunGate();
}

function serializeSearchSpace() {
    const variants = LLMSettings.variants();
    if (deriveWorkflowState().webSearchOnly) {
        return {
            ...(variants.length ? { llm: variants } : {}),
            query_expansion: ["none"],
            retrieval: ["similarity_search"],
            reranking: ["none"],
            k: [3],
        };
    }

    const space = variants.length ? { llm: variants } : {};

    // Flat stages (plain string values, custom or preset chips serialize the same way)
    const flatStages = ["query_expansion", "retrieval", "reranking", "embedding_model", "vector_db_provider"];
    flatStages.forEach(stage => {
        const group = document.querySelector(`[data-stage-group="${stage}"]`);
        if (!group) return;
        const selected = Array.from(group.querySelectorAll('.chip-toggle[data-selected="true"]')).map(c => c.getAttribute("data-val"));
        if (selected.length > 0) space[stage] = selected;
    });

    // Local Hugging Face reranker models are conditional on CrossEncoder-
    // based methods and must not multiply unrelated reranker trials.
    const localRerankerSelected = (space.reranking || []).some(
        method => ["cross_encoder", "pointwise"].includes(String(method).toLowerCase())
    );
    if (localRerankerSelected) {
        const modelGroup = document.querySelector('[data-stage-group="reranking_model"]');
        const models = Array.from(modelGroup?.querySelectorAll('.chip-toggle[data-selected="true"]') || [])
            .map(chip => chip.getAttribute("data-val")?.trim())
            .filter(Boolean);
        if (models.length > 0) space.reranking_model = models;
    }

    // K stage (integers, preset + custom chips share the same data-val shape)
    const kGroup = document.querySelector('[data-stage-group="k"]');
    if (kGroup) {
        const selectedK = Array.from(kGroup.querySelectorAll('.chip-toggle[data-selected="true"]'))
            .map(c => parseInt(c.getAttribute("data-val"), 10))
            .filter(n => !isNaN(n));
        if (selectedK.length > 0) space.k = selectedK;
    }

    // Chunking stage: every chip is a custom-added one carrying its
    // {method,size,overlap} value JSON-encoded in data-val.
    const chunkingGroup = document.querySelector('[data-stage-group="chunking"]');
    if (chunkingGroup) {
        const selected = Array.from(chunkingGroup.querySelectorAll('.chip-toggle[data-selected="true"]'))
            .map(c => {
                try { return JSON.parse(c.getAttribute("data-val")); } catch (_) { return null; }
            })
            .filter(Boolean);
        if (selected.length > 0) space.chunking = selected;
    }

    return space;
}

function collectSelectedMetrics() {
    return Array.from(document.querySelectorAll(".metric-checkbox:checked")).map(c => c.value);
}

function getPromptLanguage() {
    return document.getElementById("prompt-language-select")?.value || "ar";
}

function buildPromptContext(language = getPromptLanguage()) {
    const state = deriveWorkflowState();
    return {
        language,
        pipeline_mode: state.pipelineMode,
        retrieval_source: state.webSearchOnly ? "web_search_only" : "vector_db",
        adaptive_web_search: state.adaptiveWebSearch,
        eval_dataset_mode: state.datasetMode,
        search_space: serializeSearchSpace(),
        metrics: collectSelectedMetrics(),
        hallucination_check: true,
        hallucination_method: "text_cleaner",
    };
}

function locallyApplicablePromptKeys() {
    const context = buildPromptContext();
    const keys = new Set();
    if (context.eval_dataset_mode === "auto") keys.add("QA");
    if (context.retrieval_source !== "web_search_only") {
        const transformMap = {
            rewrite: "query_rewrite",
            multi_query: "multi_query_expansion",
            multi_query_expansion: "multi_query_expansion",
            decomposition: "query_decomposition",
            query_decomposition: "query_decomposition",
            hyde: "hyde",
            step_back: "step_back",
        };
        (context.search_space.query_expansion || []).forEach(value => {
            if (transformMap[value]) keys.add(transformMap[value]);
        });
        const llmRerankers = new Set(["llm", "llm_reranker", "llm-based", "llm_based"]);
        if ((context.search_space.reranking || []).some(value => llmRerankers.has(String(value).toLowerCase()))) {
            keys.add("reranker_scoring");
        }
    }
    if (context.pipeline_mode === "full_rag") {
        keys.add("generation");
        if (context.retrieval_source !== "web_search_only") {
            keys.add("hallucination_check_prompt");
            if (context.adaptive_web_search) keys.add("context_relevance");
        }
        if (context.metrics.includes("answer_correctness")) keys.add("answer_correctness");
        if (context.metrics.includes("llm_judge_rating")) keys.add("llm_judge_rating");
        if (context.metrics.includes("faithfulness")) keys.add("context_grounding");
    }
    return keys;
}

function saveCurrentPromptDrafts() {
    const language = renderedPromptLanguage;
    document.querySelectorAll("[data-prompt-card]").forEach(card => {
        const key = card.dataset.promptCard;
        const textarea = card.querySelector("textarea[data-prompt-template]");
        if (!key || !textarea) return;
        promptDraftsByLanguage[language][key] = {
            custom: card.dataset.custom === "true",
            value: textarea.value,
            defaultTemplate: card.dataset.defaultTemplate || "",
        };
    });
}

function collectPromptOverrides() {
    saveCurrentPromptDrafts();
    const language = getPromptLanguage();
    const applicable = new Set(promptDefinitions.map(item => item.key));
    const result = {};
    Object.entries(promptDraftsByLanguage[language]).forEach(([key, draft]) => {
        if (applicable.has(key) && draft.custom) result[key] = draft.value;
    });
    return result;
}

function validatePromptTemplate(definition, template) {
    if (!template || !template.trim()) return "Prompt cannot be empty.";
    const found = new Set();
    for (let index = 0; index < template.length;) {
        const char = template[index];
        if (char === "{") {
            if (template[index + 1] === "{") { index += 2; continue; }
            const end = template.indexOf("}", index + 1);
            if (end < 0) return "Opening brace has no matching closing brace.";
            const field = template.slice(index + 1, end);
            if (!field) return "Empty placeholders are not allowed.";
            if (field.includes(":") || field.includes("!")) return "Format specifications and conversions are not allowed.";
            if (!definition.allowed_variables.includes(field)) return `Unknown variable {${field}}.`;
            found.add(field);
            index = end + 1;
            continue;
        }
        if (char === "}") {
            if (template[index + 1] === "}") { index += 2; continue; }
            return "Closing brace has no matching opening brace.";
        }
        index += 1;
    }
    const missing = definition.required_variables.filter(name => !found.has(name));
    return missing.length ? `Missing required variables: ${missing.map(name => `{${name}}`).join(", ")}.` : null;
}

function updatePromptCardValidation(card) {
    const definition = promptDefinitions.find(item => item.key === card.dataset.promptCard);
    const textarea = card.querySelector("textarea[data-prompt-template]");
    const status = card.querySelector("[data-prompt-validation]");
    if (!definition || !textarea || !status) return true;
    const custom = card.dataset.custom === "true";
    const error = custom ? validatePromptTemplate(definition, textarea.value) : null;
    status.textContent = error || (custom ? "Valid custom prompt" : "Using packaged default");
    status.className = `font-code-sm text-[10px] ${error ? "text-error" : "text-tertiary"}`;
    textarea.classList.toggle("border-error", Boolean(error));
    return !error;
}

function renderPromptCards() {
    const container = document.getElementById("prompt-manager-cards");
    const empty = document.getElementById("prompt-manager-empty");
    if (!container || !empty) return;
    container.innerHTML = "";
    empty.classList.toggle("hidden", promptDefinitions.length > 0);
    const language = getPromptLanguage();
    renderedPromptLanguage = language;

    promptDefinitions.forEach(definition => {
        const prior = promptDraftsByLanguage[language][definition.key];
        const draft = prior || {
            custom: false,
            value: definition.default_template,
            defaultTemplate: definition.default_template,
        };
        draft.defaultTemplate = definition.default_template;
        if (!prior || !draft.custom) draft.value = definition.default_template;
        promptDraftsByLanguage[language][definition.key] = draft;

        const card = document.createElement("section");
        card.dataset.promptCard = definition.key;
        card.dataset.custom = draft.custom ? "true" : "false";
        card.dataset.defaultTemplate = definition.default_template;
        card.className = "bg-surface-container-low border border-outline-variant/30 rounded p-space-base min-w-0";
        card.innerHTML = `
          <div class="flex items-start justify-between gap-space-sm mb-space-sm">
            <div class="min-w-0">
              <div class="flex flex-wrap items-center gap-space-xs">
                <span class="font-headline-sm text-[13px] font-semibold text-on-surface">${escapeHtml(definition.label)}</span>
                <span data-prompt-mode class="px-1.5 py-0.5 rounded font-code-sm text-[10px] bg-surface-container text-outline">${draft.custom ? "Custom" : "Default"}</span>
              </div>
              <p class="text-body-xs text-outline mt-1">${escapeHtml(definition.description)}</p>
              <p class="font-code-sm text-[10px] text-primary mt-1">Stage: ${escapeHtml(definition.stage)}</p>
            </div>
            <button type="button" data-prompt-customize class="shrink-0 px-space-sm py-1 rounded border border-primary/30 text-primary font-code-sm text-[10px] hover:bg-primary/10">${draft.custom ? "Restore default" : "Customize"}</button>
          </div>
          <div class="flex flex-wrap gap-1 mb-space-xs" data-prompt-variables></div>
          <textarea data-prompt-template rows="10" spellcheck="false" dir="auto" class="w-full resize-y bg-surface-container border border-outline-variant/40 rounded p-space-sm font-code-sm text-[11px] text-on-surface focus:outline-none focus:border-primary disabled:opacity-65 disabled:cursor-not-allowed"></textarea>
          <div class="flex items-center justify-between gap-space-xs mt-space-xs">
            <span data-prompt-validation></span>
            <span class="font-code-sm text-[10px] text-outline">${escapeHtml(definition.key)}</span>
          </div>`;
        const textarea = card.querySelector("textarea[data-prompt-template]");
        textarea.value = draft.value;
        textarea.disabled = !draft.custom;
        const variableBox = card.querySelector("[data-prompt-variables]");
        definition.required_variables.forEach(variable => {
            const button = document.createElement("button");
            button.type = "button";
            button.dataset.insertVariable = variable;
            button.disabled = !draft.custom;
            button.className = "px-1.5 py-0.5 rounded bg-primary/10 border border-primary/25 text-primary font-code-sm text-[10px] disabled:opacity-40";
            button.textContent = `{${variable}}`;
            variableBox.appendChild(button);
        });
        container.appendChild(card);
        updatePromptCardValidation(card);
    });
}

async function loadApplicablePrompts(force = false) {
    saveCurrentPromptDrafts();
    const context = buildPromptContext();
    const signature = JSON.stringify(context);
    if (!force && signature === promptResolutionSignature && !promptResolutionError) return;
    const sequence = ++promptResolveSequence;
    const loading = document.getElementById("prompt-manager-loading");
    if (loading) loading.classList.remove("hidden");
    promptResolutionError = null;
    try {
        const response = await apiPost("/api/prompts/resolve", context);
        if (sequence !== promptResolveSequence) return;
        promptDefinitions = response.prompts || [];
        promptResolutionSignature = signature;
        renderPromptCards();
        const badge = document.getElementById("prompt-tab-status");
        if (badge) badge.textContent = promptDefinitions.length
            ? `${promptDefinitions.length} prompts`
            : "Skipped";
    } catch (error) {
        if (sequence !== promptResolveSequence) return;
        promptResolutionError = error.message;
        promptDefinitions = [];
        renderPromptCards();
        showToast(`Could not resolve prompts: ${error.message}`, "error");
    } finally {
        if (sequence === promptResolveSequence && loading) loading.classList.add("hidden");
    }
}

function initPromptManagerHandlers() {
    const languageSelect = document.getElementById("prompt-language-select");
    languageSelect?.addEventListener("change", () => {
        promptResolutionSignature = null;
        loadApplicablePrompts(true);
        invalidateDryRun();
    });
    document.getElementById("prompt-manager-cards")?.addEventListener("click", event => {
        const card = event.target.closest("[data-prompt-card]");
        if (!card) return;
        const textarea = card.querySelector("textarea[data-prompt-template]");
        const customize = event.target.closest("[data-prompt-customize]");
        if (customize) {
            const custom = card.dataset.custom !== "true";
            card.dataset.custom = custom ? "true" : "false";
            textarea.disabled = !custom;
            if (!custom) textarea.value = card.dataset.defaultTemplate || "";
            customize.textContent = custom ? "Restore default" : "Customize";
            card.querySelector("[data-prompt-mode]").textContent = custom ? "Custom" : "Default";
            card.querySelectorAll("[data-insert-variable]").forEach(button => button.disabled = !custom);
            if (custom) textarea.focus();
            saveCurrentPromptDrafts();
            updatePromptCardValidation(card);
            renderConfigurationSummary();
            invalidateDryRun();
            return;
        }
        const variableButton = event.target.closest("[data-insert-variable]");
        if (variableButton && !textarea.disabled) {
            const value = `{${variableButton.dataset.insertVariable}}`;
            const start = textarea.selectionStart;
            textarea.setRangeText(value, start, textarea.selectionEnd, "end");
            textarea.dispatchEvent(new Event("input", { bubbles: true }));
        }
    });
    document.getElementById("prompt-manager-cards")?.addEventListener("input", event => {
        const card = event.target.closest("[data-prompt-card]");
        if (!card) return;
        saveCurrentPromptDrafts();
        updatePromptCardValidation(card);
    });
}

function pricingModelKey(provider, model) {
    return `${encodeURIComponent(String(provider || "").trim().toLowerCase())}::${encodeURIComponent(String(model || "").trim())}`;
}

function collectDetectedPricingModels() {
    const state = deriveWorkflowState();
    const searchSpace = serializeSearchSpace();
    const models = new Map();
    const add = (provider, model, role) => {
        const normalizedProvider = String(provider || "").trim().toLowerCase();
        const normalizedModel = String(model || "").trim();
        if (!normalizedProvider || !normalizedModel) return;
        const key = pricingModelKey(normalizedProvider, normalizedModel);
        if (!models.has(key)) models.set(key, { provider: normalizedProvider, model: normalizedModel, roles: new Set() });
        models.get(key).roles.add(role);
    };

    let answerModels = [];
    if (state.needsAnswerLlm) {
        const llmSearchValues = Array.isArray(searchSpace.llm) ? searchSpace.llm : [];
        answerModels = llmSearchValues.length
            ? llmSearchValues.map(item => ({ provider: item.provider, model: item.model }))
            : [{
                provider: document.getElementById("llm-provider-select")?.value,
                model: document.getElementById("llm-model-input")?.value,
            }];
        answerModels.forEach(item => add(item.provider, item.model, "Answer"));
    }

    if (state.queryExpansionActive) {
        const queryProvider = state.showQueryTransformFields
            ? document.getElementById("query-transform-llm-provider-select")?.value
            : null;
        const queryModel = state.showQueryTransformFields
            ? document.getElementById("query-transform-llm-model-input")?.value
            : null;
        const queryModels = queryProvider && queryModel
            ? [{ provider: queryProvider, model: queryModel }]
            : answerModels;
        queryModels.forEach(item => add(item.provider, item.model, "Query Transform"));
    }

    const llmReranking = (searchSpace.reranking || []).some(value =>
        ["llm", "llm_reranker", "llm-based", "llm_based"].includes(String(value).toLowerCase())
    );
    if (llmReranking) {
        const rerankerModels = state.retrievalOnly
            ? [{
                provider: document.getElementById("query-transform-llm-provider-select")?.value,
                model: document.getElementById("query-transform-llm-model-input")?.value,
            }]
            : answerModels;
        rerankerModels.forEach(item => add(item.provider, item.model, "LLM Reranker"));
    }

    const generationJudgeEnabled = state.allowGenerationMetrics && collectSelectedMetrics().some(metric =>
        ["faithfulness", "answer_correctness", "llm_judge_rating"].includes(String(metric).toLowerCase())
    );
    if (generationJudgeEnabled) {
        const useOverride = document.getElementById("judge-llm-override-checkbox")?.checked;
        add(
            useOverride ? document.getElementById("judge-llm-provider-select")?.value : document.getElementById("llm-provider-select")?.value,
            useOverride ? document.getElementById("judge-llm-model-input")?.value : document.getElementById("llm-model-input")?.value,
            "Judge",
        );
    }

    return Array.from(models.values())
        .map(item => ({ ...item, roles: Array.from(item.roles).sort() }))
        .sort((a, b) => pricingModelKey(a.provider, a.model).localeCompare(pricingModelKey(b.provider, b.model)));
}

function captureCustomPricingDrafts() {
    document.querySelectorAll("[data-pricing-row]").forEach(row => {
        const kind = row.dataset.pricingKind;
        const draft = {
            provider: row.querySelector("[data-pricing-provider]")?.value?.trim().toLowerCase() || "",
            model: row.querySelector("[data-pricing-model]")?.value?.trim() || "",
            input: row.querySelector("[data-pricing-input]")?.value ?? "",
            output: row.querySelector("[data-pricing-output]")?.value ?? "",
            enabled: row.querySelector("[data-pricing-enabled]")?.checked !== false,
        };
        if (kind === "auto") {
            customPricingAutoDrafts[row.dataset.pricingKey] = draft;
        } else if (kind === "manual") {
            const target = customPricingManualDrafts.find(item => String(item.id) === row.dataset.pricingId);
            if (target) Object.assign(target, draft);
        }
    });
}

function pricingRateInput(attribute, value, disabled = false) {
    return `<input ${attribute} class="bg-surface-container-low border border-outline-variant/50 rounded px-space-sm py-1.5 font-code-sm text-[12px] text-on-surface text-right focus:outline-none focus:border-primary disabled:opacity-40" min="0" step="0.000001" type="number" value="${escapeHtml(value)}" ${disabled ? "disabled" : ""}/>`;
}

function renderCustomPricingModels() {
    const container = document.getElementById("custom-pricing-rows");
    if (!container) return;
    captureCustomPricingDrafts();
    const detected = collectDetectedPricingModels();
    activeDetectedPricingKeys = new Set(detected.map(item => pricingModelKey(item.provider, item.model)));

    const autoRows = detected.map(item => {
        const key = pricingModelKey(item.provider, item.model);
        const draft = customPricingAutoDrafts[key] || { input: "", output: "", enabled: false };
        customPricingAutoDrafts[key] = { ...draft, provider: item.provider, model: item.model };
        const disabled = !draft.enabled;
        return `
          <div data-pricing-row data-pricing-kind="auto" data-pricing-key="${escapeHtml(key)}" class="grid grid-cols-1 lg:grid-cols-[minmax(15rem,1.5fr)_minmax(8rem,1fr)_minmax(8rem,1fr)] gap-space-sm p-space-sm bg-surface-container-low border border-outline-variant/30 rounded">
            <div class="min-w-0">
              <label class="flex items-center gap-space-xs font-code-sm text-[11px] text-on-surface cursor-pointer">
                <input data-pricing-enabled class="w-3.5 h-3.5" type="checkbox" ${draft.enabled ? "checked" : ""}/>
                <span class="truncate">${escapeHtml(item.provider)} / ${escapeHtml(item.model)}</span>
              </label>
              <input data-pricing-provider type="hidden" value="${escapeHtml(item.provider)}"/>
              <input data-pricing-model type="hidden" value="${escapeHtml(item.model)}"/>
              <div class="flex flex-wrap gap-1 mt-1">${item.roles.map(role => `<span class="px-1.5 py-0.5 rounded bg-primary/10 text-primary font-code-sm text-[9px]">${escapeHtml(role)}</span>`).join("")}</div>
            </div>
            <label class="flex flex-col gap-1"><span class="font-code-sm text-[10px] text-outline">Input USD / 1M</span>${pricingRateInput("data-pricing-input", draft.input, disabled)}</label>
            <label class="flex flex-col gap-1"><span class="font-code-sm text-[10px] text-outline">Output USD / 1M</span>${pricingRateInput("data-pricing-output", draft.output, disabled)}</label>
            <div data-pricing-error class="hidden lg:col-span-3 text-error font-code-sm text-[10px]"></div>
          </div>`;
    }).join("");

    const manualRows = customPricingManualDrafts.map(draft => `
      <div data-pricing-row data-pricing-kind="manual" data-pricing-id="${draft.id}" class="grid grid-cols-1 lg:grid-cols-[minmax(7rem,.7fr)_minmax(12rem,1.3fr)_minmax(8rem,1fr)_minmax(8rem,1fr)_auto] gap-space-sm p-space-sm bg-surface-container-low border border-outline-variant/30 rounded">
        <label class="flex flex-col gap-1"><span class="font-code-sm text-[10px] text-outline">Provider</span><input data-pricing-provider list="pricing-provider-suggestions" class="bg-surface-container border border-outline-variant/50 rounded px-space-sm py-1.5 font-code-sm text-[12px] text-on-surface" value="${escapeHtml(draft.provider || "")}" placeholder="openai"/></label>
        <label class="flex flex-col gap-1"><span class="font-code-sm text-[10px] text-outline">Model</span><input data-pricing-model class="bg-surface-container border border-outline-variant/50 rounded px-space-sm py-1.5 font-code-sm text-[12px] text-on-surface" value="${escapeHtml(draft.model || "")}" placeholder="model identifier"/></label>
        <label class="flex flex-col gap-1"><span class="font-code-sm text-[10px] text-outline">Input USD / 1M</span>${pricingRateInput("data-pricing-input", draft.input || "")}</label>
        <label class="flex flex-col gap-1"><span class="font-code-sm text-[10px] text-outline">Output USD / 1M</span>${pricingRateInput("data-pricing-output", draft.output || "")}</label>
        <button type="button" data-remove-pricing="${draft.id}" class="self-end h-8 w-8 rounded border border-outline-variant/40 text-outline hover:text-error hover:border-error/50" title="Remove custom model"><span class="material-symbols-outlined text-[16px]">delete</span></button>
        <div data-pricing-error class="hidden lg:col-span-5 text-error font-code-sm text-[10px]"></div>
      </div>`).join("");

    container.innerHTML = autoRows + manualRows || `
      <div class="p-space-base text-center text-outline text-body-xs border border-dashed border-outline-variant/40 rounded">
        No token-priced trial LLM is currently selected. You can still add a manual provider/model override.
      </div>`;
    const suggestions = document.getElementById("pricing-provider-suggestions");
    if (suggestions) suggestions.innerHTML = (catalogData?.providers || []).map(item => `<option value="${escapeHtml(item.name)}"></option>`).join("");
    updateCustomPricingVisibility();
    validateCustomPricing(false);
}

function updateCustomPricingVisibility() {
    const enabled = document.getElementById("custom-pricing-toggle")?.checked || false;
    const controls = document.getElementById("custom-pricing-controls");
    const status = document.getElementById("custom-pricing-status");
    if (controls) controls.hidden = !enabled;
    if (status) status.textContent = enabled ? "Per-run overrides enabled" : "Optional · default catalog";
}

function parseCustomPricingRate(value) {
    if (String(value).trim() === "") return null;
    return Number(value);
}

function selectedCustomPricingRows() {
    return Array.from(document.querySelectorAll("[data-pricing-row]")).filter(row =>
        row.dataset.pricingKind === "manual" || row.querySelector("[data-pricing-enabled]")?.checked
    );
}

function validateCustomPricing(showMessage = true) {
    const enabled = document.getElementById("custom-pricing-toggle")?.checked || false;
    const rows = enabled ? selectedCustomPricingRows() : [];
    const seen = new Map();
    const rowErrors = new Map();
    rows.forEach(row => {
        const provider = row.querySelector("[data-pricing-provider]")?.value?.trim().toLowerCase() || "";
        const model = row.querySelector("[data-pricing-model]")?.value?.trim() || "";
        const inputValue = parseCustomPricingRate(row.querySelector("[data-pricing-input]")?.value ?? "");
        const outputValue = parseCustomPricingRate(row.querySelector("[data-pricing-output]")?.value ?? "");
        let error = "";
        if (!provider || !model) error = "Provider and model are required.";
        else if (inputValue === null || outputValue === null) error = "Both input and output rates are required.";
        else if (!Number.isFinite(inputValue) || !Number.isFinite(outputValue) || inputValue < 0 || outputValue < 0) error = "Rates must be finite numbers greater than or equal to zero.";
        const key = pricingModelKey(provider, model);
        if (!error && seen.has(key)) {
            error = "Duplicate provider/model override.";
            rowErrors.set(seen.get(key), error);
        }
        if (!error) seen.set(key, row);
        if (error) rowErrors.set(row, error);
    });
    document.querySelectorAll("[data-pricing-row]").forEach(row => {
        const error = rowErrors.get(row) || "";
        const errorEl = row.querySelector("[data-pricing-error]");
        if (errorEl) {
            errorEl.textContent = error;
            errorEl.classList.toggle("hidden", !error);
        }
        row.classList.toggle("border-error/60", Boolean(error));
    });
    if (showMessage && rowErrors.size) showToast(`Fix ${rowErrors.size} invalid custom pricing row${rowErrors.size === 1 ? "" : "s"}`, "error");
    return rowErrors.size === 0;
}

function collectCustomPricing() {
    captureCustomPricingDrafts();
    if (!document.getElementById("custom-pricing-toggle")?.checked) return [];
    const entries = [];
    Object.entries(customPricingAutoDrafts).forEach(([key, draft]) => {
        if (!activeDetectedPricingKeys.has(key) || !draft.enabled) return;
        entries.push(draft);
    });
    customPricingManualDrafts.forEach(draft => entries.push(draft));
    return entries.map(draft => ({
        provider: String(draft.provider || "").trim().toLowerCase(),
        model: String(draft.model || "").trim(),
        input_usd_per_million_tokens: parseCustomPricingRate(draft.input),
        output_usd_per_million_tokens: parseCustomPricingRate(draft.output),
    })).sort((a, b) => pricingModelKey(a.provider, a.model).localeCompare(pricingModelKey(b.provider, b.model)));
}

function initCustomPricingHandlers() {
    const toggle = document.getElementById("custom-pricing-toggle");
    const container = document.getElementById("custom-pricing-rows");
    toggle?.addEventListener("change", () => {
        updateCustomPricingVisibility();
        validateCustomPricing(false);
        renderConfigurationSummary();
        invalidateDryRun();
    });
    document.getElementById("btn-add-custom-pricing")?.addEventListener("click", () => {
        captureCustomPricingDrafts();
        customPricingManualSequence += 1;
        customPricingManualDrafts.push({
            id: customPricingManualSequence,
            provider: "",
            model: "",
            input: "",
            output: "",
            enabled: true,
        });
        renderCustomPricingModels();
        container?.querySelector(`[data-pricing-id="${customPricingManualSequence}"] [data-pricing-provider]`)?.focus();
        invalidateDryRun();
    });
    container?.addEventListener("click", event => {
        const remove = event.target.closest("[data-remove-pricing]");
        if (!remove) return;
        const index = customPricingManualDrafts.findIndex(item => String(item.id) === remove.dataset.removePricing);
        if (index >= 0) customPricingManualDrafts.splice(index, 1);
        renderCustomPricingModels();
        renderConfigurationSummary();
        invalidateDryRun();
    });
    container?.addEventListener("change", event => {
        if (event.target.matches("[data-pricing-enabled]")) {
            const row = event.target.closest("[data-pricing-row]");
            row?.querySelectorAll("[data-pricing-input], [data-pricing-output]").forEach(input => {
                input.disabled = !event.target.checked;
            });
        }
        captureCustomPricingDrafts();
        validateCustomPricing(false);
        renderConfigurationSummary();
        invalidateDryRun();
    });
    container?.addEventListener("input", () => {
        captureCustomPricingDrafts();
        validateCustomPricing(false);
        renderConfigurationSummary();
        invalidateDryRun();
    });
    ["llm-model-input", "query-transform-llm-model-input", "judge-llm-model-input", "reranker-llm-model-input"].forEach(id => {
        document.getElementById(id)?.addEventListener("input", renderCustomPricingModels);
    });
    renderCustomPricingModels();
}

function getFormData() {
    const workflowState = deriveWorkflowState();
    const runName = document.getElementById("run-name-input")?.value?.trim() || "rag-eval-sweep";
    const docsPath = workflowState.needsDocuments
        ? (document.getElementById("docs-path-input")?.value?.trim() || null)
        : null;
    const llmProvider = workflowState.needsAnswerLlm
        ? (document.getElementById("llm-provider-select")?.value || null)
        : null;
    const llmModel = workflowState.needsAnswerLlm
        ? (document.getElementById("llm-model-input")?.value?.trim() || null)
        : null;
    const apiKey = workflowState.needsAnswerLlm
        ? (document.getElementById("api-key-input")?.value?.trim() || null)
        : null;
    const baseUrl = workflowState.needsAnswerLlm
        ? (document.getElementById("base-url-input")?.value?.trim() || null)
        : null;
    const nJobs = parseInt(document.getElementById("n-jobs-input")?.value, 10) || 4;
    const maxSamples = parseInt(document.getElementById("max-samples-input")?.value, 10) || 50;
    const maxTrialsVal = document.getElementById("max-trials-input")?.value?.trim();
    const maxTrials = maxTrialsVal ? parseInt(maxTrialsVal, 10) : null;

    const parserName = workflowState.needsDocumentProcessing
        ? (document.getElementById("parser-select")?.value || null)
        : null;
    const useOcr = workflowState.needsDocumentProcessing
        ? (document.getElementById("use-ocr-checkbox")?.checked || false)
        : false;

    const evalDatasetMode = document.querySelector('input[name="eval-dataset-mode"]:checked')?.value || "auto";
    let evalDatasetPath = null;
    if (evalDatasetMode === "existing") {
        evalDatasetPath = document.getElementById("eval-dataset-path-input")?.value?.trim() || null;
    }

    const splitEnabled = document.getElementById("train-test-split-checkbox")?.checked || false;
    const testSize = parseFloat(document.getElementById("test-size-input")?.value) || 0.2;

    const device = workflowState.webSearchOnly
        ? null
        : (document.getElementById("device-select")?.value || null);

    const useJudgeOverride = document.getElementById("judge-llm-override-checkbox")?.checked || false;
    const judgeLlmProvider = useJudgeOverride ? (document.getElementById("judge-llm-provider-select")?.value || null) : null;
    const judgeLlmModel = useJudgeOverride ? (document.getElementById("judge-llm-model-input")?.value?.trim() || null) : null;
    const judgeApiKey = useJudgeOverride ? (document.getElementById("judge-api-key-input")?.value?.trim() || null) : null;
    const judgeBaseUrl = useJudgeOverride ? (document.getElementById("judge-base-url-input")?.value?.trim() || null) : null;

    const isWebSearchOnly = workflowState.webSearchOnly;
    const retrievalSource = isWebSearchOnly ? "web_search_only" : "vector_db";
    const adaptiveWebSearch = workflowState.adaptiveWebSearch;
    const needsSearchProvider = workflowState.showWebProvider;
    const searchProvider = needsSearchProvider ? (document.getElementById("search-provider-select")?.value || null) : null;
    const searchProviderConfig = needsSearchProvider ? collectSearchProviderFieldConfig() : null;
    const searchApiKey = needsSearchProvider ? (document.getElementById("search-api-key-input")?.value?.trim() || null) : null;

    const pipelineMode = workflowState.pipelineMode;
    const queryTransformLlmProvider = workflowState.showQueryTransformFields
        ? (document.getElementById("query-transform-llm-provider-select")?.value || null)
        : null;
    const queryTransformLlmModel = workflowState.showQueryTransformFields
        ? (document.getElementById("query-transform-llm-model-input")?.value?.trim() || null)
        : null;
    const queryTransformApiKey = workflowState.showQueryTransformFields
        ? (document.getElementById("query-transform-api-key-input")?.value?.trim() || null)
        : null;
    const queryTransformBaseUrl = workflowState.showQueryTransformFields
        ? (document.getElementById("query-transform-base-url-input")?.value?.trim() || null)
        : null;

    const datasetLlmProvider = workflowState.showDatasetLlmFields
        ? (document.getElementById("dataset-llm-provider-select")?.value || null)
        : null;
    const datasetLlmModel = workflowState.showDatasetLlmFields
        ? (document.getElementById("dataset-llm-model-input")?.value?.trim() || null)
        : null;
    const datasetApiKey = workflowState.showDatasetLlmFields
        ? (document.getElementById("dataset-api-key-input")?.value?.trim() || null)
        : null;
    const datasetBaseUrl = workflowState.showDatasetLlmFields
        ? (document.getElementById("dataset-base-url-input")?.value?.trim() || null)
        : null;

    const rerankers = selectedRerankers();
    const useRerankerLlmOverride = rerankers.has("llm")
        && (document.getElementById("reranker-llm-override-checkbox")?.checked || false);
    // The UI uses search_space.reranking_model. Keep the root field null;
    // SDK/API clients may still use it for the backward-compatible one-model path.
    const rerankingModel = null;
    const rerankerTimeout = parseFloat(
        document.getElementById("reranker-timeout-input")?.value
    ) || 30;

    return {
        run_name: runName,
        ...LLMSettings.parameterFields(),
        language: getPromptLanguage(),
        prompt_overrides: collectPromptOverrides(),
        custom_pricing: collectCustomPricing(),
        documents_path: docsPath,
        llm_provider: llmProvider,
        llm_model: llmModel,
        api_key: apiKey,
        base_url: baseUrl,
        n_jobs: nJobs,
        max_eval_samples: maxSamples,
        max_trials: maxTrials,
        search_space: serializeSearchSpace(),
        metrics: collectSelectedMetrics(),
        eval_dataset_mode: evalDatasetMode,
        eval_dataset_path: evalDatasetPath,
        document_parser: parserName || null,
        document_parser_config: parserName ? collectParserFieldConfig() : null,
        use_ocr: useOcr,
        train_test_split: evalDatasetMode !== "auto" ? { enabled: splitEnabled, test_size: testSize } : null,
        device: device,
        judge_llm_provider: judgeLlmProvider,
        judge_llm_model: judgeLlmModel,
        judge_api_key: judgeApiKey,
        judge_base_url: judgeBaseUrl,
        retrieval_source: retrievalSource,
        adaptive_web_search: adaptiveWebSearch,
        search_provider: searchProvider,
        search_provider_config: searchProviderConfig,
        search_api_key: searchApiKey,
        pipeline_mode: pipelineMode,
        query_transform_llm_provider: queryTransformLlmProvider,
        query_transform_llm_model: queryTransformLlmModel,
        query_transform_api_key: queryTransformApiKey,
        query_transform_base_url: queryTransformBaseUrl,
        dataset_llm_provider: datasetLlmProvider,
        dataset_llm_model: datasetLlmModel,
        dataset_api_key: datasetApiKey,
        dataset_base_url: datasetBaseUrl,
        reranking_model: rerankingModel,
        reranker_llm_provider: useRerankerLlmOverride
            ? (document.getElementById("reranker-llm-provider-select")?.value || null)
            : null,
        reranker_llm_model: useRerankerLlmOverride
            ? (document.getElementById("reranker-llm-model-input")?.value?.trim() || null)
            : null,
        reranker_llm_api_key: useRerankerLlmOverride
            ? (document.getElementById("reranker-llm-api-key-input")?.value?.trim() || null)
            : null,
        reranker_llm_base_url: useRerankerLlmOverride
            ? (document.getElementById("reranker-llm-base-url-input")?.value?.trim() || null)
            : null,
        reranker_base_url: rerankers.has("custom")
            ? (document.getElementById("reranker-base-url-input")?.value?.trim() || null)
            : null,
        reranker_api_key: rerankers.has("custom")
            ? (document.getElementById("reranker-api-key-input")?.value?.trim() || null)
            : null,
        reranker_model: rerankers.has("custom")
            ? (document.getElementById("reranker-model-input")?.value?.trim() || null)
            : null,
        reranker_timeout_seconds: rerankerTimeout,
    };
}

function initFormHandlers() {
    const validateBtn = document.getElementById("btn-dry-run");
    if (validateBtn) {
        validateBtn.addEventListener("click", async () => {
            if (!validateWorkflowForLaunch()) return;
            const payload = getFormData();
            const requestedSignature = getConfigurationSignature(payload);
            validatedConfigSignature = null;
            dryRunInFlight = true;
            renderDryRunGate();

            try {
                const res = await apiPost("/api/runs/validate", payload);
                if (res.valid && requestedSignature === getConfigurationSignature()) {
                    validatedConfigSignature = requestedSignature;
                    showToast(`Dry Run Validated! Total search combinations: ${res.total_combinations}`, "success");
                } else if (res.valid) {
                    showToast("Configuration changed during validation. Run the dry run again.", "warning");
                } else {
                    showToast(`Validation Failed: ${res.error}`, "error");
                }
            } catch (err) {
                showToast(`Validation Error: ${err.message}`, "error");
            } finally {
                dryRunInFlight = false;
                renderDryRunGate();
            }
        });
    }

    const launchBtn = document.getElementById("btn-launch-run");
    if (launchBtn) {
        launchBtn.addEventListener("click", async () => {
            if (!isDryRunCurrent()) {
                invalidateDryRun();
                showToast("Run Dry Run Validate for the current configuration before launch", "warning");
                return;
            }
            if (!validateWorkflowForLaunch()) return;
            const payload = getFormData();
            launchBtn.disabled = true;
            launchBtn.innerHTML = `<span class="material-symbols-outlined text-[17px] animate-spin">progress_activity</span><span>Starting...</span>`;

            try {
                const res = await apiPost("/api/runs", payload);
                showToast("Run launched successfully! Redirecting to live view...", "success");
                setTimeout(() => {
                    window.location.href = `/run_detail.html?run_id=${res.run_id}`;
                }, 800);
            } catch (err) {
                showToast(`Launch Failed: ${err.message}`, "error");
                renderDryRunGate();
            }
        });
    }
}

function resetToDefaults() {
    LLMSettings.reset();
    // Clear all preset toggle chips and remove every custom-added chip.
    document.querySelectorAll(".chip-toggle").forEach(c => {
        if (c.classList.contains("chip-custom")) {
            c.remove();
        } else {
            setChipSelected(c, false);
        }
    });

    // Defaults: query_expansion: multi_query, hyde; retrieval: similarity_search, hybrid; reranking: semantic_similarity; k: 3, 5
    const defaults = {
        query_expansion: ["multi_query", "hyde"],
        retrieval: ["similarity_search", "hybrid"],
        reranking: ["semantic_similarity"],
        k: ["3", "5"],
    };

    Object.entries(defaults).forEach(([stage, vals]) => {
        const group = document.querySelector(`[data-stage-group="${stage}"]`);
        if (!group) return;
        vals.forEach(val => {
            const chip = group.querySelector(`.chip-toggle[data-val="${val}"]`);
            if (chip && !chip.disabled) setChipSelected(chip, true);
        });
    });

    updateComboCount();
    showToast("Reset search space to defaults", "info");
}

function deriveWorkflowState() {
    const pipelineMode = document.querySelector('input[name="pipeline-mode"]:checked')?.value || "full_rag";
    const knowledgeSource = document.querySelector('input[name="retrieval-knowledge-source"]:checked')?.value || "documents";
    const datasetMode = document.querySelector('input[name="eval-dataset-mode"]:checked')?.value || "auto";
    const webSearchOnly = knowledgeSource === "web_search_only";
    const adaptiveWebSearch = knowledgeSource === "adaptive";
    const datasetNeedsGeneration = datasetMode === "auto";
    const retrievalOnly = pipelineMode === "retrieval_only";
    const queryExpansionActive = !webSearchOnly && document.querySelectorAll(
        '[data-stage-group="query_expansion"] .chip-toggle[data-selected="true"]:not([data-val="none"])'
    ).length > 0;
    const queryTransformOverride = document.getElementById("query-transform-override-checkbox")?.checked || false;
    const datasetLlmReuse = document.getElementById("dataset-llm-reuse-checkbox")?.checked !== false;
    const rerankers = selectedRerankers();
    const llmRerankingActive = rerankers.has("llm");
    const customRerankingActive = rerankers.has("custom");

    return {
        pipelineMode,
        datasetMode,
        knowledgeSource,
        webSearchOnly,
        adaptiveWebSearch,
        retrievalOnly,
        datasetNeedsGeneration,
        needsDocuments: !webSearchOnly,
        needsDocumentProcessing: !webSearchOnly,
        needsAnswerLlm: pipelineMode === "full_rag",
        needsDatasetLlm: datasetNeedsGeneration,
        datasetLlmReuse,
        showDatasetLlmFields: datasetNeedsGeneration && (retrievalOnly || !datasetLlmReuse),
        showWebProvider: (webSearchOnly || adaptiveWebSearch) && !retrievalOnly,
        needsProviderStep: pipelineMode === "full_rag" || datasetNeedsGeneration || queryExpansionActive || ((webSearchOnly || adaptiveWebSearch) && !retrievalOnly),
        queryExpansionActive,
        queryTransformOverride,
        showQueryTransformFields: queryExpansionActive && (retrievalOnly || queryTransformOverride),
        llmRerankingActive,
        customRerankingActive,
        allowRetrievalMetrics: !webSearchOnly,
        allowGenerationMetrics: pipelineMode === "full_rag",
    };
}

function setElementVisible(id, visible) {
    const element = document.getElementById(id);
    if (element) element.hidden = !visible;
}

function setMetricGroupChecked(selector, checked) {
    document.querySelectorAll(selector).forEach(input => {
        if (input.checked === checked) return;
        input.checked = checked;
        input.dispatchEvent(new Event("change", { bubbles: true }));
    });
}

function renderProviderRole(state) {
    const providerLabel = document.getElementById("main-llm-provider-label");
    const modelLabel = document.getElementById("main-llm-model-label");
    const purpose = document.getElementById("provider-card-purpose");

    if (providerLabel) providerLabel.textContent = "Answer LLM Provider";
    if (modelLabel) modelLabel.textContent = "Answer LLM Model";
    if (purpose) {
        purpose.textContent = state.needsAnswerLlm
            ? "Generates the final answer after retrieval and is the default evaluation judge."
            : "Not used by Retrieval-Only pipelines.";
    }
}

function renderConfigurationSummary() {
    const container = document.getElementById("configuration-summary");
    if (!container) return;

    const state = deriveWorkflowState();
    const payload = getFormData();
    const metrics = payload.metrics;
    const retrievalLabels = {
        documents: "Retrieval corpus directory",
        web_search_only: "Web search only",
        adaptive: "Retrieval corpus + adaptive web fallback",
    };
    const retrieval = retrievalLabels[state.knowledgeSource] || state.knowledgeSource;
    const dataset = state.datasetMode === "auto"
        ? "Auto-generate at launch"
        : `Existing file: ${payload.eval_dataset_path || "Not set"}`;
    const combinations = buildSearchSpaceCombinations(payload.search_space);
    const maximumTrials = payload.max_trials || "Unlimited";
    const effectiveTrials = payload.max_trials
        ? Math.min(combinations.length, payload.max_trials)
        : combinations.length;
    const parserConfig = sanitizeConfiguration(payload.document_parser_config);
    const searchProviderConfig = sanitizeConfiguration(payload.search_provider_config);
    const answerLlm = payload.llm_model
        ? `${payload.llm_provider} / ${payload.llm_model}`
        : "Not used";
    const datasetLlm = !state.needsDatasetLlm
        ? "Not needed"
        : (payload.dataset_llm_model
            ? `${payload.dataset_llm_provider} / ${payload.dataset_llm_model}`
            : "Reuse Answer LLM");
    const queryTransformLlm = !state.queryExpansionActive
        ? "Not needed (no query transform)"
        : (payload.query_transform_llm_model
            ? `${payload.query_transform_llm_provider} / ${payload.query_transform_llm_model}`
            : "Reuse Answer LLM");
    const judge = payload.judge_llm_model
        ? `${payload.judge_llm_provider || "default"} / ${payload.judge_llm_model}`
        : "Use answer LLM defaults";
    const rerankerLlm = !state.llmRerankingActive
        ? "Not selected"
        : (payload.reranker_llm_model
            ? `${payload.reranker_llm_provider} / ${payload.reranker_llm_model}`
            : (state.retrievalOnly ? "Reuse Query-Transform LLM" : "Reuse Answer LLM"));
    const split = payload.train_test_split?.enabled
        ? `${Math.round(payload.train_test_split.test_size * 100)}% held-out test set`
        : (payload.eval_dataset_mode === "existing" ? "Full dataset (no split)" : "Not applicable in auto mode");
    const pricingSummary = payload.custom_pricing.length
        ? `${payload.custom_pricing.length} provider/model override${payload.custom_pricing.length === 1 ? "" : "s"}`
        : "Default catalog pricing";
    const items = [
        ["Run name", payload.run_name],
        ["Pipeline", state.retrievalOnly ? "Retrieval only" : "Full RAG"],
        ["Prompt language", payload.language === "en" ? "English" : "Arabic"],
        ["Customized prompts", Object.keys(payload.prompt_overrides || {}).join(", ") || "None (packaged defaults)"],
        ["LLM pricing", pricingSummary],
        ["Retrieval source", retrieval],
        ["Answer LLM", answerLlm],
        ...Object.entries(LLMSettings.parameterFields()).map(([role, parameters]) => [role.replaceAll("_", " "), LLMSettings.summary(parameters)]),
        ["Generation variants", LLMSettings.variants().map(v => `${v.provider}/${v.model}: ${LLMSettings.summary(v.parameters)}`).join("; ") || "Fixed answer configuration"],
        ["LLM base URL", payload.base_url || "Provider default"],
        ["LLM credential", payload.api_key ? "Configured (hidden)" : "Not provided"],
        ["Dataset generation LLM", datasetLlm],
        ["Dataset LLM base URL", payload.dataset_base_url || (state.needsDatasetLlm ? "Reuse / provider default" : "Not needed")],
        ["Dataset LLM credential", payload.dataset_api_key ? "Configured (hidden)" : (state.needsDatasetLlm ? "Reuse / not provided" : "Not needed")],
        ["Query-transform LLM", queryTransformLlm],
        ["Query-transform base URL", payload.query_transform_base_url || (state.queryExpansionActive ? "Reuse / provider default" : "Not needed")],
        ["Query-transform credential", payload.query_transform_api_key ? "Configured (hidden)" : "Not provided"],
        ["Local reranker models", payload.search_space.reranking_model?.join(", ") || payload.reranking_model || "BAAI/bge-reranker-base (default)"],
        ["LLM reranker", rerankerLlm],
        ["LLM reranker credential", payload.reranker_llm_api_key ? "Configured (hidden)" : "Reuse / not provided"],
        ["Custom reranker endpoint", payload.reranker_base_url || "Not selected"],
        ["Custom reranker credential", payload.reranker_api_key ? "Configured (hidden)" : "Not provided"],
        ["Web provider", payload.search_provider || "Disabled"],
        ["Web provider options", searchProviderConfig || "Default"],
        ["Web credential", payload.search_api_key ? "Configured (hidden)" : "Not provided"],
        ["Retrieval corpus", state.needsDocuments ? (payload.documents_path || "Not set") : "Not required"],
        ["Document parser", payload.document_parser || "Built-in default"],
        ["Parser options", parserConfig || "Default"],
        ["OCR", payload.use_ocr ? "Enabled" : "Disabled"],
        ["Evaluation data", dataset],
        ["Evaluation split", split],
        ["Metrics", metrics.length ? metrics.join(", ") : "None selected"],
        ["Judge LLM", judge],
        ["Judge base URL", payload.judge_base_url || "Use answer LLM default"],
        ["Judge credential", payload.judge_api_key ? "Configured (hidden)" : "Use answer LLM default"],
        ["Device", payload.device || "Not applicable"],
        ["Resolved combinations", combinations.length.toLocaleString()],
        ["Maximum trials", String(maximumTrials)],
        ["Trials to launch", effectiveTrials.toLocaleString()],
        ["Samples / trial", String(payload.max_eval_samples)],
        ["Parallel trials", String(payload.n_jobs)],
    ];

    container.innerHTML = items.map(([label, value]) => `
        <div class="bg-surface-container-low border border-outline-variant/30 rounded px-space-sm py-space-xs min-w-0">
          <div class="font-code-sm text-[10px] uppercase tracking-wide text-outline">${escapeHtml(label)}</div>
          <div class="font-code-sm text-[12px] text-on-surface mt-0.5 break-words">${escapeHtml(value)}</div>
        </div>
    `).join("");

    renderSearchSpaceReview(payload.search_space, combinations);
    renderPromptReview();
    renderCustomPricingReview(payload.custom_pricing);
}

function renderCustomPricingReview(entries) {
    const card = document.getElementById("pricing-review-card");
    const status = document.getElementById("pricing-review-status");
    const content = document.getElementById("pricing-review-content");
    if (!card || !status || !content) return;
    card.hidden = !entries.length;
    status.textContent = entries.length
        ? `${entries.length} override${entries.length === 1 ? "" : "s"} · USD per 1M tokens`
        : "Default catalog pricing";
    content.innerHTML = entries.length ? `
      <table class="w-full text-[11px] font-code-sm border-collapse">
        <thead><tr class="text-outline">
          <th class="text-left px-space-sm py-space-xs border-b border-outline-variant/40">Provider / Model</th>
          <th class="text-right px-space-sm py-space-xs border-b border-outline-variant/40">Input / 1M</th>
          <th class="text-right px-space-sm py-space-xs border-b border-outline-variant/40">Output / 1M</th>
        </tr></thead>
        <tbody>${entries.map(entry => `
          <tr>
            <td class="px-space-sm py-space-xs border-b border-outline-variant/15 text-on-surface">${escapeHtml(entry.provider)} / ${escapeHtml(entry.model)}</td>
            <td class="px-space-sm py-space-xs border-b border-outline-variant/15 text-right text-tertiary">$${escapeHtml(entry.input_usd_per_million_tokens)}</td>
            <td class="px-space-sm py-space-xs border-b border-outline-variant/15 text-right text-tertiary">$${escapeHtml(entry.output_usd_per_million_tokens)}</td>
          </tr>`).join("")}</tbody>
      </table>` : "";
}

function renderPromptReview() {
    const container = document.getElementById("prompt-review-content");
    const status = document.getElementById("prompt-review-status");
    const card = document.getElementById("prompt-review-card");
    if (!container || !status || !card) return;
    saveCurrentPromptDrafts();
    const language = getPromptLanguage();
    const drafts = promptDraftsByLanguage[language];
    const customized = promptDefinitions.filter(item => drafts[item.key]?.custom);
    status.textContent = `${language === "en" ? "English" : "Arabic"} · ${customized.length} customized`;
    card.hidden = promptDefinitions.length === 0;
    container.innerHTML = promptDefinitions.map(definition => {
        const draft = drafts[definition.key];
        const template = draft?.custom ? draft.value : definition.default_template;
        return `
          <details class="bg-surface-container-low border border-outline-variant/30 rounded px-space-sm py-space-xs">
            <summary class="cursor-pointer flex items-center justify-between gap-space-sm font-code-sm text-[11px] text-on-surface">
              <span>${escapeHtml(definition.label)}</span>
              <span class="text-${draft?.custom ? "primary" : "outline"}">${draft?.custom ? "Custom" : "Default"}</span>
            </summary>
            <pre dir="auto" class="mt-space-xs p-space-sm rounded bg-surface-container overflow-auto whitespace-pre-wrap break-words font-code-sm text-[10px] text-on-surface-variant">${escapeHtml(template)}</pre>
          </details>`;
    }).join("");
}

function sanitizeConfiguration(config) {
    if (!config || Object.keys(config).length === 0) return "";
    const safe = {};
    Object.entries(config).forEach(([key, value]) => {
        safe[key] = /key|token|secret|password/i.test(key) ? "[configured]" : value;
    });
    return JSON.stringify(safe);
}

function formatSearchSpaceValue(value) {
    if (value === undefined || value === null) return "—";
    if (value && typeof value === "object") return JSON.stringify(value);
    return String(value);
}

function buildSearchSpaceCombinations(searchSpace) {
    const entries = Object.entries(searchSpace || {}).filter(([, values]) => Array.isArray(values) && values.length > 0);
    if (entries.length === 0) return [];
    const combinations = entries.reduce(
        (rows, [stage, values]) => rows.flatMap(row => values.map(value => ({ ...row, [stage]: value }))),
        [{}]
    );
    const seen = new Set();
    return combinations.reduce((result, combination) => {
        const normalized = { ...combination };
        const method = String(normalized.reranking || "").toLowerCase();
        if ("reranking_model" in normalized && !["cross_encoder", "pointwise"].includes(method)) {
            delete normalized.reranking_model;
        }
        const signature = JSON.stringify(normalized);
        if (seen.has(signature)) return result;
        seen.add(signature);
        result.push(normalized);
        return result;
    }, []);
}

function renderSearchSpaceReview(searchSpace, combinations) {
    const summary = document.getElementById("search-space-summary");
    const resolved = document.getElementById("resolved-combinations");
    const countBadge = document.getElementById("review-combination-count");
    const entries = Object.entries(searchSpace || {});

    if (summary) {
        summary.innerHTML = entries.map(([stage, values]) => `
          <div class="bg-surface-container-low border border-outline-variant/30 rounded px-space-sm py-space-xs min-w-0">
            <div class="font-code-sm text-[10px] uppercase tracking-wide text-outline">${escapeHtml(stage.replaceAll("_", " "))}</div>
            <div class="font-code-sm text-[12px] text-on-surface mt-1 break-words">${values.map(formatSearchSpaceValue).map(escapeHtml).join(" · ")}</div>
          </div>
        `).join("") || '<div class="text-outline text-body-xs">No active search-space values.</div>';
    }

    if (countBadge) {
        countBadge.textContent = `${combinations.length.toLocaleString()} combination${combinations.length === 1 ? "" : "s"}`;
    }
    if (!resolved) return;
    if (combinations.length === 0) {
        resolved.innerHTML = '<div class="p-space-base text-outline text-body-xs">No combinations will run.</div>';
        return;
    }

    const stages = entries.map(([stage]) => stage);
    const headers = stages.map(stage => `<th class="sticky top-0 bg-surface-container-high text-left text-outline px-space-sm py-space-xs border-b border-outline-variant/40 whitespace-nowrap">${escapeHtml(stage.replaceAll("_", " "))}</th>`).join("");
    const rows = combinations.map((combination, index) => `
      <tr class="hover:bg-surface-container-low transition-colors">
        <td class="text-primary font-bold px-space-sm py-space-xs border-b border-outline-variant/15">${index + 1}</td>
        ${stages.map(stage => `<td class="text-on-surface-variant px-space-sm py-space-xs border-b border-outline-variant/15 align-top whitespace-nowrap">${escapeHtml(formatSearchSpaceValue(combination[stage]))}</td>`).join("")}
      </tr>
    `).join("");
    resolved.innerHTML = `
      <table class="w-full text-[11px] font-code-sm border-collapse">
        <thead><tr><th class="sticky top-0 bg-surface-container-high text-left text-outline px-space-sm py-space-xs border-b border-outline-variant/40">#</th>${headers}</tr></thead>
        <tbody>${rows}</tbody>
      </table>
    `;
}

function getConfigurationSignature(payload = getFormData()) {
    const serialized = JSON.stringify(payload);
    let hash = 2166136261;
    for (let index = 0; index < serialized.length; index += 1) {
        hash ^= serialized.charCodeAt(index);
        hash = Math.imul(hash, 16777619);
    }
    return (hash >>> 0).toString(16);
}

function isDryRunCurrent() {
    return Boolean(validatedConfigSignature) && validatedConfigSignature === getConfigurationSignature();
}

function invalidateDryRun() {
    if (validatedConfigSignature && !isDryRunCurrent()) validatedConfigSignature = null;
    renderDryRunGate();
}

function renderDryRunGate() {
    const validateBtn = document.getElementById("btn-dry-run");
    const launchBtn = document.getElementById("btn-launch-run");
    const badge = document.getElementById("review-validation-badge");
    const icon = document.getElementById("dry-run-status-icon");
    const title = document.getElementById("dry-run-status-title");
    const message = document.getElementById("dry-run-status-message");
    const current = !dryRunInFlight && isDryRunCurrent();
    const combinations = buildSearchSpaceCombinations(serializeSearchSpace());
    const maximumTrials = parseInt(document.getElementById("max-trials-input")?.value, 10);
    const launchCount = Number.isFinite(maximumTrials)
        ? Math.min(maximumTrials, combinations.length)
        : combinations.length;

    if (validateBtn) {
        validateBtn.disabled = dryRunInFlight;
        validateBtn.innerHTML = dryRunInFlight
            ? '<span class="material-symbols-outlined text-[16px] animate-spin">progress_activity</span><span>Validating...</span>'
            : '<span class="material-symbols-outlined text-[16px] text-tertiary">check_circle</span><span>Dry Run Validate</span>';
    }
    if (launchBtn) {
        launchBtn.hidden = !current;
        launchBtn.disabled = !current || combinations.length === 0;
        const label = `Launch ${launchCount.toLocaleString()} Trial${launchCount === 1 ? "" : "s"}`;
        launchBtn.innerHTML = `<span class="material-symbols-outlined text-[17px]">rocket_launch</span><span>${label}</span>`;
    }
    if (badge) {
        badge.textContent = dryRunInFlight ? "Validating" : (current ? "Validated" : "Validate first");
        badge.className = current
            ? "px-2 py-0.5 rounded-full bg-tertiary/15 border border-tertiary/40 font-code-sm text-[10px] text-tertiary font-semibold"
            : "px-2 py-0.5 rounded-full bg-surface-container border border-outline-variant/40 font-code-sm text-[10px] text-outline";
    }
    if (icon) {
        icon.textContent = dryRunInFlight ? "progress_activity" : (current ? "verified" : "verified_user");
        icon.className = `material-symbols-outlined text-[22px] mt-0.5 ${current ? "text-tertiary" : "text-outline"}${dryRunInFlight ? " animate-spin" : ""}`;
    }
    if (title) title.textContent = dryRunInFlight ? "Validating configuration" : (current ? "Dry run succeeded" : "Validate before launch");
    if (message) {
        message.textContent = current
            ? "The current configuration passed preflight. Launch is now available."
            : (dryRunInFlight
                ? "Checking providers, dataset, metrics, and every search-space dimension."
                : "Run the dry-run check to verify the current providers, dataset, metrics, and search space. Launch appears only after it succeeds.");
    }
}

function renderWorkflow() {
    LLMSettings.refresh();
    if (workflowRenderInProgress) return;
    workflowRenderInProgress = true;

    try {
        let state = deriveWorkflowState();

        // Retrieval-only evaluates a local index, so web-backed sources are
        // unavailable. Web-only retrieval requires a supplied evaluation
        // dataset because automatic generation depends on a local corpus.
        if (state.retrievalOnly && (state.webSearchOnly || state.adaptiveWebSearch)) {
            const documentsSource = document.querySelector('input[name="retrieval-knowledge-source"][value="documents"]');
            if (documentsSource) documentsSource.checked = true;
        }
        state = deriveWorkflowState();
        if (state.webSearchOnly && state.datasetMode === "auto") {
            const existingDataset = document.querySelector('input[name="eval-dataset-mode"][value="existing"]');
            if (existingDataset) {
                existingDataset.checked = true;
                existingDataset.dispatchEvent(new Event("change", { bubbles: true }));
            }
        }
        state = deriveWorkflowState();

        document.querySelectorAll(".retrieval-source-tile").forEach(tile => {
            const value = tile.getAttribute("data-source");
            const unavailable = state.retrievalOnly && (value === "web_search_only" || value === "adaptive");
            const input = tile.querySelector('input[name="retrieval-knowledge-source"]');
            if (input) input.disabled = unavailable;
            tile.classList.toggle("opacity-40", unavailable);
            tile.classList.toggle("cursor-not-allowed", unavailable);
            tile.classList.toggle("cursor-pointer", !unavailable);
            const selected = value === state.knowledgeSource;
            tile.classList.toggle("bg-surface-container-high", selected);
            tile.classList.toggle("border-primary", selected);
            tile.classList.toggle("text-primary", selected);
            tile.classList.toggle("bg-surface-container-low", !selected);
            tile.classList.toggle("border-outline-variant/40", !selected);
            tile.classList.toggle("text-outline", !selected);
        });

        document.querySelectorAll(".dataset-mode-tile").forEach(tile => {
            const mode = tile.getAttribute("data-mode");
            const unavailable = mode === "auto" && state.webSearchOnly;
            const input = tile.querySelector('input[name="eval-dataset-mode"]');
            if (input) input.disabled = unavailable;
            tile.classList.toggle("opacity-40", unavailable);
            tile.classList.toggle("cursor-not-allowed", unavailable);
            const selected = mode === state.datasetMode;
            tile.classList.toggle("active", selected);
            tile.classList.toggle("bg-surface-container-high", selected);
            tile.classList.toggle("border-primary", selected);
            tile.classList.toggle("text-primary", selected);
            tile.classList.toggle("bg-surface-container-low", !selected);
            tile.classList.toggle("border-outline-variant/40", !selected);
            tile.classList.toggle("text-outline", !selected);
        });

        setElementVisible("dataset-existing-fields", state.datasetMode === "existing");
        const splitCheckbox = document.getElementById("train-test-split-checkbox");
        const testSizeInput = document.getElementById("test-size-input");
        if (splitCheckbox) splitCheckbox.disabled = state.datasetMode !== "existing";
        if (testSizeInput) testSizeInput.disabled = state.datasetMode !== "existing" || !splitCheckbox?.checked;

        setElementVisible("retrieval-source-card", true);
        setElementVisible("documents-config", state.needsDocuments);
        setElementVisible("provider-data-card", state.needsAnswerLlm);
        setElementVisible("web-provider-card", state.showWebProvider);
        setElementVisible("dataset-provider-card", state.needsDatasetLlm);
        setElementVisible("split-policy-card", state.datasetMode !== "auto");
        setElementVisible("retrieval-metrics-card", state.allowRetrievalMetrics);
        setElementVisible("generation-metrics-card", state.allowGenerationMetrics);
        setElementVisible("local-search-space", !state.webSearchOnly);
        setElementVisible("device-config", !state.webSearchOnly);
        setElementVisible("web-search-only-notice", state.webSearchOnly);

        const parserPurpose = document.getElementById("document-processing-purpose");
        const ocrLabel = document.getElementById("use-ocr-label");
        if (parserPurpose) {
            parserPurpose.textContent = state.webSearchOnly
                ? "Optional — used while generating evaluation data"
                : "Optional — scanned / complex documents";
        }
        if (ocrLabel) {
            ocrLabel.textContent = state.webSearchOnly
                ? "Use OCR for evaluation-data generation"
                : "Use OCR for indexing";
        }

        const docsMarker = document.getElementById("docs-path-required-marker");
        if (docsMarker) docsMarker.style.display = state.needsDocuments ? "" : "none";
        const docsHelp = document.getElementById("documents-path-help");
        if (docsHelp) {
            docsHelp.textContent = state.datasetMode === "auto"
                ? "Corpus used for indexing and for generating evaluation Q&A."
                : "Local corpus that will be parsed, chunked, and indexed for retrieval.";
        }

        const datasetReuse = document.getElementById("dataset-llm-reuse-checkbox");
        const datasetReuseControl = document.getElementById("dataset-llm-reuse-control");
        if (datasetReuse) {
            if (state.retrievalOnly) datasetReuse.checked = false;
            datasetReuse.disabled = state.retrievalOnly;
        }
        if (datasetReuseControl) datasetReuseControl.hidden = state.retrievalOnly;
        setElementVisible(
            "dataset-llm-fields",
            state.needsDatasetLlm && (state.retrievalOnly || !datasetReuse?.checked)
        );

        state = deriveWorkflowState();
        renderQueryTransformControls(state);

        if (!state.allowRetrievalMetrics) {
            setMetricGroupChecked("#metrics-retrieval-group .metric-checkbox:checked", false);
            if (!document.querySelector("#metrics-generation-group .metric-checkbox:checked")) {
                setMetricGroupChecked("#metrics-generation-group .metric-checkbox", true);
            }
        }
        if (!state.allowGenerationMetrics) {
            setMetricGroupChecked("#metrics-generation-group .metric-checkbox:checked", false);
            if (!document.querySelector("#metrics-retrieval-group .metric-checkbox:checked")) {
                const defaults = new Set(catalogData?.metrics?.default || ["recall", "precision", "mrr"]);
                document.querySelectorAll("#metrics-retrieval-group .metric-checkbox").forEach(input => {
                    if (!defaults.has(input.value) || input.checked) return;
                    input.checked = true;
                    input.dispatchEvent(new Event("change", { bubbles: true }));
                });
            }
        }

        const hasGenerationMetric = Boolean(
            document.querySelector("#metrics-generation-group .metric-checkbox:checked")
        );
        setElementVisible("judge-override-controls", state.allowGenerationMetrics && hasGenerationMetric);
        if (!hasGenerationMetric) {
            const override = document.getElementById("judge-llm-override-checkbox");
            if (override) override.checked = false;
            setElementVisible("judge-llm-fields", false);
        }

        renderProviderRole(state);
        renderCustomPricingModels();
        updateMetricsCounters();
        updateWorkflowTabs();
        updateComboCount();
    } finally {
        workflowRenderInProgress = false;
    }
}

function validateWorkflowStep(stepId) {
    const state = deriveWorkflowState();

    if (stepId === "tab-panel-data") {
        if (state.datasetMode === "existing" && !document.getElementById("eval-dataset-path-input")?.value?.trim()) {
            showToast("Enter the evaluation dataset file path", "error");
            return false;
        }
        if (state.needsDocuments && !document.getElementById("docs-path-input")?.value?.trim()) {
            showToast("Enter the retrieval corpus path", "error");
            return false;
        }
    }

    if (stepId === "tab-panel-providers") {
        if (state.needsAnswerLlm && !document.getElementById("llm-model-input")?.value?.trim()) {
            showToast("Enter the Answer LLM model", "error");
            return false;
        }
        if (state.showWebProvider && !document.getElementById("search-provider-select")?.value) {
            showToast("Choose a web search provider", "error");
            return false;
        }
        if (state.showDatasetLlmFields && !document.getElementById("dataset-llm-model-input")?.value?.trim()) {
            showToast("Enter the Dataset Generation LLM model", "error");
            return false;
        }
        if (state.showQueryTransformFields && !document.getElementById("query-transform-llm-model-input")?.value?.trim()) {
            showToast("Enter the Query-Transform LLM model", "error");
            return false;
        }
        if (!validateCustomPricing()) return false;
    }

    if (stepId === "tab-panel-eval") {
        if (collectSelectedMetrics().length === 0) {
            showToast("Select at least one applicable evaluation metric", "error");
            return false;
        }
    }

    if (stepId === "tab-panel-search-space") {
        if (state.showQueryTransformFields && !document.getElementById("query-transform-llm-model-input")?.value?.trim()) {
            showToast("Enter the Query-Transform LLM model", "error");
            return false;
        }
        if (buildSearchSpaceCombinations(serializeSearchSpace()).length === 0) {
            showToast("Select at least one search-space value", "error");
            return false;
        }
        if (state.llmRerankingActive) {
            const dedicated = document.getElementById("reranker-llm-override-checkbox")?.checked || false;
            if (dedicated && !document.getElementById("reranker-llm-model-input")?.value?.trim()) {
                showToast("Enter the dedicated LLM reranker model", "error");
                return false;
            }
            if (state.retrievalOnly && !dedicated && !getFormData().query_transform_llm_model) {
                showToast("Choose a dedicated LLM for reranking in Retrieval-Only mode, or configure a Query-Transform LLM to reuse", "error");
                return false;
            }
        }
        if (state.customRerankingActive) {
            const endpoint = document.getElementById("reranker-base-url-input")?.value?.trim() || "";
            if (!/^https?:\/\//i.test(endpoint)) {
                showToast("Enter a custom reranker endpoint starting with http:// or https://", "error");
                return false;
            }
        }
    }

    if (stepId === "tab-panel-prompts") {
        const expectedSignature = JSON.stringify(buildPromptContext());
        if (promptResolutionError) {
            showToast(`Prompt resolution failed: ${promptResolutionError}`, "error");
            return false;
        }
        if (locallyApplicablePromptKeys().size > 0 && promptResolutionSignature !== expectedSignature) {
            showToast("Wait for the applicable prompts to finish loading", "warning");
            loadApplicablePrompts();
            return false;
        }
        const invalid = Array.from(document.querySelectorAll("[data-prompt-card]"))
            .filter(card => !updatePromptCardValidation(card));
        if (invalid.length) {
            showToast(`Fix ${invalid.length} invalid custom prompt${invalid.length === 1 ? "" : "s"} before continuing`, "error");
            invalid[0].querySelector("textarea")?.focus();
            return false;
        }
    }

    return true;
}

function validateWorkflowForLaunch() {
    const parameterErrors = LLMSettings.errors();
    if (parameterErrors.length) {
        showToast(parameterErrors.join(" "), "error");
        return false;
    }
    return validateWorkflowStep("tab-panel-data")
        && validateWorkflowStep("tab-panel-providers")
        && validateWorkflowStep("tab-panel-eval")
        && validateWorkflowStep("tab-panel-search-space")
        && validateWorkflowStep("tab-panel-prompts");
}

function updateWorkflowTabs() {
    const state = deriveWorkflowState();
    document.querySelectorAll(".notebook-tab").forEach((tab, index) => {
        const isIrrelevantParserStep = tab.getAttribute("data-tab") === "tab-panel-parser" && !state.needsDocumentProcessing;
        const isIrrelevantProviderStep = tab.getAttribute("data-tab") === "tab-panel-providers" && !state.needsProviderStep;
        const isIrrelevantPromptStep = tab.getAttribute("data-tab") === "tab-panel-prompts" && locallyApplicablePromptKeys().size === 0;
        const locked = index > highestUnlockedStep || isIrrelevantParserStep || isIrrelevantProviderStep || isIrrelevantPromptStep;
        tab.disabled = locked;
        tab.setAttribute("aria-disabled", locked ? "true" : "false");
        tab.classList.toggle("opacity-40", locked);
        tab.classList.toggle("cursor-not-allowed", locked);
        if (locked) tab.classList.remove("cursor-pointer");
        else tab.classList.add("cursor-pointer");
    });
}

function advanceWorkflow(targetId) {
    const current = document.querySelector(".notebook-tab-panel:not(.hidden)")?.id || WORKFLOW_STEPS[0];
    if (!validateWorkflowStep(current)) return;

    let resolvedTarget = targetId;
    const state = deriveWorkflowState();
    if (targetId === "tab-panel-parser" && !state.needsDocumentProcessing) {
        resolvedTarget = "tab-panel-providers";
    }
    if (resolvedTarget === "tab-panel-providers" && !state.needsProviderStep) {
        resolvedTarget = "tab-panel-eval";
    }
    if (resolvedTarget === "tab-panel-prompts" && locallyApplicablePromptKeys().size === 0) {
        resolvedTarget = "tab-panel-review";
    }

    const targetIndex = WORKFLOW_STEPS.indexOf(resolvedTarget);
    highestUnlockedStep = Math.max(highestUnlockedStep, targetIndex);
    updateWorkflowTabs();
    switchNotebookTab(resolvedTarget);
}

function backWorkflow(targetId) {
    const state = deriveWorkflowState();
    let resolvedTarget = targetId;
    if (resolvedTarget === "tab-panel-providers" && !state.needsProviderStep) {
        resolvedTarget = "tab-panel-parser";
    }
    if (resolvedTarget === "tab-panel-parser" && !state.needsDocumentProcessing) {
        resolvedTarget = "tab-panel-data";
    }
    if (resolvedTarget === "tab-panel-prompts" && locallyApplicablePromptKeys().size === 0) {
        resolvedTarget = "tab-panel-search-space";
    }
    switchNotebookTab(resolvedTarget);
}

function initGuidedWorkflow() {
    const dependencySelector = [
        'input[name="pipeline-mode"]',
        'input[name="retrieval-knowledge-source"]',
        'input[name="eval-dataset-mode"]',
        "#dataset-llm-reuse-checkbox",
        "#query-transform-override-checkbox",
        "#reranker-llm-override-checkbox",
        "#reranker-llm-provider-select",
        "#llm-provider-select",
        "#query-transform-llm-provider-select",
        "#judge-llm-override-checkbox",
        "#judge-llm-provider-select",
        ".metric-checkbox",
    ].join(",");

    document.addEventListener("change", event => {
        if (event.target.matches(dependencySelector)) renderWorkflow();
        else renderConfigurationSummary();
        invalidateDryRun();
    });
    document.addEventListener("input", () => {
        renderConfigurationSummary();
        invalidateDryRun();
    });
    document.addEventListener("click", event => {
        if (event.target.closest(".chip-toggle, #btn-add-k, #btn-add-chunking, #btn-add-embedding, #btn-add-reranking-model, #btn-reset")) {
            window.setTimeout(invalidateDryRun, 0);
        }
    });

    updateWorkflowTabs();
    renderWorkflow();
    switchNotebookTab(WORKFLOW_STEPS[0]);
}

function initNotebookTabs() {
    const tabs = document.querySelectorAll(".notebook-tab");
    tabs.forEach(tab => {
        tab.addEventListener("click", () => {
            const targetId = tab.getAttribute("data-tab");
            const currentId = document.querySelector(".notebook-tab-panel:not(.hidden)")?.id;
            if (currentId && targetId !== currentId && !validateWorkflowStep(currentId)) return;
            if (targetId === "tab-panel-review" && locallyApplicablePromptKeys().size > 0 && !validateWorkflowStep("tab-panel-prompts")) {
                switchNotebookTab("tab-panel-prompts");
                return;
            }
            if (targetId) switchNotebookTab(targetId);
        });
    });
}

function switchNotebookTab(targetId) {
    const tabs = document.querySelectorAll(".notebook-tab");
    const panels = document.querySelectorAll(".notebook-tab-panel");

    panels.forEach(p => {
        if (p.id === targetId) {
            p.classList.remove("hidden");
        } else {
            p.classList.add("hidden");
        }
    });

    tabs.forEach(t => {
        if (t.getAttribute("data-tab") === targetId) {
            t.classList.add("active", "text-primary", "border-primary", "bg-surface-container/60", "shadow-sm");
            t.classList.remove("text-on-surface-variant", "border-transparent", "hover:bg-surface-container/30");
            t.setAttribute("aria-selected", "true");
        } else {
            t.classList.remove("active", "text-primary", "border-primary", "bg-surface-container/60", "shadow-sm");
            t.classList.add("text-on-surface-variant", "border-transparent", "hover:bg-surface-container/30");
            t.setAttribute("aria-selected", "false");
        }
    });

    if (targetId === "tab-panel-prompts") {
        loadApplicablePrompts();
    }
    if (targetId === "tab-panel-review") {
        renderConfigurationSummary();
        renderDryRunGate();
    }

    // Scroll smoothly to top of main if user navigated via step footer
    window.scrollTo({ top: 0, behavior: "smooth" });
}

window.switchNotebookTab = switchNotebookTab;
window.advanceWorkflow = advanceWorkflow;
window.backWorkflow = backWorkflow;
