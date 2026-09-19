const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const source = fs.readFileSync(path.join(__dirname, '../static/js/llm_settings.js'), 'utf8');

// Minimal DOM doubles exercise the real editor handlers without a provider/server.
function harness() {
    const ids = new Map();
    class Element {
        constructor(tag = 'div') { this.tag = tag; this.children = []; this.dataset = {}; this.style = {}; this.validity = { badInput: false }; this.value = ''; }
        set innerHTML(html) {
            this.html = html; this.children = [];
            for (const match of html.matchAll(/<([a-z][\w-]*)\b([^>]*)>/g)) {
                const child = new Element(match[1]);
                for (const attr of match[2].matchAll(/([\w-]+)(?:="([^"]*)")?/g)) child.setAttribute(attr[1], attr[2] ?? '');
                this.children.push(child);
            }
        }
        get innerHTML() { return this.html || ''; }
        setAttribute(key, val) {
            if (key === 'id') ids.set(val, this);
            if (key === 'value') this.value = val;
            if (key === 'checked') this.checked = true;
            if (key.startsWith('data-')) this.dataset[key.slice(5).replace(/-([a-z])/g, (_, s) => s.toUpperCase())] = val;
        }
        appendChild(child) { this.children.push(child); return child; }
        querySelector(selector) {
            const attr = selector.match(/^\[data-([\w-]+)(?:="([^"]*)")?\]$/);
            const key = attr?.[1].replace(/-([a-z])/g, (_, s) => s.toUpperCase());
            const match = child => attr ? key in child.dataset && (attr[2] === undefined || child.dataset[key] === attr[2]) : child.tag === selector;
            for (const child of this.children) {
                if (match(child)) return child;
                const nested = child.querySelector(selector);
                if (nested) return nested;
            }
            return null;
        }
        showModal() { this.open = true; }
        close() { this.open = false; }
    }
    const state = { needsAnswerLlm: true, allowGenerationMetrics: true, needsDatasetLlm: true,
        queryExpansionActive: true, llmRerankingActive: true, retrievalOnly: false };
    const existing = ['provider-data-card', 'judge-override-controls', 'query-transform-provider-card', 'reranker-llm-card', 'dataset-provider-card',
        'llm-provider-select', 'llm-model-input', 'judge-llm-provider-select', 'query-transform-llm-provider-select', 'dataset-llm-provider-select', 'reranker-llm-provider-select'];
    existing.forEach(id => ids.set(id, new Element()));
    ids.get('llm-provider-select').value = 'openai';
    ids.get('llm-model-input').value = 'same-model';
    const spec = { temperature: { label: 'Temperature', min: 0, max: 2, nullable: true },
        max_tokens: { label: 'Maximum output tokens', min: 1, integer: true },
        top_p: { label: 'Top-p', min: 0, max: 1 }, seed: { label: 'Seed', integer: true },
        stop: { label: 'Stop sequences', type: 'strings' }, max_retries: { label: 'Provider request retries', min: 0, integer: true } };
    const anthropic = { temperature: { ...spec.temperature, max: 1 }, max_tokens: spec.max_tokens, top_p: spec.top_p };
    const ctx = { document: { getElementById: id => ids.get(id), createElement: tag => new Element(tag), body: new Element('body'), addEventListener() {} },
        deriveWorkflowState: () => state, collectSelectedMetrics: () => ['faithfulness'],
        escapeHtml: text => String(text).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('"', '&quot;'),
        invalidateDryRun() {}, updateComboCount() {} };
    vm.createContext(ctx);
    vm.runInContext(source + '\nglobalThis.settings = LLMSettings;', ctx);
    ctx.settings.init({ llm_parameter_capabilities: { openai: spec, anthropic } });
    const plain = val => JSON.parse(JSON.stringify(val));
    const input = (host, key, val) => { const node = host.querySelector(`[data-parameter="${key}"]`); node.value = val; node.oninput(); };
    return { ids, state, settings: ctx.settings, input, plain };
}

test('add, duplicate, edit, remove variants and block exact duplicates', () => {
    const { ids, settings, input, plain } = harness();
    ids.get('llm-add-variant').onclick();
    ids.get('llm-variant-save').onclick();
    assert.equal(settings.variants().length, 1);
    let row = ids.get('llm-variant-list').children[0];
    row.querySelector('[data-duplicate]').onclick();
    ids.get('llm-variant-save').onclick();
    assert.match(ids.get('llm-variant-error').textContent, /already exists/);
    input(ids.get('llm-variant-editor'), 'temperature', '0.7');
    ids.get('llm-variant-save').onclick();
    assert.deepEqual(plain(settings.variants().map(v => v.parameters.temperature)), [0, 0.7]);
    row = ids.get('llm-variant-list').children[1];
    row.querySelector('[data-edit]').onclick();
    input(ids.get('llm-variant-editor'), 'temperature', '0.3');
    ids.get('llm-variant-save').onclick();
    assert.equal(settings.variants()[1].parameters.temperature, 0.3);
    ids.get('llm-variant-list').children[0].querySelector('[data-remove]').onclick();
    assert.equal(settings.variants().length, 1);
    assert.equal(settings.parameterFields().judge_llm_parameters.temperature, 0);
});

test('provider changes retain unsupported values for explicit correction', () => {
    const { ids, settings, input } = harness();
    ids.get('llm-add-variant').onclick();
    input(ids.get('llm-variant-editor'), 'seed', '0');
    ids.get('llm-variant-provider').value = 'anthropic';
    ids.get('llm-variant-provider').onchange();
    ids.get('llm-variant-save').onclick();
    assert.match(ids.get('llm-variant-error').textContent, /does not support seed/);
    assert.equal(settings.variants().length, 0);
});

test('temperature omission, zero retries, reset, and inactive roles', () => {
    const { ids, settings, state, input, plain } = harness();
    const panel = ids.get('provider-data-card').children[0];
    const editor = panel.querySelector('div');
    input(editor, 'max_retries', '0');
    const grid = editor.querySelector('[data-parameter-grid]');
    const checkbox = grid.children[1].querySelector('input');
    checkbox.onchange({ target: { checked: true } });
    assert.equal(settings.parameterFields().llm_parameters.temperature, null);
    assert.equal(settings.parameterFields().llm_parameters.max_retries, 0);
    assert.deepEqual(plain(settings.errors()), []);
    state.needsAnswerLlm = false; state.retrievalOnly = true; state.allowGenerationMetrics = false;
    settings.refresh();
    assert.ok(panel.hidden);
    assert.ok(!('llm_parameters' in settings.parameterFields()));
    assert.ok(!('judge_llm_parameters' in settings.parameterFields()));
    settings.reset();
    state.needsAnswerLlm = true;
    assert.equal(settings.parameterFields().llm_parameters.temperature, 0);
});

test('search serialization and web-only counts include parameterized variants', () => {
    const run = fs.readFileSync(path.join(__dirname, '../static/js/new_run.js'), 'utf8');
    const code = run.slice(run.indexOf('function serializeSearchSpace()'), run.indexOf('\nfunction collectSelectedMetrics()'));
    const variant = { provider: 'openai', model: 'same', parameters: { temperature: 0.7 } };
    const ctx = { LLMSettings: { variants: () => [variant] }, deriveWorkflowState: () => ({ webSearchOnly: true }) };
    vm.createContext(ctx); vm.runInContext(code, ctx);
    assert.equal(ctx.serializeSearchSpace().llm[0].parameters.temperature, 0.7);
    assert.match(run, /total = Math\.max\(1, LLMSettings\.variants\(\)\.length\)/);
});
