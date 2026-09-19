const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/js/run_detail.js'), 'utf8');
const start = source.indexOf('async function loadTrialPythonExport()');
const end = source.indexOf('\nfunction renderRunInspectorContent', start);

function harness(apiGet) {
    const container = { innerHTML: '' };
    const button = { addEventListener(type, callback) { this[type] = callback; } };
    const ctx = {
        inspectorData: { trial: { trial_id: 7 } }, runId: 'run-7', activeInspectorTab: 'export',
        apiGet, document: { getElementById: id => id === 'trial-python-export' ? container : button },
        navigator: { clipboard: { async writeText(code) { ctx.copied = code; } } },
        escapeHtml: value => String(value).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'),
    };
    vm.createContext(ctx);
    vm.runInContext(source.slice(start, end), ctx);
    return { ctx, container, button };
}

const preview = {
    metadata: { pipeline_mode: 'full_rag', retrieval_source: 'vector_db', status: 'failed' },
    notices: ['Trial failed'], requirements: 'Muffakir', environment: [], readme: 'Set up',
    code: 'print("<script>prompt</script>")',
};

test('preview, copy and download links target the selected failed trial', async () => {
    let requested;
    const { ctx, container, button } = harness(async url => { requested = url; return preview; });
    await ctx.loadTrialPythonExport();
    assert.equal(requested, '/api/runs/run-7/trials/7/export?format=preview');
    assert.match(container.innerHTML, /trials\/7\/export\?format=python/);
    assert.match(container.innerHTML, /trials\/7\/export\?format=zip/);
    assert.match(container.innerHTML, /Original status: failed/);
    assert.ok(!container.innerHTML.includes('<script>'));
    await button.click({ currentTarget: button });
    assert.equal(ctx.copied, preview.code);
    assert.equal(button.textContent, 'Copied');
});

test('a stale response cannot overwrite a newly selected trial', async () => {
    let resolve;
    const { ctx, container } = harness(() => new Promise(done => { resolve = done; }));
    const pending = ctx.loadTrialPythonExport();
    ctx.inspectorData = { trial: { trial_id: 8 } };
    container.innerHTML = 'Trial 8';
    resolve(preview);
    await pending;
    assert.equal(container.innerHTML, 'Trial 8');
});

test('unusable saved configuration displays reason and disables download', async () => {
    const { ctx, container } = harness(async () => { throw new Error('No saved resolved configuration'); });
    await ctx.loadTrialPythonExport();
    assert.match(container.innerHTML, /No saved resolved configuration/);
    assert.match(container.innerHTML, /button disabled/);
    assert.ok(!container.innerHTML.includes('href='));
});

test('clipboard denial leaves a useful download alternative', async () => {
    const { ctx, button } = harness(async () => preview);
    ctx.navigator.clipboard.writeText = async () => { throw new Error('denied'); };
    await ctx.loadTrialPythonExport();
    await button.click({ currentTarget: button });
    assert.match(button.textContent, /download Python/);
});
