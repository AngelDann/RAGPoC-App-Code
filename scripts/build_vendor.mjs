#!/usr/bin/env node
/**
 * Rebuilds src/ragpoc/static/vendor/ -- every third-party asset the console UI needs.
 *
 * The UI used to pull these from cdn.jsdelivr.net, esm.sh and fonts.googleapis.com at runtime,
 * which meant a compiled desktop app could not draw its own window without internet. Worse than
 * a clean failure: the whole UI lives in one <script type="module">, and an ES module whose
 * imports fail never runs a single statement -- so the window opened, painted a fully styled
 * page, and then sat there completely inert, with nothing in ragpoc.log because the failure was
 * entirely browser-side. Everything is bundled into the build now (see ragpoc.spec's datas and
 * knowledge.views.vendor_view).
 *
 * Run from the repo root:  node scripts/build_vendor.mjs
 * Needs network access -- this is a build step, not something the app ever does at runtime.
 */
import { execFileSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const VENDOR = join(ROOT, 'src', 'ragpoc', 'static', 'vendor');
const WORK = join(ROOT, 'build', 'vendor-bundle');

// Every @tiptap/* package is pinned to one exact version, transitive ones included. Left to
// npm's own resolution the extensions starter-kit depends on float up to the newest 2.x while
// @tiptap/core stays at the requested 2.11.5, and the build dies on a mismatched internal API
// ("No matching export ... for import canInsertNode"). The overrides block is what keeps the
// whole TipTap tree on a single self-consistent version.
const TIPTAP = '2.11.5';
const TIPTAP_PACKAGES = [
  'core', 'pm', 'html', 'starter-kit',
  'extension-image', 'extension-link', 'extension-code-block-lowlight',
  'extension-table', 'extension-table-row', 'extension-table-header', 'extension-table-cell',
  'extension-blockquote', 'extension-bold', 'extension-bullet-list', 'extension-code',
  'extension-code-block', 'extension-document', 'extension-dropcursor', 'extension-gapcursor',
  'extension-hard-break', 'extension-heading', 'extension-history', 'extension-horizontal-rule',
  'extension-italic', 'extension-list-item', 'extension-ordered-list', 'extension-paragraph',
  'extension-strike', 'extension-text', 'extension-text-style',
];
// Direct dependencies only; the rest of the list above exists purely to be pinned.
const DIRECT = [
  'core', 'pm', 'html', 'starter-kit', 'extension-image', 'extension-link',
  'extension-code-block-lowlight', 'extension-table', 'extension-table-row',
  'extension-table-header', 'extension-table-cell',
];
const OTHER_DEPS = { lowlight: '3.1.0', marked: '15.0.7', dompurify: '3.2.4', esbuild: '^0.25.0' };

const LIBS = {
  'bootstrap/bootstrap.min.css': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css',
  'bootstrap/bootstrap.bundle.min.js': 'https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js',
  'bootstrap-icons/bootstrap-icons.min.css': 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css',
  'highlight/github-dark.min.css': 'https://cdn.jsdelivr.net/npm/highlight.js@11.9.0/styles/github-dark.min.css',
  'katex/katex.min.css': 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
  'katex/katex.min.js': 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js',
  // Pinned exactly, unlike the "mermaid@10" range the page used to request: a range means the
  // bundled copy silently changes version between builds.
  'mermaid/mermaid.min.js': 'https://cdn.jsdelivr.net/npm/mermaid@10.9.3/dist/mermaid.min.js',
  'chartjs/chart.umd.js': 'https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.js',
  'jspdf/jspdf.umd.min.js': 'https://cdn.jsdelivr.net/npm/jspdf@2.5.1/dist/jspdf.umd.min.js',
  'html2canvas/html2canvas.min.js': 'https://cdn.jsdelivr.net/npm/html2canvas@1.4.1/dist/html2canvas.min.js',
};
// KaTeX's and bootstrap-icons' stylesheets already reference their fonts relatively
// ("fonts/KaTeX_Main-Regular.woff2"), so the files just have to land beside them -- no rewriting.
const FONT_SOURCES = [
  ['katex/katex.min.css', 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/'],
  ['bootstrap-icons/bootstrap-icons.min.css', 'https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/'],
];
const GOOGLE_FONTS =
  'https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800' +
  '&family=JetBrains+Mono:wght@400;500&display=swap';
// Google Fonts serves TTF instead of woff2 unless the request looks like a modern browser.
const UA = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36';
// A sourceMappingURL pointing back at the CDN would quietly reintroduce the very network
// dependency being removed here, since devtools fetches it.
const SOURCEMAP = /\/([*/])# sourceMappingURL=.*?(\*\/|\n|$)/gs;

async function download(url) {
  const res = await fetch(url, { headers: { 'User-Agent': UA } });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText} -- ${url}`);
  return Buffer.from(await res.arrayBuffer());
}

function put(rel, data) {
  const dest = join(VENDOR, rel);
  mkdirSync(dirname(dest), { recursive: true });
  writeFileSync(dest, data);
  console.log(`  ${rel.padEnd(46)} ${String(data.length).padStart(9)} B`);
}

console.log(`Rebuilding ${VENDOR}\n`);
rmSync(VENDOR, { recursive: true, force: true });

// ---- 1. The ES module bundle (TipTap & co.) ------------------------------------------------
console.log('Bundling editor dependencies...');
mkdirSync(WORK, { recursive: true });
const dependencies = { ...OTHER_DEPS };
for (const p of DIRECT) dependencies[`@tiptap/${p}`] = TIPTAP;
const overrides = {};
for (const p of TIPTAP_PACKAGES) overrides[`@tiptap/${p}`] = TIPTAP;
writeFileSync(
  join(WORK, 'package.json'),
  JSON.stringify(
    { name: 'ragpoc-vendor-bundle', private: true, version: '1.0.0', type: 'module', dependencies, overrides },
    null,
    2,
  ),
);
// Same export names console.html's module expects, so its body needs no changes.
writeFileSync(
  join(WORK, 'entry.js'),
  [
    "export { Editor, Node, Extension } from '@tiptap/core';",
    "export { Plugin, PluginKey } from '@tiptap/pm/state';",
    "export { generateHTML } from '@tiptap/html';",
    "export { default as StarterKit } from '@tiptap/starter-kit';",
    "export { default as Image } from '@tiptap/extension-image';",
    "export { default as Link } from '@tiptap/extension-link';",
    "export { default as CodeBlockLowlight } from '@tiptap/extension-code-block-lowlight';",
    "export { default as Table } from '@tiptap/extension-table';",
    "export { default as TableRow } from '@tiptap/extension-table-row';",
    "export { default as TableHeader } from '@tiptap/extension-table-header';",
    "export { default as TableCell } from '@tiptap/extension-table-cell';",
    "export { createLowlight, common } from 'lowlight';",
    "export { marked } from 'marked';",
    "export { default as DOMPurify } from 'dompurify';",
    '',
  ].join('\n'),
);
// On Windows npm is a .cmd shim, and since Node 20 execFileSync refuses to spawn one directly
// (the CVE-2024-27980 fix for batch-file argument injection) -- it dies with a bare EINVAL.
// Running npm's own CLI script under this same node binary sidesteps the shim entirely, which
// beats re-enabling shell:true: that works but is deprecated (DEP0190) precisely because it
// concatenates arguments unescaped. Only the npm launch needs this; esbuild below is a real exe.
const npmCli = join(dirname(process.execPath), 'node_modules', 'npm', 'bin', 'npm-cli.js');
const [npmCmd, npmArgs] = existsSync(npmCli)
  ? [process.execPath, [npmCli]]
  : [process.platform === 'win32' ? 'npm.cmd' : 'npm', []];
execFileSync(npmCmd, [...npmArgs, 'install', '--no-audit', '--no-fund'], {
  cwd: WORK,
  stdio: 'inherit',
  shell: npmCmd.endsWith('.cmd'),
});
// esbuild's own binary, not the npm wrapper: the wrapper swallows compile errors behind a
// generic "Command failed" stack, which hides exactly the version-mismatch message above.
const esbuildBin = join(
  WORK, 'node_modules', '@esbuild', `${process.platform}-${process.arch}`,
  process.platform === 'win32' ? 'esbuild.exe' : 'bin/esbuild',
);
execFileSync(
  esbuildBin,
  ['entry.js', '--bundle', '--format=esm', '--minify', '--target=es2020', '--legal-comments=none',
    `--outfile=${join(VENDOR, 'app-deps.js')}`],
  { cwd: WORK, stdio: 'inherit' },
);
console.log(`  ${'app-deps.js'.padEnd(46)} ${String(readFileSync(join(VENDOR, 'app-deps.js')).length).padStart(9)} B`);

// ---- 2. Plain <script>/<link> libraries ----------------------------------------------------
console.log('\nLibraries:');
for (const [rel, url] of Object.entries(LIBS)) {
  put(rel, Buffer.from(String(await download(url)).replace(SOURCEMAP, '')));
}

// ---- 3. Fonts the stylesheets reference relatively -----------------------------------------
console.log('\nStylesheet fonts:');
for (const [cssRel, base] of FONT_SOURCES) {
  const css = readFileSync(join(VENDOR, cssRel), 'utf8');
  const refs = new Set(
    [...css.matchAll(/url\(([^)]+)\)/g)]
      .map((m) => m[1].split('?')[0].replace(/^["']|["']$/g, '').replace(/^\.\//, ''))
      .filter((r) => r && !r.startsWith('data:') && !r.startsWith('http')),
  );
  for (const ref of [...refs].sort()) {
    put(`${dirname(cssRel)}/${ref}`.replaceAll('\\', '/'), await download(base + ref));
  }
}

// ---- 4. Google Fonts -- the one stylesheet with absolute URLs to rewrite --------------------
console.log('\nGoogle Fonts:');
let fontCss = String(await download(GOOGLE_FONTS));
const remote = [...new Set(
  [...fontCss.matchAll(/url\((https:\/\/fonts\.gstatic\.com\/[^)]+)\)/g)].map((m) => m[1]),
)].sort();
for (const [i, url] of remote.entries()) {
  const name = `${url.split('/').pop().split('.')[0]}-${i}.woff2`;
  put(`fonts/${name}`, await download(url));
  fontCss = fontCss.replaceAll(url, name);
}
put('fonts/fonts.css', Buffer.from(fontCss));

const leftover = fontCss.match(/https?:\/\//g);
if (leftover) throw new Error(`fonts.css still has ${leftover.length} remote URL(s)`);
console.log('\nDone. No runtime network dependency remains in the console UI.');
