/**
 * Render worker driven by the Rust orchestrator (src/stages/render.rs).
 *
 * Bundles the composition once, opens headless Chromium once, then renders
 * every manifest_*.json in --manifests-dir to --out-dir reusing the browser
 * instance (PRD §4 warm process). Never aborts on a single failed clip; prints
 * a JSON summary on stdout that Rust parses.
 *
 *   tsx src/render-worker.ts --manifests-dir <dir> --out-dir <dir>
 */

import fs from "node:fs";
import path from "node:path";
import {fileURLToPath} from "node:url";
import {bundle} from "@remotion/bundler";
import {openBrowser, renderMedia, selectComposition} from "@remotion/renderer";
import type {RenderManifest} from "./lib/manifest";

function parseArgs(argv: string[]) {
  const args: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const next = argv[i + 1];
      if (next !== undefined && !next.startsWith("--")) {
        args[key] = next;
        i++;
      }
    }
  }
  return args;
}

async function main() {
  const {["manifests-dir"]: manifestsDir, ["out-dir"]: outDir} = parseArgs(process.argv.slice(2));
  if (!manifestsDir || !outDir) {
    throw new Error("usage: render-worker.ts --manifests-dir <dir> --out-dir <dir>");
  }

  fs.mkdirSync(outDir, {recursive: true});

  const manifestPaths = fs
    .readdirSync(manifestsDir)
    .filter((f) => f.startsWith("manifest_") && f.endsWith(".json"))
    .sort()
    .map((f) => path.join(manifestsDir, f));

  const rendered: string[] = [];
  const failed: {clip_id: string; error: string}[] = [];

  const finish = (): void => {
    const summary = JSON.stringify({rendered, failed});
    // Stdout is polluted by Remotion/bundler logs, so persist the summary to a
    // file the Rust orchestrator reads instead of parsing stdout.
    fs.writeFileSync(path.join(outDir, "render-summary.json"), summary);
    console.log(summary);
  };

  if (manifestPaths.length === 0) {
    finish();
    return;
  }

  const serveUrl = await bundle({
    entryPoint: path.resolve(path.dirname(fileURLToPath(import.meta.url)), "index.ts"),
    onProgress: () => undefined,
  });

  const browserInstance = await openBrowser("chrome", {
    chromiumOptions: {headless: true},
  });

  try {
    for (const p of manifestPaths) {
      const manifest: RenderManifest = JSON.parse(fs.readFileSync(p, "utf8"));
      try {
        const composition = await selectComposition({
          serveUrl,
          puppeteerInstance: browserInstance,
          id: "AIClip",
          inputProps: {manifest},
        });
        const outPath = path.join(outDir, `${manifest.clip_id}.mp4`);
        await renderMedia({
          composition,
          serveUrl,
          puppeteerInstance: browserInstance,
          codec: "h264",
          audioCodec: "aac",
          outputLocation: outPath,
          concurrency: 2,
          onProgress: () => undefined,
        });
        rendered.push(manifest.clip_id);
      } catch (err) {
        failed.push({
          clip_id: manifest.clip_id,
          error: err instanceof Error ? err.message : String(err),
        });
      }
    }
  } finally {
    await browserInstance.close({silent: true});
  }

  finish();
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});