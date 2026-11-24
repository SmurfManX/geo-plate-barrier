// Screenshot the UI in headless Chrome with a fake camera (no real webcam needed).
// node tools/screenshot.mjs <url> <video.y4m> <out.png> <waitMs> [js-to-run-after-load] [file-for-#file-input]
import { spawn } from "node:child_process";
import { writeFileSync, rmSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const [, , url, y4m, out, waitMs, js = "", upload = ""] = process.argv;
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const profile = mkdtempSync(join(tmpdir(), "pb-shot-"));
const port = 9300 + Math.floor(Math.random() * 500);
const chrome = spawn(CHROME, [
  "--headless=new", "--disable-gpu", "--no-first-run", `--user-data-dir=${profile}`, `--remote-debugging-port=${port}`,
  "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${y4m}`,
  "--autoplay-policy=no-user-gesture-required", "--hide-scrollbars", url], { stdio: "ignore" });
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

let targets = [];
for (let i = 0; i < 50 && !targets.some((t) => t.type === "page"); i++) {
  try { targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json(); } catch {}
  await sleep(200);
}
const ws = new WebSocket(targets.find((t) => t.type === "page").webSocketDebuggerUrl);
await new Promise((r) => ws.addEventListener("open", r));
let id = 0;
const pending = new Map();
ws.addEventListener("message", (m) => {
  const d = JSON.parse(m.data);
  if (d.id && pending.has(d.id)) { pending.get(d.id)(d); pending.delete(d.id); }
});
const send = (method, params = {}) => new Promise((r) => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });

await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: 1000, deviceScaleFactor: 1, mobile: false });
await send("Page.reload");
await sleep(1500);
if (js) await send("Runtime.evaluate", { expression: js, awaitPromise: true });
if (upload) {
  const doc = await send("DOM.getDocument");
  const node = await send("DOM.querySelector", { nodeId: doc.result.root.nodeId, selector: "#file" });
  await send("DOM.setFileInputFiles", { nodeId: node.result.nodeId, files: [upload] });
}
await sleep(+waitMs);
const h = await send("Runtime.evaluate", { expression: "document.documentElement.scrollHeight", returnByValue: true });
await send("Emulation.setDeviceMetricsOverride", { width: 1440, height: Math.max(1000, h.result.result.value), deviceScaleFactor: 1, mobile: false });
await sleep(300);
const shot = await send("Page.captureScreenshot", { format: "png" });
writeFileSync(out, Buffer.from(shot.result.data, "base64"));
const exited = new Promise((r) => chrome.once("exit", r));
chrome.kill();
await Promise.race([exited, sleep(3000)]);
try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); } catch {}
process.exit(0);
