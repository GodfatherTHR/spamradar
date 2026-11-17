(function () {
  const PYODIDE_URL = "https://cdn.jsdelivr.net/pyodide/v0.28.3/full/";
  const realFetch = window.fetch.bind(window);
  let ready = null;
  let stats = null;
  let health = null;
  let examples = null;
  let pyodide = null;

  const overlay = () => document.getElementById("bootOverlay");
  const bootText = () => document.getElementById("bootText");
  function setBoot(message, isError = false) {
    const text = bootText();
    if (text) {
      text.textContent = message;
      text.classList.toggle("boot-error", isError);
    }
    const card = overlay()?.querySelector(".boot-card");
    if (card) card.classList.toggle("boot-failed", isError);
  }
  function hideBoot() {
    overlay()?.classList.add("hidden");
  }
  function jsonResponse(payload, status = 200) {
    return new Response(JSON.stringify(payload), {
      status,
      headers: { "Content-Type": "application/json; charset=utf-8" },
    });
  }
  function writeText(fs, path, text) {
    const parts = path.split("/");
    parts.pop();
    if (parts.length) fs.mkdirTree(parts.join("/"));
    fs.writeFile(path, new TextEncoder().encode(text));
  }
  function writeBytes(fs, path, bytes) {
    const parts = path.split("/");
    parts.pop();
    if (parts.length) fs.mkdirTree(parts.join("/"));
    fs.writeFile(path, new Uint8Array(bytes));
  }
  async function fetchText(path) {
    const response = await realFetch(path, { cache: "force-cache" });
    if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
    return response.text();
  }
  async function fetchBytes(path) {
    const response = await realFetch(path, { cache: "force-cache" });
    if (!response.ok) throw new Error(`Could not load ${path} (${response.status})`);
    return response.arrayBuffer();
  }

  async function boot() {
    if (typeof window.loadPyodide !== "function") {
      throw new Error("The Python runtime failed to load. Check the network and reload.");
    }
    setBoot("Starting the Python runtime...");
    pyodide = await window.loadPyodide({ indexURL: PYODIDE_URL });
    setBoot("Loading the numeric runtime...");
    await pyodide.loadPackage("numpy");
    setBoot("Downloading the detector model...");
    const [modelBytes, thresholds, statsPayload, healthPayload, examplesPayload] = await Promise.all([
      fetchBytes("browser_model.json.gz"),
      fetchText("browser_thresholds.json"),
      fetchText("browser_stats.json"),
      fetchText("browser_health.json"),
      fetchText("browser_examples.json"),
    ]);
    stats = JSON.parse(statsPayload);
    health = JSON.parse(healthPayload);
    examples = JSON.parse(examplesPayload);
    setBoot("Preparing the detector...");
    const files = [
      ["browser_model.py", await fetchText("browser_model.py")],
      ["src/__init__.py", await fetchText("src/__init__.py")],
      ["src/features.py", await fetchText("src/features.py")],
      ["src/utils.py", await fetchText("src/utils.py")],
      ["browser_thresholds.json", thresholds],
    ];
    writeBytes(pyodide.FS, "/browser_model.json.gz", modelBytes);
    for (const [path, text] of files) writeText(pyodide.FS, `/${path}`, text);
    pyodide.runPython(`
import sys
sys.path.insert(0, "/")
from browser_model import BrowserModel
model = BrowserModel()
`);
    setBoot("Running a warmup prediction...");
    pyodide.globals.set("warmup_json", JSON.stringify({ text: "hello, how are you?" }));
    pyodide.runPython("model.predict_payload_json(warmup_json)");
    setBoot("Ready");
    setTimeout(hideBoot, 350);
    return true;
  }

  function predict(text) {
    pyodide.globals.set("payload_json", JSON.stringify({ text: String(text || "") }));
    return JSON.parse(pyodide.runPython("model.predict_payload_json(payload_json)"));
  }

  function predictBatch(messages) {
    pyodide.globals.set("payload_json", JSON.stringify({ messages }));
    return JSON.parse(pyodide.runPython("model.batch_payload_json(payload_json)"));
  }

  window.fetch = async function (input, init = {}) {
    const requestUrl = typeof input === "string" ? input : input.url;
    const url = new URL(requestUrl, window.location.href);
    const method = String(init.method || (typeof input === "string" ? "GET" : input.method) || "GET").toUpperCase();
    if (url.pathname === "/predict" && method === "POST") {
      try {
        await ready;
        const body = JSON.parse(init.body || "{}");
        return jsonResponse(predict(body.text));
      } catch (error) {
        return jsonResponse({ detail: String(error?.message || error) }, 503);
      }
    }
    if (url.pathname === "/predict_batch" && method === "POST") {
      try {
        await ready;
        const body = JSON.parse(init.body || "{}");
        return jsonResponse(predictBatch(Array.isArray(body.messages) ? body.messages : []));
      } catch (error) {
        return jsonResponse({ detail: String(error?.message || error) }, 503);
      }
    }
    if (url.pathname === "/health") {
      try {
        await ready;
        return jsonResponse(health);
      } catch (error) {
        return jsonResponse({ detail: String(error?.message || error) }, 503);
      }
    }
    if (url.pathname === "/stats") {
      try {
        await ready;
        return jsonResponse(stats);
      } catch (error) {
        return jsonResponse({ detail: String(error?.message || error) }, 503);
      }
    }
    if (url.pathname === "/examples") {
      try {
        await ready;
        return jsonResponse(examples);
      } catch (error) {
        return jsonResponse({ detail: String(error?.message || error) }, 503);
      }
    }
    return realFetch(input, init);
  };

  ready = boot();
  ready.catch((error) => {
    console.error(error);
    setBoot(`Model failed to load: ${error?.message || error}`, true);
  });
})();
