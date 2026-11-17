const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const state = {
  lastResult: null,
  recent: [],
  stats: null,
  statsLoading: false,
  health: null,
  examplesLoaded: false,
};

const TIER_LABEL = { SPAM: "spam", SUSPICIOUS: "suspicious", NORMAL: "normal" };
const RISK_TEXT = { high: "high risk", review: "review", low: "low risk" };
const FALLBACK_EXAMPLES = [
  "Give me $5 bucks I will give you an RTX 3090",
  "URGENT: Your bKash account is suspended. Click http://bkash-verify.xyz to reactivate.",
  "আপনি ৫,০০,০০০ টাকা জিতেছেন! পুরস্কার পেতে আপনার বিকাশ নম্বর পাঠান।",
  "Aapka KYC pending hai, turant is link par click karke update karein: bit[.]ly/kyc-update",
  "Bro can you send me $5 for lunch? I'll pay you back tomorrow.",
  "Your OTP is 482913. Do not share it with anyone.",
];

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? new Intl.NumberFormat().format(number) : "—";
}

function formatPct(value, digits = 1) {
  const number = Number(value);
  return Number.isFinite(number) ? `${(number * 100).toFixed(digits)}%` : "—";
}

function clamp01(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return 0;
  return Math.max(0, Math.min(1, number));
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let payload = null;
  try {
    payload = text ? JSON.parse(text) : null;
  } catch {
    payload = { detail: text };
  }
  if (!response.ok) {
    const detail = payload?.detail || payload?.error || text || `Request failed (${response.status})`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return payload;
}

function setStatus(kind, text) {
  const pill = $("#statusPill");
  pill.classList.remove("ok", "bad");
  if (kind === "ok") pill.classList.add("ok");
  if (kind === "bad") pill.classList.add("bad");
  $("#statusText").textContent = text;
}

function setLoading(button, loading) {
  if (!button) return;
  button.classList.toggle("loading", loading);
  button.disabled = loading;
}

function showError(message) {
  const box = $("#errorState");
  box.textContent = message;
  box.classList.remove("hidden");
  $("#emptyState").classList.add("hidden");
  $("#resultContent").classList.add("hidden");
}

function hideError() {
  $("#errorState").classList.add("hidden");
}

function updateCharCount() {
  const input = $("#messageInput");
  $("#charCount").textContent = `${input.value.length} / 4000`;
}

function updateBatchCount() {
  const lines = $("#batchInput").value.split("\n").map((line) => line.trim()).filter(Boolean);
  $("#batchCount").textContent = `${lines.length} message${lines.length === 1 ? "" : "s"}`;
}

function renderExamples(examples) {
  const row = $("#samples");
  row.innerHTML = "";
  examples.slice(0, 8).forEach((example) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "sample-chip";
    button.title = example;
    button.textContent = example.length > 68 ? `${example.slice(0, 66)}…` : example;
    button.addEventListener("click", () => {
      $("#messageInput").value = example;
      updateCharCount();
      analyzeText();
    });
    row.appendChild(button);
  });
}

function renderRecent() {
  const row = $("#recent");
  row.innerHTML = "";
  state.recent.slice(0, 5).forEach((entry) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "recent-chip";
    button.title = entry.text;
    button.textContent = entry.text.length > 32 ? `${entry.text.slice(0, 30)}…` : entry.text;
    button.addEventListener("click", () => {
      $("#messageInput").value = entry.text;
      updateCharCount();
      analyzeText();
    });
    row.appendChild(button);
  });
}

function addRecent(text) {
  state.recent = [{ text }, ...state.recent.filter((item) => item.text !== text)].slice(0, 5);
  renderRecent();
}

function scoreCard(label, value, barValue, suffix = "") {
  const display = value === null || value === undefined || Number.isNaN(Number(value))
    ? "—"
    : `${Number(value).toFixed(3)}${suffix}`;
  return `
    <div class="score-card">
      <span class="score-label">${escapeHtml(label)}</span>
      <strong class="score-value">${escapeHtml(display)}</strong>
      <div class="score-bar"><span style="width:${(clamp01(barValue) * 100).toFixed(1)}%"></span></div>
    </div>`;
}

function renderResult(data) {
  hideError();
  $("#emptyState").classList.add("hidden");
  $("#resultContent").classList.remove("hidden");
  const card = $("#resultCard");
  card.dataset.tier = TIER_LABEL[data.label] || "normal";
  const probability = clamp01(data.spam_probability);
  $("#gauge").style.setProperty("--value", (probability * 100).toFixed(2));
  $("#probValue").textContent = formatPct(probability);
  $("#verdictLabel").textContent = data.label;
  $("#riskBadge").textContent = RISK_TEXT[data.risk_tier] || "risk";
  if (data.decision?.source === "rule_escalation") {
    $("#riskBadge").textContent = "high risk · rule";
  } else if (data.decision?.source === "rule_review") {
    $("#riskBadge").textContent = "review · rule";
  }

  const thresholdText = `SUSPICIOUS ≥ ${Number(data.thresholds?.suspicious ?? 0).toFixed(2)} · SPAM ≥ ${Number(data.thresholds?.spam ?? 0).toFixed(2)}`;
  const sub = data.decision?.source === "rule_escalation"
    ? "A high-confidence scam rule escalated this message. Verify the sender before clicking, paying, or sharing a code."
    : data.decision?.source === "rule_review"
      ? `An emergency-money review rule placed this message in the review band. ${thresholdText}.`
      : data.label === "SPAM"
        ? "The calibrated blend crossed the block threshold. Treat links, payment requests, and OTP prompts as hostile until verified."
        : data.label === "SUSPICIOUS"
          ? `This message sits in the review band. ${thresholdText}.`
          : `Below the review threshold. ${thresholdText}.`;
  $("#verdictSub").textContent = sub;
  $("#languageBadge").textContent = `${data.language || "unknown"}${data.language_confidence ? ` · ${(data.language_confidence * 100).toFixed(0)}%` : ""}`;
  $("#latencyBadge").textContent = `${Number(data.latency_ms || 0).toFixed(1)} ms`;
  $("#versionBadge").textContent = `v${data.model_version || "1.0.0"}`;

  $("#probFill").style.width = `${(probability * 100).toFixed(2)}%`;
  $("#suspiciousMarker").style.left = `${clamp01(data.thresholds?.suspicious) * 100}%`;
  $("#spamMarker").style.left = `${clamp01(data.thresholds?.spam) * 100}%`;

  const scores = data.scores || {};
  $("#scoreGrid").innerHTML = [
    scoreCard("Lexical LR", scores.lexical_logistic, scores.lexical_logistic),
    scoreCard("Semantic", scores.semantic, scores.semantic ?? 0),
    scoreCard("Heuristic", scores.heuristic, scores.heuristic),
    scoreCard("SVM margin", scores.lexical_svm, Math.max(0, Math.min(1, Number(scores.lexical_svm) || 0))),
  ].join("");

  const signals = Array.isArray(data.detected_signals) ? data.detected_signals : [];
  $("#signalCount").textContent = `${signals.length} signal${signals.length === 1 ? "" : "s"}`;
  $("#signals").innerHTML = signals.length
    ? signals.slice(0, 10).map((signal) => {
        const severity = ["high", "medium", "low"].includes(signal.severity) ? signal.severity : "low";
        const evidence = Array.isArray(signal.evidence)
          ? signal.evidence.slice(0, 3).map((item) => escapeHtml(item)).join(" · ")
          : "";
        return `
          <div class="signal ${severity}">
            <div class="signal-top">
              <span class="signal-title">${escapeHtml(signal.title || signal.name || "Signal")}</span>
              <span class="signal-sev">${escapeHtml(severity)}</span>
            </div>
            ${evidence ? `<div class="signal-evidence">${evidence}</div>` : ""}
          </div>`;
      }).join("")
    : `<div class="signal-empty">No strong heuristic signals. The verdict is driven by the calibrated model blend.</div>`;

  const cues = Array.isArray(data.model_cues) ? data.model_cues : [];
  $("#cues").innerHTML = cues.length
    ? cues.slice(0, 8).map((cue) => `
        <span class="cue">${escapeHtml(cue.feature)} <b>${Number(cue.contribution) >= 0 ? "+" : ""}${Number(cue.contribution).toFixed(3)}</b></span>
      `).join("")
    : `<span class="cue-empty">No strong positive lexical cues.</span>`;

  const semanticBox = $("#semanticMatch");
  if (data.semantic_match) {
    semanticBox.classList.remove("hidden");
    semanticBox.innerHTML = `<strong>Semantic match ${Number(data.semantic_match.similarity).toFixed(3)}</strong> — ${escapeHtml(data.semantic_match.prototype)}`;
  } else {
    semanticBox.classList.add("hidden");
    semanticBox.innerHTML = "";
  }

  $("#explanation").textContent = data.explanation || "";
  state.lastResult = data;
}

async function analyzeText() {
  const input = $("#messageInput");
  const text = input.value.trim();
  if (!text) {
    showError("Enter a message to analyze.");
    input.focus();
    return;
  }
  hideError();
  const button = $("#analyzeBtn");
  setLoading(button, true);
  try {
    const data = await fetchJSON("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    renderResult(data);
    addRecent(text);
  } catch (error) {
    showError(error.message || "Prediction failed.");
  } finally {
    setLoading(button, false);
  }
}

async function batchAnalyze() {
  const input = $("#batchInput");
  const messages = input.value.split("\n").map((line) => line.trim()).filter(Boolean).slice(0, 256);
  if (!messages.length) {
    $("#batchSummary").classList.remove("hidden");
    $("#batchSummary").innerHTML = `<span class="summary-pill">Enter at least one message.</span>`;
    return;
  }
  const button = $("#batchBtn");
  setLoading(button, true);
  try {
    const data = await fetchJSON("/predict_batch", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages }),
    });
    const summary = data.summary || {};
    $("#batchSummary").classList.remove("hidden");
    $("#batchSummary").innerHTML = `
      <span class="summary-pill">${data.count} messages</span>
      <span class="summary-pill spam">${summary.spam || 0} spam</span>
      <span class="summary-pill suspicious">${summary.suspicious || 0} review</span>
      <span class="summary-pill normal">${summary.normal || 0} normal</span>
      <span class="summary-pill">${Number(summary.mean_latency_ms || 0).toFixed(1)} ms / message</span>
    `;
    const tbody = $("#batchTable tbody");
    tbody.innerHTML = "";
    data.predictions.forEach((prediction, index) => {
      const source = messages[index] || "";
      const topSignal = prediction.detected_signals?.[0]?.title || "—";
      const tier = TIER_LABEL[prediction.label] || "normal";
      const row = document.createElement("tr");
      row.innerHTML = `
        <td>${index + 1}</td>
        <td><div class="msg" title="${escapeHtml(source)}">${escapeHtml(source)}</div></td>
        <td><span class="tag ${tier}">${escapeHtml(prediction.label)}</span></td>
        <td>${Number(prediction.spam_probability).toFixed(3)}</td>
        <td>${escapeHtml(prediction.language)}</td>
        <td>${escapeHtml(topSignal)}</td>
      `;
      tbody.appendChild(row);
    });
  } catch (error) {
    $("#batchSummary").classList.remove("hidden");
    $("#batchSummary").innerHTML = `<span class="summary-pill">${escapeHtml(error.message || "Batch failed.")}</span>`;
    $("#batchTable tbody").innerHTML = "";
  } finally {
    setLoading(button, false);
  }
}

function detailRow(label, value) {
  return `<div class="detail-row"><span>${escapeHtml(label)}</span><span>${escapeHtml(value)}</span></div>`;
}

function metricCard(label, value) {
  return `<div class="metric-card"><span>${escapeHtml(value)}</span><small>${escapeHtml(label)}</small></div>`;
}

function renderModel(stats) {
  state.stats = stats;
  const model = stats.model || {};
  const thresholds = stats.thresholds || {};
  const semantic = stats.semantic;
  const lexical = model.lexical || {};
  const evaluation = stats.evaluation || {};

  $("#modelBackend").textContent = semantic ? semantic.backend : "lexical only";
  $("#modelDetails").innerHTML = [
    detailRow("Model type", model.model_type || "hybrid lexical + semantic"),
    detailRow("Multilingual backbone", semantic?.model_name || "not loaded"),
    detailRow("Semantic embedding", semantic?.embedding_dim ? `${semantic.embedding_dim} dims` : "not loaded"),
    detailRow("Lexical features", lexical.feature_count ? formatNumber(lexical.feature_count) : "—"),
    detailRow("Best lexical C", lexical.best_C ?? "—"),
    detailRow("Training rows", stats.dataset?.label_by_split?.train?.rows ? formatNumber(stats.dataset.label_by_split.train.rows) : "—"),
    detailRow("Spam threshold", thresholds.spam_threshold ? Number(thresholds.spam_threshold).toFixed(4) : "—"),
    detailRow("Review threshold", thresholds.suspicious_threshold ? Number(thresholds.suspicious_threshold).toFixed(4) : "—"),
    detailRow("Calibration", semantic?.backend === "pyodide-lexical" ? "lexical LR probability · precision-constrained threshold" : "validation stacker · 20% prior"),
  ].join("");

  const evalBlock = evaluation.test_realistic || evaluation.test || {};
  const metrics = evalBlock.metrics || {};
  const cards = [
    ["PR-AUC", metrics.pr_auc],
    ["ROC-AUC", metrics.roc_auc],
    ["F1", metrics.f1],
    ["Recall", metrics.recall],
    ["Precision", metrics.precision],
    ["FPR", metrics.fpr],
    ["Brier", metrics.brier],
    ["Accuracy", metrics.accuracy],
  ];
  $("#metricGrid").innerHTML = cards.map(([label, value]) => {
    let display = "—";
    if (value !== undefined && value !== null && Number.isFinite(Number(value))) {
      display = label === "PR-AUC" || label === "ROC-AUC" || label === "F1" || label === "Recall" || label === "Precision" || label === "Accuracy"
        ? formatPct(value)
        : label === "FPR"
          ? formatPct(value, 2)
          : Number(value).toFixed(3);
    }
    return metricCard(label, display);
  }).join("");
  $("#evalNote").textContent = evalBlock.metrics
    ? `Held-out ${evalBlock.prior ? `${(evalBlock.prior * 100).toFixed(0)}% spam prior` : "test"} subset. Threshold tuned at FPR ≤ 1% on a disjoint validation split.`
    : "Run `python -m src.evaluate` to populate held-out metrics.";

  const languageData = stats.dataset?.language_by_split?.train || {};
  const rows = Object.entries(languageData).sort((a, b) => b[1] - a[1]).slice(0, 8);
  const max = rows.length ? Math.max(...rows.map(([, count]) => Number(count) || 0)) : 1;
  $("#datasetBars").innerHTML = rows.length
    ? rows.map(([language, count]) => `
        <div class="dataset-row">
          <span>${escapeHtml(language)}</span>
          <div class="bar"><span style="width:${((Number(count) || 0) / max * 100).toFixed(1)}%"></span></div>
          <b>${formatNumber(count)}</b>
        </div>`).join("")
    : `<div class="muted">Dataset statistics are unavailable.</div>`;

  const trainRows = stats.dataset?.label_by_split?.train?.rows;
  const languageCount = Object.values(languageData).filter((n) => Number(n) > 0).length;
  if (trainRows) $("#metricRows").textContent = formatNumber(trainRows);
  if (languageCount) $("#metricLanguages").textContent = String(languageCount);
  const latency = evalBlock.latency_ms ?? evalBlock.metrics?.mean_latency_ms;
  if (latency) $("#metricLatency").textContent = `${Number(latency).toFixed(1)} ms`;
}

async function loadStats(silent = false) {
  if (state.statsLoading) return;
  state.statsLoading = true;
  try {
    const stats = await fetchJSON("/stats");
    renderModel(stats);
    if (!silent && !stats.evaluation) {
      $("#evalNote").textContent = "Run `python -m src.evaluate` to populate held-out metrics.";
    }
  } catch (error) {
    if (!silent) {
      $("#modelBackend").textContent = "unavailable";
      $("#modelDetails").innerHTML = `<div class="muted">${escapeHtml(error.message || "Stats unavailable.")}</div>`;
    }
  } finally {
    state.statsLoading = false;
  }
}

async function loadHealth() {
  try {
    const health = await fetchJSON("/health");
    state.health = health;
    setStatus("ok", `Model ready · ${health.semantic_backend === "disabled" ? "lexical" : health.semantic_backend}`);
    if (health.thresholds) {
      const spam = Number(health.thresholds.spam_threshold);
      const suspicious = Number(health.thresholds.suspicious_threshold);
      if (Number.isFinite(spam) && Number.isFinite(suspicious)) {
        $("#metricLatency").textContent = health.semantic_backend === "onnx-int8" ? "< 15 ms" : "< 30 ms";
      }
    }
  } catch (error) {
    setStatus("bad", "Model unavailable");
    showError(error.message || "The detector service is unavailable. Start the FastAPI server and reload.");
  }
}

async function loadExamples() {
  if (state.examplesLoaded) return;
  try {
    const data = await fetchJSON("/examples");
    renderExamples(Array.isArray(data.examples) && data.examples.length ? data.examples : FALLBACK_EXAMPLES);
  } catch {
    renderExamples(FALLBACK_EXAMPLES);
  }
  state.examplesLoaded = true;
}

function activateTab(name) {
  $$(".tab").forEach((tab) => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", String(active));
  });
  $$(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === `tab-${name}`);
  });
  if (name === "model") loadStats(false);
}

function bindEvents() {
  $$(".tab").forEach((tab) => tab.addEventListener("click", () => activateTab(tab.dataset.tab)));

  $("#messageInput").addEventListener("input", updateCharCount);
  $("#batchInput").addEventListener("input", updateBatchCount);
  $("#analyzeBtn").addEventListener("click", analyzeText);
  $("#clearBtn").addEventListener("click", () => {
    $("#messageInput").value = "";
    updateCharCount();
    $("#resultContent").classList.add("hidden");
    $("#emptyState").classList.remove("hidden");
    hideError();
    $("#messageInput").focus();
  });
  $("#retryBtn").addEventListener("click", () => $("#messageInput").focus());
  $("#copyBtn").addEventListener("click", async () => {
    if (!state.lastResult) return;
    try {
      await navigator.clipboard.writeText(JSON.stringify(state.lastResult, null, 2));
      const button = $("#copyBtn");
      const previous = button.textContent;
      button.textContent = "Copied";
      setTimeout(() => { button.textContent = previous; }, 1400);
    } catch {
      showError("Clipboard access was blocked by the browser.");
    }
  });
  $("#batchBtn").addEventListener("click", batchAnalyze);
  $("#batchClear").addEventListener("click", () => {
    $("#batchInput").value = "";
    updateBatchCount();
    $("#batchSummary").classList.add("hidden");
    $("#batchTable tbody").innerHTML = "";
  });
  document.addEventListener("keydown", (event) => {
    if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
      event.preventDefault();
      if ($("#tab-analyze").classList.contains("active")) analyzeText();
    }
  });
}

async function init() {
  bindEvents();
  updateCharCount();
  updateBatchCount();
  await Promise.all([loadHealth(), loadExamples()]);
  loadStats(true);
}

document.addEventListener("DOMContentLoaded", init);
