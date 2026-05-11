/**
 * RESC-AI-LING · Two-Page Frontend
 * Fixes:
 *  1. Radar chart normalized by SCALE_MAX (0–1), raw values in tooltip
 *  2. Ensemble table shows Score / Max / %
 *  3. Features Inference: benchmarks always manual, no structural inputs
 *  4. Inputs use type="text" to avoid browser scientific notation issues
 */

const API = "http://localhost:8000";

// ── Constants ─────────────────────────────────────────────────────────────
const COLORS_A = {
  reasoning_score: "#4E79A7",
  musr:            "#59A14F",
  co2:             "#F28E2B",
  math:            "#9B59B6",
  gpqa:            "#E05C5C",
  ifeval:          "#17BECF",
};
const LABELS_SHORT = {
  reasoning_score: "Reasoning",
  musr:  "MuSR",
  co2:   "CO₂",
  math:  "MATH",
  gpqa:  "GPQA",
  ifeval:"IFEval",
};
const LABELS_FULL = {
  reasoning_score: "Reasoning Score (BBH+MMLU-PRO)",
  musr:            "MuSR (Multi-step Reasoning)",
  co2:             "CO₂ Training Cost (log)",
  math:            "MATH (Logic & Maths)",
  gpqa:            "GPQA (Expert Knowledge)",
  ifeval:          "IFEval (Instruction Following)",
};
// Maximum possible score per target — used for radar normalisation
const SCALE_MAX = {
  reasoning_score: 70,
  musr:            25,
  co2:             25,
  math:            80,
  gpqa:            60,
  ifeval:          90,
};
const ORDER_A = ["reasoning_score","musr","co2","math","gpqa","ifeval"];

const REG_COLORS = {
  "Parameters (B)": "#4E79A7",
  "Training FLOPs": "#F28E2B",
  "Dataset Size":   "#59A14F",
};
const REG_ICONS = {
  "Parameters (B)": "⚙️",
  "Training FLOPs": "⚡",
  "Dataset Size":   "🗄️",
};

let radarChart = null;
let regCharts  = {};

const $ = id => document.getElementById(id);

// ── Robust number parser (handles "3e23", "3E+23", "300000" etc.) ─────────
function parseNum(str) {
  const v = parseFloat(String(str).replace(/,/g, "").trim());
  return isNaN(v) ? null : v;
}

// ── PAGE NAVIGATION ───────────────────────────────────────────────────────
document.querySelectorAll(".nav-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    const pid = `page${btn.dataset.page.charAt(0).toUpperCase()}${btn.dataset.page.slice(1)}`;
    document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
    $(pid).classList.add("active");
  });
});

// ── CHIPS ─────────────────────────────────────────────────────────────────
document.querySelectorAll(".chip").forEach(c => {
  c.addEventListener("click", () => {
    const inp = c.dataset.input;
    $(inp).value = c.dataset.val;
    document.querySelectorAll(`.chip[data-input="${inp}"]`).forEach(x => x.classList.remove("active"));
    c.classList.add("active");
  });
});

// ── HEALTH CHECK ──────────────────────────────────────────────────────────
async function checkHealth() {
  try {
    const r = await fetch(`${API}/health`, { signal: AbortSignal.timeout(4000), cache: "no-store" });
    if (!r.ok) throw new Error();
    const d = await r.json();
    $("statusDot").className = "status-dot ok";
    $("statusLabel").textContent =
      `API online · A:${d.group_a_targets_loaded} B:${d.group_b_regressors_loaded}+${d.group_b_classifiers_loaded}`;
  } catch {
    $("statusDot").className = "status-dot error";
    $("statusLabel").textContent = "API offline";
  }
}
checkHealth();
setInterval(checkHealth, 30_000);

// ══════════════════════════════════════════════════════════
//  PAGE 1 — BENCHMARK INFERENCE
// ══════════════════════════════════════════════════════════
$("runBenchmarkBtn").addEventListener("click", runBenchmark);

async function runBenchmark() {
  $("benchmarkError").classList.add("hidden");

  // Parse all inputs with robust parser
  const flops  = parseNum($("a_flops").value);
  const params = parseNum($("a_params").value);
  const ds     = parseNum($("a_dataset").value);

  if (!flops  || flops  <= 0) { showErr("benchmarkError", "Training FLOPs: invalid value (e.g. 3e23)");  return; }
  if (!params || params <= 0) { showErr("benchmarkError", "Parameters: invalid value (e.g. 7e9)");       return; }
  if (!ds     || ds     <= 0) { showErr("benchmarkError", "Dataset Size: invalid value (e.g. 1e12)");    return; }

  const payload = {
    training_flops:      flops,
    parameters:          params,
    dataset_size:        ds,
    architecture:        $("a_arch").value.trim()  || "Qwen2ForCausalLM",
    model_type:          $("a_mtype").value,
    organization_type:   $("a_org").value,
    override_benchmarks: false,
  };

  // Show what was actually sent
  showSent("featTableA", payload);
  $("featBlockA").classList.remove("hidden");

  setLoading("runBenchmarkBtn", "spinA", true);
  setState("benchmark", "loader");

  try {
    const r = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    renderBenchmark(data.group_a || {});
    setState("benchmark", "results");
  } catch (err) {
    showErr("benchmarkError", err.message || "Unexpected error.");
    setState("benchmark", "placeholder");
  } finally {
    setLoading("runBenchmarkBtn", "spinA", false);
  }
}

// Show exactly what was sent to the API
function showSent(tableId, payload) {
  const tbl = $(tableId);
  tbl.innerHTML = "";
  const display = {
    training_flops:    fmtLargeNum(payload.training_flops),
    parameters:        fmtLargeNum(payload.parameters),
    dataset_size:      fmtLargeNum(payload.dataset_size),
    architecture:      payload.architecture,
    model_type:        payload.model_type,
    organization_type: payload.organization_type,
  };
  Object.entries(display).forEach(([k, v]) => {
    const row = document.createElement("div");
    row.className = "feat-row";
    row.innerHTML = `<span class="feat-key">${k}</span><span class="feat-val">${v}</span>`;
    tbl.appendChild(row);
  });
}

// ── RENDER Benchmark results ──────────────────────────────────────────────
function renderBenchmark(groupA) {
  const targets = ORDER_A.filter(t => groupA[t]);

  // ── RADAR — normalized by SCALE_MAX ─────────────────────────────────
  const rawVals = targets.map(t => groupA[t].ensemble_mean ?? 0);
  const normVals = targets.map((t, i) => {
    const max = SCALE_MAX[t] || 100;
    return Math.max(0, Math.min(1, rawVals[i] / max));  // 0–1
  });
  const rColors = targets.map(t => COLORS_A[t]);

  if (radarChart) radarChart.destroy();
  radarChart = new Chart($("radarChart"), {
    type: "radar",
    data: {
      labels: targets.map(t => LABELS_SHORT[t]),
      datasets: [{
        label: "Score / Max",
        data: normVals,
        backgroundColor: "rgba(26,86,219,.1)",
        borderColor: "#1a56db",
        borderWidth: 2.5,
        pointBackgroundColor: rColors,
        pointBorderColor: "#fff",
        pointRadius: 5,
        pointHoverRadius: 7,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => {
              const t = targets[ctx.dataIndex];
              const raw = rawVals[ctx.dataIndex];
              const max = SCALE_MAX[t] || 100;
              return ` ${fmtNum(raw)} / ${max}  (${Math.round(ctx.raw * 100)}%)`;
            }
          }
        }
      },
      scales: {
        r: {
          min: 0, max: 1,
          ticks: {
            stepSize: 0.25,
            callback: v => `${Math.round(v * 100)}%`,
            font: { family: "'JetBrains Mono'", size: 9 },
            color: "#94a3b8",
            backdropColor: "transparent",
          },
          pointLabels: { font: { family: "'Inter'", size: 11, weight: "600" }, color: "#334155" },
          grid: { color: "#e2e8f0" },
          angleLines: { color: "#cbd5e1" },
        }
      }
    }
  });

  // ── ENSEMBLE TABLE — with Max and % columns ──────────────────────────
  const tbody = $("ensembleTable").querySelector("tbody");
  tbody.innerHTML = "";
  targets.forEach((t, i) => {
    const val = rawVals[i];
    const max = SCALE_MAX[t] || 100;
    const pct = Math.max(0, Math.min(100, (val / max) * 100)).toFixed(1);
    const dot = `<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${COLORS_A[t]};margin-right:.5rem;vertical-align:middle;flex-shrink:0"></span>`;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${dot}${LABELS_FULL[t]}</td>
      <td>${fmtNum(val)}</td>
      <td class="col-max">${max}</td>
      <td class="col-pct">${pct}%</td>`;
    tbody.appendChild(tr);
  });

  // ── PER-MODEL BARS ───────────────────────────────────────────────────
  const grid = $("modelBarsGrid");
  grid.innerHTML = "";

  targets.forEach((t, i) => {
    const d     = groupA[t];
    const preds = d.predictions || {};
    const ens   = rawVals[i];
    const color = COLORS_A[t];

    const allV = [...Object.values(preds), ens].filter(v => typeof v === "number");
    const maxV = Math.max(...allV, Math.abs(ens) * 1.1, 1e-9);

    const block = document.createElement("div");
    block.className = "target-block";

    const lbl = document.createElement("div");
    lbl.className = "target-block-label";
    lbl.style.color = color;
    lbl.textContent = LABELS_FULL[t];
    block.appendChild(lbl);

    Object.entries(preds).forEach(([mname, mval]) => {
      const w = Math.max(0, (mval / maxV) * 100).toFixed(1);
      const row = document.createElement("div");
      row.className = "bar-row";
      row.innerHTML = `
        <span class="bar-name">${mname}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${w}%;background:${hexAlpha(color,.48)}"></div></div>
        <span class="bar-val">${fmtNum(mval)}</span>`;
      block.appendChild(row);
    });

    const ensW = Math.max(0, (ens / maxV) * 100).toFixed(1);
    const ensRow = document.createElement("div");
    ensRow.className = "bar-row ens";
    ensRow.innerHTML = `
      <span class="bar-name">Ensemble μ</span>
      <div class="bar-track"><div class="bar-fill" style="width:${ensW}%;background:${color}"></div></div>
      <span class="bar-val">${fmtNum(ens)}</span>`;
    block.appendChild(ensRow);

    grid.appendChild(block);
  });
}

// ══════════════════════════════════════════════════════════
//  PAGE 2 — FEATURES INFERENCE
// ══════════════════════════════════════════════════════════
$("runFeaturesBtn").addEventListener("click", runFeatures);

async function runFeatures() {
  $("featuresError").classList.add("hidden");

  const manual = {
    ifeval:   parseNum($("b_ifeval").value),
    bbh:      parseNum($("b_bbh").value),
    math:     parseNum($("b_math").value),
    gpqa:     parseNum($("b_gpqa").value),
    musr:     parseNum($("b_musr").value),
    mmlu_pro: parseNum($("b_mmlu").value),
  };

  // Validate all 6 benchmarks
  const nullKey = Object.entries(manual).find(([,v]) => v === null);
  if (nullKey) {
    showErr("featuresError", `Invalid value for ${nullKey[0]} — enter a number between 0 and 100.`);
    return;
  }

  // Structural fields needed by the API — use sensible defaults
  // (Group B only uses benchmark scores; preprocessing still runs)
  const payload = {
    training_flops:      3e23,
    parameters:          7e9,
    dataset_size:        1e12,
    architecture:        "Qwen2ForCausalLM",
    model_type:          "chatmodels",
    organization_type:   "Industry",
    override_benchmarks: true,
    manual_benchmarks:   manual,
  };

  setLoading("runFeaturesBtn", "spinB", true);
  setState("features", "loader");

  try {
    const r = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({ detail: r.statusText }));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    renderFeatures(data.group_b || {}, manual);
    setState("features", "results");
  } catch (err) {
    showErr("featuresError", err.message || "Unexpected error.");
    setState("features", "placeholder");
  } finally {
    setLoading("runFeaturesBtn", "spinB", false);
  }
}

// ── RENDER Features results ───────────────────────────────────────────────
function renderFeatures(groupB, sentBenchmarks) {
  const reg   = groupB.regression     || {};
  const clf   = groupB.classification || {};
  const proba = groupB.clf_proba      || {};

  // ── 3 SEPARATE REGRESSION CHARTS ────────────────────────────────────
  const regGrid = $("regThreeGrid");
  regGrid.innerHTML = "";
  Object.values(regCharts).forEach(c => c.destroy());
  regCharts = {};

  Object.entries(reg).forEach(([label, val], i) => {
    const color = REG_COLORS[label] || "#4E79A7";
    const icon  = REG_ICONS[label]  || "◈";

    const wrap = document.createElement("div");
    wrap.className = "reg-single";
    wrap.innerHTML = `
      <div class="reg-title">${icon} ${label}</div>
      <div class="reg-value" style="color:${color}">${fmtLargeNum(val)}</div>
      <canvas id="rc_${i}" height="90"></canvas>`;
    regGrid.appendChild(wrap);

    regCharts[label] = new Chart(document.getElementById(`rc_${i}`), {
      type: "bar",
      data: {
        labels: [label],
        datasets: [{
          data: [val],
          backgroundColor: hexAlpha(color, .65),
          borderColor: color,
          borderWidth: 1.5,
          borderRadius: 6,
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: ctx => ` ${fmtLargeNum(ctx.raw)}` } }
        },
        scales: {
          x: {
            ticks: { font: { family: "'JetBrains Mono'", size: 8 }, callback: v => fmtLargeNum(v), maxTicksLimit: 4 },
            grid: { color: "#f1f5f9" },
          },
          y: { display: false, grid: { display: false } }
        }
      }
    });
  });

  // ── CLASSIFICATION GRID ──────────────────────────────────────────────
  const clfGrid = $("clfGrid");
  clfGrid.innerHTML = "";

  Object.entries(clf).forEach(([label, pred]) => {
    const p   = proba[label] ?? (pred ? 1 : 0);
    const yes = pred === 1 || pred === true;
    const pct = Math.round(p * 100);

    const item = document.createElement("div");
    item.className = `clf-item${yes ? " yes" : ""}`;
    item.innerHTML = `
      <span class="clf-badge-icon">${yes ? "✅" : "⬜"}</span>
      <div class="clf-item-name">${label}</div>
      <div class="clf-item-bar">
        <div class="clf-item-fill" style="width:${pct}%"></div>
      </div>
      <div class="clf-item-pct">${pct}%</div>`;
    clfGrid.appendChild(item);
  });

  // ── BENCHMARK ECHO — show exactly what was sent ──────────────────────
  const echo = $("benchEchoGrid");
  echo.innerHTML = "";
  const BENCH_LABELS = {
    ifeval: "IFEval", bbh: "BBH", math: "MATH",
    gpqa: "GPQA", musr: "MuSR", mmlu_pro: "MMLU-Pro"
  };
  Object.entries(sentBenchmarks).forEach(([k, v]) => {
    const tag = document.createElement("div");
    tag.className = "bench-tag";
    tag.innerHTML = `<span class="bench-key">${BENCH_LABELS[k] || k}</span><span class="bench-val">${v}</span>`;
    echo.appendChild(tag);
  });
}

// ── UI STATE ──────────────────────────────────────────────────────────────
function setState(section, state) {
  $(`${section}Placeholder`).classList.toggle("hidden", state !== "placeholder");
  $(`${section}Loader`).classList.toggle("hidden",      state !== "loader");
  $(`${section}Results`).classList.toggle("hidden",     state !== "results");
}

function setLoading(btnId, spinnerId, on) {
  const btn = $(btnId);
  btn.disabled = on;
  const txt = btn.querySelector(".run-text");
  if (txt) txt.classList.toggle("hidden", on);
  $(spinnerId).classList.toggle("hidden", !on);
}

function showErr(id, msg) {
  const el = $(id);
  el.textContent = `⚠ ${msg}`;
  el.classList.remove("hidden");
}

// ── NUMBER FORMATTERS ─────────────────────────────────────────────────────
function fmtNum(v) {
  if (v === null || v === undefined || (typeof v === "number" && isNaN(v))) return "—";
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
  if (Math.abs(v) >= 10000) return v.toFixed(1);
  return v.toFixed(3);
}

function fmtLargeNum(v) {
  if (v === null || v === undefined) return "—";
  const a = Math.abs(v);
  if (a >= 1e24) return (v/1e24).toFixed(2) + " Y";
  if (a >= 1e21) return (v/1e21).toFixed(2) + " Z";
  if (a >= 1e18) return (v/1e18).toFixed(2) + " E";
  if (a >= 1e15) return (v/1e15).toFixed(2) + " P";
  if (a >= 1e12) return (v/1e12).toFixed(2) + " T";
  if (a >= 1e9)  return (v/1e9).toFixed(2)  + " B";
  if (a >= 1e6)  return (v/1e6).toFixed(2)  + " M";
  if (a >= 1e3)  return (v/1e3).toFixed(2)  + " K";
  return v.toFixed(2);
}

function hexAlpha(hex, a) {
  hex = hex.replace("#", "");
  if (hex.length === 3) hex = hex.split("").map(c => c+c).join("");
  const r = parseInt(hex.slice(0,2), 16);
  const g = parseInt(hex.slice(2,4), 16);
  const b = parseInt(hex.slice(4,6), 16);
  return `rgba(${r},${g},${b},${a})`;
}
