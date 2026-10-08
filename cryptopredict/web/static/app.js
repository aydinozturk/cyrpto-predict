// CryptoPredict dashboard: vanilla JS, no build step, talks only to /api/*.
"use strict";

(() => {
  const TOKEN_KEY = "cryptopredict.dashboardToken";
  const POLL_MS = 2000;
  const POLL_RETRY_MS = 5000;
  const HEALTH_MS = 30000;
  const LOCALE = "tr-TR";

  const JOB_KINDS = { fetch: "Veri çekme", train: "Eğitim", backtest: "Backtest" };
  const JOB_STATUS = { queued: "Sırada", running: "Çalışıyor", done: "Bitti", failed: "Hata" };
  const DIRECTIONS = {
    up: { text: "Yükseliş ▲", cls: "up" },
    down: { text: "Düşüş ▼", cls: "down" },
    flat: { text: "Yatay ■", cls: "" },
  };
  // Backtest/metric keys: Turkish label and how to format the value.
  const METRICS = {
    mae: ["MAE", "num"],
    rmse: ["RMSE", "num"],
    smape: ["sMAPE (%)", "num"],
    directional_accuracy: ["Yön isabeti", "pct"],
    relative_mae: ["Göreli MAE", "num"],
    n: ["Örnek (n)", "int"],
    total_return: ["Toplam getiri", "pct"],
    bh_total_return: ["Al-tut getirisi", "pct"],
    excess_return: ["Fazla getiri", "pct"],
    sharpe: ["Sharpe", "num"],
    bh_sharpe: ["Al-tut Sharpe", "num"],
    max_drawdown: ["Maks. düşüş", "pct"],
    bh_max_drawdown: ["Al-tut maks. düşüş", "pct"],
    n_trades: ["İşlem sayısı", "int"],
    exposure: ["Piyasada kalma", "pct"],
    total_cost: ["Toplam maliyet", "pct"],
    n_bars: ["Bar sayısı", "int"],
  };

  const state = {
    config: null,
    jobs: [],
    knownFinished: null, // Set of job ids already finished at first load (no toasts for them)
    pollTimer: null,
    watchBacktest: null,
    selectedJob: null,
    bars: [],
    chartKey: null,
    hoverIndex: null,
  };

  // ---------- DOM helpers ----------

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  function el(tag, attrs = {}, ...children) {
    const node = document.createElement(tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value === undefined || value === null || value === false) continue;
      if (key === "class") node.className = value;
      else if (key === "dataset") Object.assign(node.dataset, value);
      else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
      else node.setAttribute(key, value === true ? "" : value);
    }
    for (const child of children.flat()) {
      if (child === undefined || child === null || child === false) continue;
      node.append(child instanceof Node ? child : String(child));
    }
    return node;
  }

  function fillBody(table, rows, emptyText, colspan) {
    const body = $("tbody", table);
    body.replaceChildren(...(rows.length ? rows : [el("tr", {}, el("td", { class: "empty", colspan }, emptyText))]));
  }

  let toastTimer = null;
  function toast(message, isError = false) {
    const box = $("#toast");
    box.textContent = message;
    box.classList.toggle("error", isError);
    box.hidden = false;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { box.hidden = true; }, isError ? 8000 : 4000);
  }

  // ---------- formatting ----------

  const isNum = (v) => typeof v === "number" && Number.isFinite(v);

  function fmtNum(v, maxDigits = 4) {
    if (!isNum(v)) return "—";
    return v.toLocaleString(LOCALE, { maximumFractionDigits: maxDigits });
  }

  function fmtPrice(v) {
    if (!isNum(v)) return "—";
    const abs = Math.abs(v);
    const digits = abs >= 1000 ? 2 : abs >= 1 ? 4 : 8;
    return v.toLocaleString(LOCALE, { minimumFractionDigits: Math.min(2, digits), maximumFractionDigits: digits });
  }

  function fmtPct(v, digits = 2) {
    if (!isNum(v)) return "—";
    return `${v.toLocaleString(LOCALE, { minimumFractionDigits: digits, maximumFractionDigits: digits })}%`;
  }

  function fmtDate(iso, withTime = true) {
    if (!iso) return "—";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return String(iso);
    return d.toLocaleString(LOCALE, withTime ? { dateStyle: "short", timeStyle: "short" } : { dateStyle: "short" });
  }

  function fmtDuration(ms) {
    if (!isNum(ms) || ms < 0) return "—";
    const s = Math.round(ms / 1000);
    return s < 60 ? `${s} sn` : `${Math.floor(s / 60)} dk ${s % 60} sn`;
  }

  function fmtMetric(key, v) {
    const kind = METRICS[key]?.[1];
    if (kind === "pct") return isNum(v) ? fmtPct(v * 100) : "—";
    if (kind === "int") return fmtNum(v, 0);
    if (typeof v === "boolean") return v ? "evet" : "hayır";
    if (typeof v === "string") return v;
    return fmtNum(v, 6);
  }

  const metricLabel = (key) => METRICS[key]?.[0] ?? key;

  function fmtParams(params) {
    if (!params || typeof params !== "object") return "";
    return Object.entries(params)
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => `${k}=${v}`)
      .join(" · ");
  }

  // ---------- API ----------

  class ApiError extends Error {
    constructor(status, message) {
      super(message);
      this.status = status;
    }
  }

  function detailText(data) {
    const detail = data && typeof data === "object" ? data.detail : data;
    if (Array.isArray(detail)) {
      // FastAPI validation errors: [{loc: [...], msg: "..."}]
      return detail.map((d) => `${(d.loc || []).filter((p) => p !== "body").join(".")}: ${d.msg}`).join("; ");
    }
    return detail ? String(detail) : "";
  }

  async function api(path, { method = "GET", body } = {}) {
    const headers = { Accept: "application/json" };
    const token = localStorage.getItem(TOKEN_KEY);
    if (token) headers.Authorization = `Bearer ${token}`;
    if (body !== undefined) headers["Content-Type"] = "application/json";
    let res;
    try {
      res = await fetch(path, { method, headers, body: body === undefined ? undefined : JSON.stringify(body) });
    } catch {
      throw new ApiError(0, "Sunucuya ulaşılamadı.");
    }
    const text = res.status === 204 ? "" : await res.text();
    let data = null;
    if (text) {
      try { data = JSON.parse(text); } catch { data = text; }
    }
    if (!res.ok) {
      if (res.status === 401) showAuth(token ? "Erişim anahtarı geçersiz." : "Bu panel bir erişim anahtarı istiyor.");
      throw new ApiError(res.status, detailText(data) || `${res.status} ${res.statusText}`);
    }
    return data;
  }

  // Runs an async UI action; reports errors as a toast (401 is shown by the auth card instead).
  async function guard(fn, button) {
    if (button) button.disabled = true;
    try {
      return await fn();
    } catch (err) {
      if (!(err instanceof ApiError && err.status === 401)) toast(err.message || String(err), true);
      return undefined;
    } finally {
      if (button) button.disabled = false;
    }
  }

  // ---------- auth ----------

  function showAuth(message) {
    $("#auth").hidden = false;
    $("#auth-msg").textContent = message || "";
  }

  function bindAuth() {
    $("#token-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const input = ev.target.elements.token;
      localStorage.setItem(TOKEN_KEY, input.value.trim());
      input.value = "";
      $("#auth-msg").textContent = "Anahtar kaydedildi, yeniden yükleniyor…";
      boot();
    });
    $("#token-clear").addEventListener("click", () => {
      localStorage.removeItem(TOKEN_KEY);
      showAuth("Anahtar bu tarayıcıdan silindi.");
    });
  }

  // ---------- health + config ----------

  async function checkHealth() {
    const box = $("#health");
    try {
      const h = await api("/api/health");
      box.className = "health ok";
      $(".health-text", box).textContent = `API çalışıyor (${h.status})`;
      $("#version").textContent = `sürüm ${h.version}`;
    } catch (err) {
      box.className = "health down";
      $(".health-text", box).textContent = `API erişilemiyor: ${err.message}`;
    }
  }

  function setOptions(select, values, selected, labels = {}) {
    const current = select.value;
    select.replaceChildren(...values.map((v) => el("option", { value: v }, labels[v] ?? v)));
    const pick = values.includes(current) ? current : selected;
    if (values.includes(pick)) select.value = pick;
  }

  async function loadConfig() {
    const cfg = await api("/api/config");
    state.config = cfg;
    $("#version").textContent = `sürüm ${cfg.version}`;
    for (const sel of $$("select[name=interval]")) setOptions(sel, cfg.intervals, cfg.default_interval);
    for (const sel of $$("select[data-models]")) {
      const values = "all" in sel.dataset ? [...cfg.models, "all"] : cfg.models;
      setOptions(sel, values, "all" in sel.dataset ? "all" : cfg.models.includes("ridge") ? "ridge" : cfg.models[0],
        { all: "all — tüm modelleri karşılaştır" });
    }
    for (const input of $$("input[name=symbol]")) if (!input.value) input.value = cfg.default_symbol;
    $("#dirs").textContent = `Veri: ${cfg.data_dir} · Modeller: ${cfg.model_dir}`;
    if (cfg.auth_required) showAuth("Erişim anahtarı bu tarayıcıda kayıtlı.");
    else $("#auth").hidden = true;
  }

  // ---------- tabs ----------

  const TABS = ["data", "train", "models", "backtest", "jobs"];

  function selectTab(name) {
    if (!TABS.includes(name)) name = "data";
    for (const btn of $$(".tabs button")) btn.setAttribute("aria-selected", String(btn.dataset.tab === name));
    for (const tab of TABS) $(`#tab-${tab}`).hidden = tab !== name;
    if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
    if (name === "data") requestAnimationFrame(drawChart);
    if (!state.config) return; // boot() loads everything once the config is in
    if (name === "data") guard(loadDatasets);
    if (name === "models") guard(loadModels);
    if (name === "jobs") guard(loadJobs);
  }

  // ---------- forms ----------

  // Collects non-empty form values; `numbers` keys become numbers, checkboxes booleans.
  function formData(form, numbers = []) {
    const out = {};
    for (const field of form.elements) {
      if (!field.name) continue;
      if (field.type === "checkbox") { out[field.name] = field.checked; continue; }
      const value = field.value.trim();
      if (value === "") continue;
      out[field.name] = numbers.includes(field.name) ? Number(value) : value;
    }
    if (out.symbol) out.symbol = out.symbol.toUpperCase();
    return out;
  }

  async function submitJob(path, body, label) {
    const res = await api(path, { method: "POST", body });
    toast(`${label} işi #${res.job_id} kuyruğa alındı.`);
    await loadJobs();
    return res.job_id;
  }

  function bindForms() {
    $("#fetch-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      guard(() => submitJob("/api/fetch", formData(ev.target), "Veri çekme"), ev.submitter);
    });

    $("#train-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      guard(() => submitJob("/api/train", formData(ev.target, ["horizon"]), "Eğitim"), ev.submitter);
    });

    $("#backtest-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const body = formData(ev.target, ["horizon", "splits", "fee_bps", "slippage_bps", "threshold", "da_threshold"]);
      guard(async () => {
        const id = await submitJob("/api/backtest", body, "Backtest");
        state.watchBacktest = id;
        const card = $("#backtest-result");
        card.hidden = false;
        card.replaceChildren(el("h2", {}, `Backtest işi #${id}`), el("p", { class: "hint" }, "İş çalışıyor; bitince sonuç burada görünecek."));
      }, ev.submitter);
    });

    $("#chart-form").addEventListener("submit", (ev) => {
      ev.preventDefault();
      const f = formData(ev.target, ["limit"]);
      guard(() => loadChart(f.symbol, f.interval, f.limit), ev.submitter);
    });

    for (const btn of $$("[data-action]")) {
      const action = { "reload-datasets": loadDatasets, "reload-models": loadModels, "reload-jobs": loadJobs }[btn.dataset.action];
      btn.addEventListener("click", () => guard(action, btn));
    }
  }

  // Copies symbol/interval into every form so the next action targets the same dataset.
  function prefill(symbol, interval) {
    for (const form of $$("form")) {
      if (form.elements.symbol) form.elements.symbol.value = symbol;
      const sel = form.elements.interval;
      if (sel && Array.from(sel.options).some((o) => o.value === interval)) sel.value = interval;
    }
  }

  // ---------- datasets + chart ----------

  async function loadDatasets() {
    const rows = await api("/api/datasets");
    const trs = rows.map((d) => {
      const key = `${d.symbol}_${d.interval}`;
      const open = () => { prefill(d.symbol, d.interval); guard(() => loadChart(d.symbol, d.interval)); };
      return el("tr", { class: `clickable${state.chartKey === key ? " selected" : ""}`, dataset: { key }, onclick: open },
        el("td", {}, d.symbol), el("td", {}, d.interval), el("td", { class: "num" }, fmtNum(d.rows, 0)),
        el("td", {}, fmtDate(d.start)), el("td", {}, fmtDate(d.end)),
        el("td", { class: "actions-cell" }, el("button", { type: "button", class: "secondary small" }, "Grafik")));
    });
    fillBody($("#datasets-table"), trs, "Henüz önbellekte veri yok; aşağıdan Binance'ten veri çekin.", 6);
    if (!state.chartKey && rows.length) {
      const first = rows.find((d) => d.symbol === state.config?.default_symbol) || rows[0];
      prefill(first.symbol, first.interval);
      await loadChart(first.symbol, first.interval);
    }
  }

  async function loadChart(symbol, interval, limit) {
    const form = $("#chart-form");
    limit = limit || Number(form.elements.limit.value) || 500;
    const params = new URLSearchParams({ symbol, interval, limit: String(limit) });
    const data = await api(`/api/ohlcv?${params}`);
    state.bars = data.bars.map((b) => ({ ...b, time: new Date(b.t).getTime() }));
    state.chartKey = `${data.symbol}_${data.interval}`;
    state.hoverIndex = null;
    form.elements.symbol.value = data.symbol;
    form.elements.interval.value = data.interval;
    for (const tr of $$("#datasets-table tbody tr")) tr.classList.toggle("selected", tr.dataset.key === state.chartKey);
    drawChart();
  }

  const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  function chartGeometry(canvas) {
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    const pad = { left: 12, right: 78, top: 28, bottom: 28 };
    const closes = state.bars.map((b) => b.c);
    let min = Math.min(...closes);
    let max = Math.max(...closes);
    if (min === max) { min -= 1; max += 1; }
    const span = max - min;
    min -= span * 0.05;
    max += span * 0.05;
    const n = state.bars.length;
    const x = (i) => pad.left + (n === 1 ? 0 : (i / (n - 1)) * (w - pad.left - pad.right));
    const y = (v) => pad.top + (1 - (v - min) / (max - min)) * (h - pad.top - pad.bottom);
    return { w, h, pad, min, max, x, y };
  }

  function drawChart() {
    const canvas = $("#chart");
    const empty = $("#chart-empty");
    const ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (!w || !h) return; // tab hidden
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    empty.hidden = state.bars.length > 0;
    if (!state.bars.length) return;

    const g = chartGeometry(canvas);
    const colors = { grid: cssVar("--grid"), muted: cssVar("--muted"), text: cssVar("--text"), line: cssVar("--accent") };
    ctx.font = "12px system-ui, sans-serif";
    ctx.lineWidth = 1;

    // horizontal grid + price labels
    ctx.textBaseline = "middle";
    ctx.textAlign = "left";
    for (let k = 0; k <= 4; k++) {
      const v = g.min + ((g.max - g.min) * k) / 4;
      const yy = Math.round(g.y(v)) + 0.5;
      ctx.strokeStyle = colors.grid;
      ctx.beginPath(); ctx.moveTo(g.pad.left, yy); ctx.lineTo(g.w - g.pad.right, yy); ctx.stroke();
      ctx.fillStyle = colors.muted;
      ctx.fillText(fmtPrice(v), g.w - g.pad.right + 6, yy);
    }

    // time labels
    const n = state.bars.length;
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    const ticks = Math.min(5, n);
    for (let k = 0; k < ticks; k++) {
      const i = ticks === 1 ? 0 : Math.round((k * (n - 1)) / (ticks - 1));
      const xx = g.x(i);
      ctx.textAlign = k === 0 ? "left" : k === ticks - 1 ? "right" : "center";
      ctx.fillText(fmtDate(state.bars[i].t, (state.bars[n - 1].time - state.bars[0].time) < 3 * 86400e3), xx, g.h - g.pad.bottom + 8);
    }

    // close line
    ctx.strokeStyle = colors.line;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = "round";
    ctx.beginPath();
    state.bars.forEach((b, i) => (i ? ctx.lineTo(g.x(i), g.y(b.c)) : ctx.moveTo(g.x(i), g.y(b.c))));
    ctx.stroke();

    // title: symbol + last close
    const last = state.bars[n - 1];
    ctx.fillStyle = colors.text;
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(`${state.chartKey.replace("_", " · ")}  son kapanış ${fmtPrice(last.c)}  (${n} bar)`, g.pad.left, 6);

    // hover crosshair
    const hi = state.hoverIndex;
    if (hi !== null && state.bars[hi]) {
      const xx = g.x(hi);
      const yy = g.y(state.bars[hi].c);
      ctx.strokeStyle = colors.muted;
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(xx, g.pad.top); ctx.lineTo(xx, g.h - g.pad.bottom); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = colors.line;
      ctx.beginPath(); ctx.arc(xx, yy, 3.5, 0, Math.PI * 2); ctx.fill();
    }
  }

  function bindChart() {
    const canvas = $("#chart");
    const tip = $("#chart-tip");
    canvas.addEventListener("mousemove", (ev) => {
      if (!state.bars.length) return;
      const g = chartGeometry(canvas);
      const rect = canvas.getBoundingClientRect();
      const px = ev.clientX - rect.left;
      const n = state.bars.length;
      const ratio = (px - g.pad.left) / (g.w - g.pad.left - g.pad.right);
      const i = Math.max(0, Math.min(n - 1, Math.round(ratio * (n - 1))));
      if (i !== state.hoverIndex) { state.hoverIndex = i; drawChart(); }
      const b = state.bars[i];
      tip.replaceChildren(
        el("div", {}, el("strong", {}, fmtDate(b.t))),
        el("div", {}, `A ${fmtPrice(b.o)} · Y ${fmtPrice(b.h)} · D ${fmtPrice(b.l)} · K ${fmtPrice(b.c)}`),
        el("div", {}, `Hacim ${fmtNum(b.v, 2)}`));
      tip.hidden = false;
      const left = g.x(i) + 12;
      tip.style.left = `${Math.min(left, g.w - tip.offsetWidth - 4)}px`;
      tip.style.top = `${g.pad.top}px`;
    });
    canvas.addEventListener("mouseleave", () => {
      tip.hidden = true;
      state.hoverIndex = null;
      drawChart();
    });
    window.addEventListener("resize", drawChart);
    matchMedia("(prefers-color-scheme: dark)").addEventListener("change", drawChart);
  }

  // ---------- models + prediction ----------

  async function loadModels() {
    const models = await api("/api/models");
    const trs = models.map((m) => el("tr", {},
      el("td", {}, el("strong", {}, m.name)), el("td", {}, m.model), el("td", {}, m.symbol ?? "—"), el("td", {}, m.interval ?? "—"),
      el("td", { class: "num" }, fmtNum(m.horizon, 0)),
      el("td", {}, m.train_period ? `${fmtDate(m.train_period.start, false)} – ${fmtDate(m.train_period.end, false)}` : "—"),
      el("td", { class: "num" }, fmtNum(m.n_samples, 0)), el("td", {}, fmtDate(m.created_at)),
      el("td", { class: "actions-cell" },
        el("button", { type: "button", class: "small", onclick: (ev) => guard(() => predict(m.name), ev.currentTarget) }, "Tahmin et"),
        el("button", { type: "button", class: "danger small", onclick: (ev) => guard(() => deleteModel(m.name), ev.currentTarget) }, "Sil"))));
    fillBody($("#models-table"), trs, "Kayıtlı model yok; Eğit sekmesinden bir model eğitin.", 9);
  }

  async function predict(name) {
    const refresh = $("#predict-refresh").checked;
    const p = await api("/api/predict", { method: "POST", body: { name, refresh } });
    const dir = DIRECTIONS[p.direction] || { text: p.direction, cls: "" };
    const stat = (label, value, cls) => el("div", { class: "stat" }, el("div", { class: "label" }, label), el("div", { class: `value ${cls || ""}` }, value));
    const card = $("#prediction");
    card.hidden = false;
    card.dataset.model = name;
    card.replaceChildren(
      el("div", { class: "card-head" }, el("h2", {}, `Tahmin: ${name}`), el("span", { class: "hint" }, `${p.symbol} · ${p.interval} · ${p.horizon} bar ileri`)),
      el("div", { class: "prediction-grid" },
        stat("Yön", dir.text, dir.cls),
        stat("Beklenen fiyat", fmtPrice(p.expected_price)),
        stat("Değişim", fmtPct(p.predicted_pct_change, 3), dir.cls),
        stat("Son kapanış", fmtPrice(p.last_close)),
        stat("Hedef zaman", fmtDate(p.target_time)),
        stat("Veri tarihi (as of)", fmtDate(p.as_of))),
      el("p", { class: "hint" },
        `Tahmini log-getiri: ${fmtNum(p.predicted_log_return, 6)}.`,
        p.dropped_open_bars ? ` Henüz kapanmamış ${p.dropped_open_bars} mum hesaba katılmadı.` : "",
        " Yatırım tavsiyesi değildir."));
  }

  async function deleteModel(name) {
    if (!confirm(`"${name}" modeli silinsin mi?`)) return;
    await api(`/api/models/${encodeURIComponent(name)}`, { method: "DELETE" });
    toast(`"${name}" silindi.`);
    if ($("#prediction").dataset.model === name) $("#prediction").hidden = true;
    await loadModels();
  }

  // ---------- backtest results ----------

  function backtestView(result) {
    if (result?.table) {
      const rows = result.table;
      const keys = [...new Set(rows.flatMap((r) => Object.keys(r)))].filter((k) => k !== "model");
      return el("div", { class: "table-wrap" }, el("table", {},
        el("thead", {}, el("tr", {}, el("th", {}, "Model"), keys.map((k) => el("th", { class: "num" }, metricLabel(k))))),
        el("tbody", {}, rows.map((r) => el("tr", {}, el("td", {}, el("strong", {}, r.model)),
          keys.map((k) => el("td", { class: "num" }, fmtMetric(k, r[k]))))))));
    }
    if (result?.summary) {
      return el("div", { class: "table-wrap" }, el("table", {},
        el("tbody", {}, Object.entries(result.summary).map(([k, v]) => el("tr", {},
          el("th", {}, metricLabel(k)), el("td", { class: "num" }, fmtMetric(k, v)))))));
    }
    return el("pre", {}, JSON.stringify(result, null, 2));
  }

  function renderBacktest(job) {
    const card = $("#backtest-result");
    card.hidden = false;
    const title = el("div", { class: "card-head" }, el("h2", {}, `Backtest işi #${job.id}`), el("span", { class: "hint" }, fmtParams(job.params)));
    if (job.status === "failed") card.replaceChildren(title, el("p", { class: "error-text" }, job.error || "İş başarısız oldu."));
    else card.replaceChildren(title, backtestView(job.result),
      el("p", { class: "hint" }, "Getiri, düşüş, maliyet ve yön isabeti yüzde olarak gösterilir; al-tut = buy & hold."));
  }

  // ---------- jobs ----------

  const isActive = (job) => job.status === "queued" || job.status === "running";

  function jobSummary(job) {
    if (job.status === "failed") return el("span", { class: "error-text" }, job.error || "hata");
    if (job.status !== "done" || !job.result) return "";
    const r = job.result;
    if (job.kind === "fetch") return `${fmtNum(r.rows, 0)} satır · ${fmtDate(r.start, false)} – ${fmtDate(r.end, false)}`;
    if (job.kind === "train") return `Model: ${r.name}`;
    if (job.kind === "backtest") return r.table ? `${r.table.length} model karşılaştırıldı — ayrıntı için tıklayın` : "Özet için tıklayın";
    return "";
  }

  function jobDuration(job) {
    if (!job.started_at) return "—";
    const end = job.finished_at ? new Date(job.finished_at) : new Date();
    return fmtDuration(end - new Date(job.started_at));
  }

  function renderJobs() {
    const trs = state.jobs.map((job) => el("tr", {
      class: `clickable${state.selectedJob === job.id ? " selected" : ""}`,
      onclick: () => { state.selectedJob = job.id; renderJobs(); },
    },
      el("td", {}, job.id), el("td", {}, JOB_KINDS[job.kind] || job.kind),
      el("td", {}, el("span", { class: `status ${job.status}` }, JOB_STATUS[job.status] || job.status)),
      el("td", { class: "params" }, fmtParams(job.params)), el("td", {}, fmtDate(job.created_at)),
      el("td", { class: "num" }, jobDuration(job)), el("td", {}, jobSummary(job))));
    fillBody($("#jobs-table"), trs, "Henüz iş yok.", 7);

    const active = state.jobs.filter(isActive).length;
    const badge = $("#jobs-active");
    badge.hidden = active === 0;
    badge.textContent = String(active);

    const detail = $("#job-detail");
    const job = state.jobs.find((j) => j.id === state.selectedJob);
    detail.hidden = !job;
    if (!job) return;
    const head = el("div", { class: "card-head" }, el("h2", {}, `İş #${job.id} — ${JOB_KINDS[job.kind] || job.kind}`),
      el("span", { class: `status ${job.status}` }, JOB_STATUS[job.status] || job.status));
    let body;
    if (job.status === "failed") body = el("p", { class: "error-text" }, job.error || "İş başarısız oldu.");
    else if (job.status !== "done") body = el("p", { class: "hint" }, "İş henüz bitmedi.");
    else if (job.kind === "backtest") body = backtestView(job.result);
    else body = el("pre", {}, JSON.stringify(job.result, null, 2));
    detail.replaceChildren(head, el("p", { class: "params" }, fmtParams(job.params)), body);
  }

  // Side effects when a job finishes during this session.
  function onJobFinished(job) {
    const kind = JOB_KINDS[job.kind] || job.kind;
    if (job.status === "failed") {
      toast(`${kind} işi #${job.id} başarısız: ${job.error || "bilinmeyen hata"}`, true);
    } else {
      toast(`${kind} işi #${job.id} tamamlandı.`);
      if (job.kind === "fetch") {
        guard(loadDatasets);
        const key = job.result ? `${job.result.symbol}_${job.result.interval}` : null;
        if (key && key === state.chartKey) guard(() => loadChart(job.result.symbol, job.result.interval));
      }
      if (job.kind === "train") guard(loadModels);
    }
    if (job.kind === "backtest" && job.id === state.watchBacktest) renderBacktest(job);
  }

  async function loadJobs() {
    clearTimeout(state.pollTimer);
    try {
      state.jobs = await api("/api/jobs");
    } catch (err) {
      if (!(err instanceof ApiError && err.status === 401)) state.pollTimer = setTimeout(loadJobs, POLL_RETRY_MS);
      throw err;
    }
    const finished = state.jobs.filter((j) => !isActive(j));
    if (state.knownFinished === null) {
      state.knownFinished = new Set(finished.map((j) => j.id));
    } else {
      for (const job of finished) {
        if (state.knownFinished.has(job.id)) continue;
        state.knownFinished.add(job.id);
        onJobFinished(job);
      }
    }
    renderJobs();
    if (state.jobs.some(isActive)) state.pollTimer = setTimeout(() => guard(loadJobs), POLL_MS);
  }

  // ---------- boot ----------

  async function boot() {
    await checkHealth();
    const ok = await guard(async () => { await loadConfig(); return true; });
    if (!ok) return;
    await Promise.all([guard(loadDatasets), guard(loadModels), guard(loadJobs)]);
  }

  function init() {
    for (const btn of $$(".tabs button")) btn.addEventListener("click", () => selectTab(btn.dataset.tab));
    window.addEventListener("hashchange", () => selectTab(location.hash.slice(1)));
    bindAuth();
    bindForms();
    bindChart();
    selectTab(location.hash.slice(1) || "data");
    setInterval(checkHealth, HEALTH_MS);
    boot();
  }

  init();
})();
