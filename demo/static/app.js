const appState = {
  scenarios: [],
  currentScenarioId: null,
  currentState: null,
  modelInfo: null,
  trainingStatus: null,
  compareRecommendations: {
    model: [],
    heuristic: [],
  },
  selectedRecommendation: {
    source: "model",
    index: 0,
  },
  busy: {
    action: false,
    training: false,
  },
  theme: "light",
  pollCounter: 0,
};

const ICONS = {
  refresh: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M20 11a8 8 0 1 0 2 5.3"/><path d="M20 4v7h-7"/></svg>`,
  play: `<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5.5v13l10-6.5z"/></svg>`,
  spark: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m13 2 1.8 5.2L20 9l-5.2 1.8L13 16l-1.8-5.2L6 9l5.2-1.8z"/><path d="M5 17l.9 2.1L8 20l-2.1.9L5 23l-.9-2.1L2 20l2.1-.9z"/></svg>`,
  rocket: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4.5 19.5c2.3-.1 4.1-.8 5.6-2.3l5.7-5.7c1.2-1.2 2-2.7 2.3-4.4l.5-3.1-3.1.5c-1.7.3-3.2 1.1-4.4 2.3l-5.7 5.7c-1.5 1.5-2.2 3.3-2.3 5.6Z"/><path d="M13 11l-2-2"/><path d="M4 20l-1 3 3-1"/></svg>`,
  flask: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M10 2v6.2L4.8 17a3 3 0 0 0 2.6 4.5h9.2A3 3 0 0 0 19.2 17L14 8.2V2"/><path d="M8 2h8"/><path d="M7 16h10"/></svg>`,
  sun: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M4.9 4.9l1.8 1.8M17.3 17.3l1.8 1.8M2 12h2.5M19.5 12H22M4.9 19.1l1.8-1.8M17.3 6.7l1.8-1.8"/></svg>`,
  moon: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M21 12.8A9 9 0 1 1 11.2 3 7 7 0 0 0 21 12.8Z"/></svg>`,
  trend: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 17 9 11l4 4 8-8"/><path d="M14 7h7v7"/></svg>`,
};

function icon(name) {
  return ICONS[name] || "";
}

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `请求失败：${path}`);
  }
  return payload;
}

function formatNumber(value, digits = 1) {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric)) return "-";
  return Number.isInteger(numeric) ? `${numeric}` : numeric.toFixed(digits);
}

function formatPct(value) {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric)) return "-";
  return `${(numeric * 100).toFixed(1)}%`;
}

function formatDuration(seconds) {
  const total = Number(seconds || 0);
  if (!Number.isFinite(total) || total <= 0) return "-";
  const hour = Math.floor(total / 3600);
  const minute = Math.floor((total % 3600) / 60);
  const second = Math.floor(total % 60);
  if (hour > 0) return `${hour}h ${minute}m`;
  if (minute > 0) return `${minute}m ${second}s`;
  return `${second}s`;
}

function metricCard(label, value, subvalue = "") {
  const template = document.querySelector("#metric-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".metric-label").textContent = label;
  node.querySelector(".metric-value").textContent = value;
  node.querySelector(".metric-subvalue").textContent = subvalue;
  return node;
}

function emptyState(title, description) {
  return `
    <div class="empty-state">
      <strong>${title}</strong>
      <p>${description}</p>
    </div>
  `;
}

function showToast(message, type = "info") {
  const root = document.querySelector("#toast-root");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  root.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

function stageText(stage) {
  const mapping = {
    idle: "空闲",
    queued: "已排队",
    building_dataset: "构建数据集",
    sweep: "超参筛选",
    training: "正式训练",
    evaluating: "评估新模型",
    finished: "已完成",
    failed: "失败",
  };
  return mapping[stage] || stage || "-";
}

function statusText(status) {
  const mapping = {
    idle: "空闲",
    queued: "排队中",
    building_dataset: "构建中",
    training: "训练中",
    evaluating: "评估中",
    running: "进行中",
    finished: "完成",
    failed: "失败",
  };
  return mapping[status] || status || "-";
}

function readableStatusLabel(rawLabel, status) {
  const fallback = statusText(status || "idle");
  const label = typeof rawLabel === "string" ? rawLabel.trim() : "";
  if (!label) return fallback;
  if (/^\?+$/.test(label)) return fallback;
  if (label.includes("锟") || label.includes("�")) return fallback;
  return label;
}

function trainingStage(status = appState.trainingStatus?.status) {
  return status || "idle";
}

function trainingIsActive(status = appState.trainingStatus?.status) {
  return ["queued", "building_dataset", "training", "evaluating"].includes(status || "");
}

function currentInsights() {
  const current = appState.currentState || {};
  if (current.insights) return current.insights;
  const meta = current.meta || {};
  return {
    risk_level: meta.risk_label || "-",
    risk_code: meta.risk_level || "low",
    decision_type: meta.decision_label || "-",
    decision_hint: meta.starter_hint || "",
    risk_reason: meta.risk_hint || "",
    score_gap: 0,
    tags: [],
  };
}

function trainingHistoryRows() {
  const liveProgress = Array.isArray(appState.trainingStatus?.progress) ? appState.trainingStatus.progress : [];
  const liveHistory = liveProgress
    .map((entry) => entry.metrics || entry)
    .filter((entry) => Number.isFinite(Number(entry.train_loss)) || Number.isFinite(Number(entry.val_loss)));
  if (liveHistory.length) return liveHistory;
  return Array.isArray(appState.modelInfo?.history) ? appState.modelInfo.history : [];
}

function trainingProgressValue() {
  const training = appState.trainingStatus || {};
  const value = Number(
    training.training?.progress ??
      training.dataset?.progress ??
      (Array.isArray(training.progress) && training.progress.length
        ? training.progress[training.progress.length - 1]?.progress
        : 0)
  );
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
}

function renderIconSlots(root = document) {
  root.querySelectorAll("[data-icon]").forEach((node) => {
    node.innerHTML = icon(node.dataset.icon);
  });
}

function applyTheme(theme) {
  appState.theme = theme;
  document.body.dataset.theme = theme;
  localStorage.setItem("balatro_mvp_theme", theme);
  const toggle = document.querySelector("#theme-toggle");
  toggle.innerHTML = `
    <button class="secondary subtle" id="theme-button" title="切换浅色 / 深色模式">
      <span class="button-icon">${theme === "dark" ? icon("sun") : icon("moon")}</span>
      ${theme === "dark" ? "切到浅色" : "切到深色"}
    </button>
  `;
  toggle.querySelector("button").addEventListener("click", () => {
    applyTheme(appState.theme === "dark" ? "light" : "dark");
  });
}

function initTheme() {
  const saved = localStorage.getItem("balatro_mvp_theme");
  applyTheme(saved || "light");
}

function setStatusMessage(text) {
  document.querySelector("#status-message").textContent = text || "";
}

function currentRunId() {
  const runDir = appState.modelInfo?.run_dir || "";
  return runDir ? runDir.split("\\").pop() : "-";
}

function getSelectedRecommendation() {
  const { source, index } = appState.selectedRecommendation;
  return (appState.compareRecommendations[source] || [])[index] || null;
}

function ensureSelectedRecommendation() {
  const current = getSelectedRecommendation();
  if (current) return;
  if ((appState.compareRecommendations.model || []).length) {
    appState.selectedRecommendation = { source: "model", index: 0 };
    return;
  }
  if ((appState.compareRecommendations.heuristic || []).length) {
    appState.selectedRecommendation = { source: "heuristic", index: 0 };
  }
}

function renderTopSummary() {
  const current = appState.currentState;
  if (!current) return;
  document.querySelector("#status-scenario").textContent = current.scenario.name;
  document.querySelector("#status-model").textContent = appState.modelInfo?.model_name || current.model_name || "-";
  document.querySelector("#status-run-id").textContent = `运行 ID：${currentRunId()}`;
  document.querySelector("#status-mode").textContent = current.mode === "autoplay" ? "自动演示" : "手动";
  document.querySelector("#status-phase").textContent = current.phase_label || current.phase_text || current.phase;
  document.querySelector("#scenario-focus-inline").textContent = current.scenario.focus || "-";
  document.querySelector("#hero-description").textContent =
    current.scenario.talk_track || "先看推荐，再执行动作或自动演示。";
}

function renderStatusStrip() {
  const current = appState.currentState;
  if (!current) return;
  const insights = currentInsights();
  document.querySelector("#status-risk").textContent = insights.risk_level;
  document.querySelector("#status-decision-type").textContent = insights.decision_type;
  document.querySelector("#status-gap").textContent = `${formatNumber(insights.score_gap, 0)} 筹码`;
  document.querySelector("#status-training").textContent =
    readableStatusLabel(appState.trainingStatus?.status_label, appState.trainingStatus?.status);
}

function renderScenarios() {
  const container = document.querySelector("#scenario-list");
  container.innerHTML = "";
  appState.scenarios.forEach((scenario) => {
    const active = scenario.id === appState.currentScenarioId ? " active" : "";
    const tags = (scenario.tags || []).map((tag) => `<span class="chip">${tag}</span>`).join("");
    const node = document.createElement("div");
    node.className = `scenario-card${active}`;
    node.innerHTML = `
      <h4>${scenario.name}</h4>
      <p>${scenario.summary}</p>
      <div class="scenario-meta">
        <span>${scenario.focus}</span>
        <span>手数 ${scenario.hands_left} / 弃牌 ${scenario.discards_left}</span>
      </div>
      <div class="tag-row">${tags}</div>
      <p class="muted">${scenario.talk_track || ""}</p>
      <div class="hero-actions" style="margin-top:12px;">
        <button class="secondary subtle">载入场景</button>
      </div>
    `;
    node.addEventListener("click", async () => {
      await loadScenario(scenario.id);
    });
    container.appendChild(node);
  });
}

function renderInsightTags() {
  const insights = currentInsights();
  const root = document.querySelector("#insight-tags");
  root.innerHTML = (insights.tags || []).map((tag) => `<span class="chip">${tag}</span>`).join("");
}

function renderStateSummary() {
  const current = appState.currentState;
  const root = document.querySelector("#state-summary-card");
  if (!current) return;
  const insights = currentInsights();
  root.innerHTML = `
    <h4>${insights.decision_type}</h4>
    <p>${insights.decision_hint}</p>
    <p class="muted" style="margin-top:10px;">${insights.risk_reason}</p>
  `;
}

function renderResources() {
  const current = appState.currentState;
  if (!current) return;
  const grid = document.querySelector("#resource-grid");
  grid.innerHTML = "";
  const resources = current.resources;
  [
    ["当前盲注", resources.blind_label || resources.blind_text || resources.blind, "当前回合目标"],
    ["筹码进度", `${formatNumber(resources.score_chips, 0)} / ${formatNumber(resources.target_chips, 0)}`, "越接近目标越安全"],
    ["剩余手数", `${resources.hands_left}`, resources.hands_left <= 1 ? "资源紧张" : "仍有试错空间"],
    ["剩余弃牌", `${resources.discards_left}`, resources.discards_left <= 1 ? "需要慎用" : "可继续优化牌型"],
    ["金币", `$${formatNumber(resources.money, 0)}`, resources.money <= 2 ? "经济偏紧" : "仍可支撑后续回合"],
    ["回合编号", `Ante ${resources.ante} / Round ${resources.round_num}`, "用于解释当前难度阶段"],
    ["重掷成本", `$${formatNumber(resources.reroll_cost, 0)}`, "商店阶段相关"],
    ["上一手牌型", current.score.last_hand_type || "无", "帮助解释当前节奏"],
  ].forEach(([label, value, subvalue]) => grid.appendChild(metricCard(label, value, subvalue)));

  const zoneCounts = document.querySelector("#zone-counts");
  zoneCounts.innerHTML = "";
  [
    ["牌堆", current.zones.deck_count, "剩余可抽牌数"],
    ["弃牌堆", current.zones.discard.length, "已弃出的牌"],
    ["已打出", current.zones.played.length, "本轮已使用"],
    ["Joker", current.jokers.length, "当前生效中的 Joker"],
  ].forEach(([label, value, subvalue]) => zoneCounts.appendChild(metricCard(label, `${value}`, subvalue)));
}

function renderHand() {
  const current = appState.currentState;
  if (!current) return;
  document.querySelector("#hand-meta").textContent = `${current.zones.hand.length} 张可见手牌`;
  const container = document.querySelector("#hand-cards");
  container.innerHTML = "";
  const selectedIndices = new Set((getSelectedRecommendation()?.action?.indices || []).map((value) => Number(value)));
  current.zones.hand.forEach((card) => {
    const node = document.createElement("div");
    node.className = `playing-card${selectedIndices.has(Number(card.index)) ? " selected" : ""}`;
    node.innerHTML = `
      <span class="card-index">#${Number(card.index) + 1}</span>
      <div class="card-rank">${card.rank}</div>
      <div class="card-suit">${card.suit_text}</div>
      <div class="card-note">${card.status_text || "标准牌"}</div>
    `;
    container.appendChild(node);
  });

  const jokerList = document.querySelector("#joker-list");
  jokerList.innerHTML = "";
  if (!current.jokers.length) {
    jokerList.innerHTML = emptyState("当前没有 Joker", "这个场景更适合解释纯粹的出牌 / 弃牌决策。");
    return;
  }
  current.jokers.forEach((joker) => {
    const chip = document.createElement("div");
    chip.className = "note-card";
    chip.innerHTML = `
      <strong>${joker.label_zh || joker.label}</strong>
      <p>${joker.effect_text || "当前局面存在 Joker 协同。推荐会把这部分收益算进去。"}</p>
    `;
    jokerList.appendChild(chip);
  });
}

function recommendationCard(recommendation, source, index) {
  const isSelected =
    appState.selectedRecommendation.source === source && appState.selectedRecommendation.index === index;
  const riskClass =
    recommendation.risk_level === "高" ? "high" : recommendation.risk_level === "低" ? "low" : "";
  const chipsDelta = recommendation.preview?.expected_score || recommendation.preview?.reward || 0;
  return `
    <div class="recommendation-card${isSelected ? " active" : ""}" data-source="${source}" data-index="${index}">
      <h4>#${recommendation.rank} ${recommendation.label}</h4>
      <div class="recommendation-meta">
        <span class="recommendation-chip">${recommendation.source_text || recommendation.source_label || source}</span>
        <span class="recommendation-chip ${riskClass}">风险 ${recommendation.risk_level}</span>
        <span class="recommendation-chip">置信度 ${formatPct(recommendation.confidence)}</span>
        <span class="recommendation-chip">预计 ${formatNumber(chipsDelta, 0)} 筹码</span>
      </div>
      <p>${recommendation.reason}</p>
      <div class="recommendation-tags" style="margin-top:10px;">
        ${(recommendation.tags || []).map((tag) => `<span class="chip">${tag}</span>`).join("")}
      </div>
      <p class="muted" style="margin-top:10px;">${recommendation.why_not_next || recommendation.risk_hint || recommendation.summary || ""}</p>
    </div>
  `;
}

function renderRecommendations() {
  ["model", "heuristic"].forEach((source) => {
    const container = document.querySelector(`#recommendations-${source}`);
    const list = appState.compareRecommendations[source] || [];
    if (!list.length) {
      container.innerHTML = emptyState("当前阶段没有候选动作", "切换场景或执行一步后再试。");
      return;
    }
    container.innerHTML = list.map((entry, index) => recommendationCard(entry, source, index)).join("");
    container.querySelectorAll(".recommendation-card").forEach((node) => {
      node.addEventListener("click", () => {
        appState.selectedRecommendation = {
          source: node.dataset.source,
          index: Number(node.dataset.index),
        };
        renderRecommendations();
        renderHand();
        renderPreview();
      });
    });
  });
}

function renderPreview() {
  const summaryRoot = document.querySelector("#preview-summary");
  const deltaRoot = document.querySelector("#preview-delta");
  summaryRoot.innerHTML = "";
  deltaRoot.innerHTML = "";
  const recommendation = getSelectedRecommendation();
  if (!recommendation) {
    summaryRoot.innerHTML = emptyState("请选择一个推荐动作", "点击左右任意推荐卡，下面会联动展示一步后的预期变化。");
    return;
  }

  const preview = recommendation.preview || {};
  const delta = preview.delta || {};
  summaryRoot.innerHTML = `
    <h4>${recommendation.label}</h4>
    <p>${recommendation.reason}</p>
    <div class="pill-row" style="margin-top:12px;">
      <span class="chip">${recommendation.source_text || recommendation.source_label || recommendation.source}</span>
      <span class="chip">阶段将变为：${preview.phase_after_text || preview.phase_after || "-"}</span>
      <span class="chip">预计得分：${formatNumber(preview.expected_score || preview.reward || 0, 0)}</span>
      <span class="chip">预计牌型：${preview.expected_hand_type || "-"}</span>
    </div>
    <div class="preview-highlight">
      <span class="button-icon">${icon("trend")}</span>
      ${recommendation.why_not_next || recommendation.risk_hint || recommendation.summary || ""}
    </div>
  `;
  [
    ["手数变化", delta.hands_left || 0, "负数表示会消耗出牌次数"],
    ["弃牌变化", delta.discards_left || 0, "负数表示会消耗弃牌次数"],
    ["回合筹码", formatNumber(delta.round_chips || 0, 0), "一步后本回合筹码变化"],
    ["总筹码", formatNumber(delta.score_chips || 0, 0), "整体得分变化"],
    ["金币变化", formatNumber(delta.money || 0, 0), "经济变化"],
    ["牌堆变化", formatNumber(delta.deck_count || 0, 0), "牌堆剩余变化"],
    ["弃牌堆变化", formatNumber(delta.discard_count || 0, 0), "弃牌堆数量变化"],
    ["已打出变化", formatNumber(delta.played_count || 0, 0), "已打出区域变化"],
  ].forEach(([label, value, subvalue]) => {
    deltaRoot.appendChild(metricCard(label, `${value}`, subvalue));
  });
}

function renderTimeline() {
  const container = document.querySelector("#timeline");
  const timeline = appState.currentState?.timeline || [];
  if (!timeline.length) {
    container.innerHTML = emptyState("时间线为空", "载入场景、执行动作之后，这里会自动记录最近步骤。");
    return;
  }
  container.innerHTML = timeline
    .slice()
    .reverse()
    .map((entry) => {
      const sceneKind = entry.kind === "scenario_loaded" ? "scene" : "action";
      const subtitle =
        entry.kind === "scenario_loaded"
          ? entry.summary || entry.focus || ""
          : `${entry.delta?.phase_before || entry.phase_before || "-"} → ${
              entry.delta?.phase_after || entry.phase_after || "-"
            } ｜ 收益 ${formatNumber(entry.reward || 0, 0)}`;
      return `
        <div class="timeline-entry">
          <span class="timeline-kind ${sceneKind}">${entry.kind_label || entry.kind}</span>
          <h4>${entry.label}</h4>
          <p>${subtitle}</p>
        </div>
      `;
    })
    .join("");
}

function buildLine(points, width, height, padding, accessor, minValue, maxValue) {
  if (!points.length) return "";
  const xScale = (index) =>
    padding + (index / Math.max(points.length - 1, 1)) * Math.max(0, width - padding * 2);
  const yScale = (value) =>
    height - padding - ((value - minValue) / Math.max(maxValue - minValue, 1e-6)) * Math.max(0, height - padding * 2);
  return points.map((point, index) => `${xScale(index)},${yScale(Number(accessor(point) || 0))}`).join(" ");
}

function renderTrainingChart(history) {
  const chartRoot = document.querySelector("#training-chart");
  if (!history.length) {
    chartRoot.innerHTML = emptyState("还没有训练曲线", "启动一次训练后，这里会显示 train / val loss 的变化。");
    return;
  }
  const width = 720;
  const height = 240;
  const padding = 24;
  const values = history.flatMap((point) => [Number(point.train_loss || 0), Number(point.val_loss || 0)]);
  const maxValue = Math.max(...values, 1);
  const minValue = Math.min(...values, 0);
  const trainLine = buildLine(history, width, height, padding, (point) => point.train_loss, minValue, maxValue);
  const valLine = buildLine(history, width, height, padding, (point) => point.val_loss, minValue, maxValue);

  chartRoot.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <line x1="${padding}" y1="${height - padding}" x2="${width - padding}" y2="${height - padding}" stroke="currentColor" opacity="0.12" />
      <line x1="${padding}" y1="${padding}" x2="${padding}" y2="${height - padding}" stroke="currentColor" opacity="0.12" />
      <polyline fill="none" stroke="#d7a94f" stroke-width="3" points="${trainLine}"></polyline>
      <polyline fill="none" stroke="#0e7c66" stroke-width="3" points="${valLine}"></polyline>
      <text x="${padding}" y="18" fill="currentColor" opacity="0.65" font-size="12">训练损失（黄） / 验证损失（绿）</text>
      <text x="${padding}" y="${height - 8}" fill="currentColor" opacity="0.65" font-size="12">第 1 轮</text>
      <text x="${width - 96}" y="${height - 8}" fill="currentColor" opacity="0.65" font-size="12">第 ${history.length} 轮</text>
    </svg>
  `;
}

function renderTrainingNotes() {
  const root = document.querySelector("#training-side-notes");
  const training = appState.trainingStatus || {};
  const modelInfo = appState.modelInfo || {};
  const scenarioEval =
    modelInfo.scenario_eval?.results ||
    modelInfo.scenario_eval?.scenarios ||
    training.evaluation?.results ||
    training.scenario_eval?.results ||
    [];
  const notes = [];
  const trainingLabel =
    training.status === "idle"
      ? statusText("idle")
      : readableStatusLabel(training.status_label, training.status);
  notes.push(`
    <div class="note-card">
      <strong>训练状态</strong>
      <p>${trainingLabel} / ${stageText(trainingStage())}</p>
    </div>
  `);
  if (modelInfo.verdict?.new_best_val_loss || modelInfo.verdict?.improved_over_previous !== undefined) {
    notes.push(`
      <div class="note-card">
        <strong>与上一版对比</strong>
        <p>${
          modelInfo.verdict?.improved_over_previous
            ? "新模型优于上一版，可直接作为默认展示模型。"
            : "新模型未显著优于上一版，适合展示训练过程与评估闭环。"
        }</p>
      </div>
    `);
  }
  if (scenarioEval.length) {
    notes.push(
      ...scenarioEval.slice(0, 3).map(
        (row) => `
        <div class="note-card">
          <strong>${row.scenario_name}</strong>
          <p>${row.top1_label || row.top_label || "暂无推荐"}</p>
          <p class="muted">教师一致：${row.teacher_agrees ? "是" : "否"} ｜ 预计 ${formatNumber(
            row.expected_score,
            0
          )}</p>
        </div>
      `
      )
    );
  }
  root.innerHTML = notes.join("");
}

function renderTrainingPanel() {
  const training = appState.trainingStatus || {};
  const modelInfo = appState.modelInfo || {};
  const statusRoot = document.querySelector("#training-status-card");
  const modelMeta = document.querySelector("#model-meta");
  const history = trainingHistoryRows();
  const progressValue = trainingProgressValue();
  const displayStatusLabel =
    training.status === "idle"
      ? statusText("idle")
      : readableStatusLabel(training.status_label, training.status || "idle");
  const displayMessage =
    training.status === "idle"
      ? "当前没有训练任务。可在页面中启动快速烟雾训练或 2 小时训练。"
      : training.message || "尚未启动训练。当前界面会展示最近一次已完成的模型结果。";

  statusRoot.innerHTML = `
    <h4>${displayStatusLabel} / ${stageText(trainingStage())}</h4>
    <p>${displayMessage}</p>
    <div class="progress-shell" style="margin-top:12px;">
      <div class="progress-bar" style="width:${progressValue * 100}%"></div>
    </div>
    <div class="pill-row" style="margin-top:12px;">
      <span class="chip">任务 ID：${training.job_id || "-"}</span>
      <span class="chip">阶段：${stageText(trainingStage())}</span>
      <span class="chip">预计剩余：${formatDuration(training.training?.eta_sec || training.eta_sec)}</span>
      <span class="chip">最佳轮次：${training.training?.best_epoch || modelInfo.metrics?.best_epoch || "-"}</span>
      <span class="chip">最佳损失：${formatNumber(training.training?.best_val_loss || modelInfo.metrics?.best_val_loss || 0, 3)}</span>
    </div>
  `;

  modelMeta.innerHTML = "";
  [
    ["当前运行 ID", currentRunId(), modelInfo.loaded ? "当前已由页面加载" : "当前仍回退到启发式推荐"],
    [
      "模型状态",
      modelInfo.loaded ? "已加载" : "未加载",
      modelInfo.loaded ? "当前模型已成功接管推荐。" : "旧模型文件与当前架构不兼容，可直接在界面中发起新训练。",
    ],
    ["样本规模", `${modelInfo.dataset_stats?.total_records || modelInfo.metrics?.train_samples || "-"}`, "数据集总样本"],
    ["最优验证损失", formatNumber(modelInfo.metrics?.best_val_loss || 0, 3), "越低越好"],
    ["验证 Top-1", formatPct(modelInfo.metrics?.final?.val_acc1 || 0), "单点命中率"],
    ["验证 Top-3", formatPct(modelInfo.metrics?.final?.val_acc3 || 0), "前三命中率"],
    ["设备", modelInfo.config?.device_used || "-", "最近一次训练设备"],
  ].forEach(([label, value, subvalue]) => modelMeta.appendChild(metricCard(label, value, subvalue)));

  renderTrainingChart(history);
  renderTrainingNotes();
}

function refreshButtonStates() {
  const selected = getSelectedRecommendation();
  const trainingRunning = trainingIsActive();
  document.querySelector("#step-button").disabled = appState.busy.action || !selected;
  document.querySelector("#autoplay-button").disabled = appState.busy.action;
  document.querySelector("#reload-button").disabled = appState.busy.action;
  document.querySelector("#train-standard-button").disabled = trainingRunning;
  document.querySelector("#train-smoke-button").disabled = trainingRunning;
}

function renderAll() {
  renderTopSummary();
  renderStatusStrip();
  renderScenarios();
  renderInsightTags();
  renderStateSummary();
  renderResources();
  renderHand();
  renderRecommendations();
  renderPreview();
  renderTimeline();
  renderTrainingPanel();
  refreshButtonStates();
  renderIconSlots();
}

async function refreshState() {
  appState.currentState = await apiFetch("/api/state");
  appState.currentScenarioId = appState.currentState.scenario.id;
}

async function refreshModelInfo() {
  appState.modelInfo = await apiFetch("/api/model_info");
}

async function refreshTrainingStatus() {
  appState.trainingStatus = await apiFetch("/api/training/status");
}

async function refreshRecommendations() {
  const [modelPayload, heuristicPayload] = await Promise.all([
    apiFetch("/api/recommend", {
      method: "POST",
      body: JSON.stringify({ policy: "model", topk: 3 }),
    }),
    apiFetch("/api/recommend", {
      method: "POST",
      body: JSON.stringify({ policy: "heuristic", topk: 3 }),
    }),
  ]);
  appState.compareRecommendations.model = modelPayload.recommendations || [];
  appState.compareRecommendations.heuristic = heuristicPayload.recommendations || [];
  ensureSelectedRecommendation();
}

async function loadScenario(scenarioId) {
  appState.busy.action = true;
  refreshButtonStates();
  try {
    await apiFetch("/api/scenario/load", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId }),
    });
    await refreshState();
    await refreshRecommendations();
    renderAll();
    setStatusMessage(appState.currentState?.scenario?.talk_track || "");
    showToast(`已切换到场景：${appState.currentState?.scenario?.name}`, "success");
  } catch (error) {
    showToast(error.message, "error");
    throw error;
  } finally {
    appState.busy.action = false;
    refreshButtonStates();
  }
}

async function executeSelected() {
  const recommendation = getSelectedRecommendation();
  if (!recommendation) return;
  appState.busy.action = true;
  refreshButtonStates();
  try {
    await apiFetch("/api/step", {
      method: "POST",
      body: JSON.stringify({ policy: recommendation.source, action: recommendation.action }),
    });
    await refreshState();
    await refreshRecommendations();
    renderAll();
    showToast(`已执行：${recommendation.label}`, "success");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    appState.busy.action = false;
    refreshButtonStates();
  }
}

async function autoplay() {
  appState.busy.action = true;
  refreshButtonStates();
  try {
    await apiFetch("/api/autoplay", {
      method: "POST",
      body: JSON.stringify({ policy: "model", steps: 3 }),
    });
    await refreshState();
    await refreshRecommendations();
    renderAll();
    showToast("自动演示完成。", "success");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    appState.busy.action = false;
    refreshButtonStates();
  }
}

async function startTraining(profile) {
  appState.busy.training = true;
  refreshButtonStates();
  try {
    const payload = await apiFetch("/api/training/start", {
      method: "POST",
      body: JSON.stringify({ profile }),
    });
    appState.trainingStatus = payload;
    renderTrainingPanel();
    refreshButtonStates();
    showToast(profile === "smoke" ? "快速烟雾训练已启动。" : "2 小时预算训练已在后台启动。", "info");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    appState.busy.training = false;
    refreshButtonStates();
  }
}

function bindControls() {
  document.querySelector("#reload-button").addEventListener("click", async () => {
    if (appState.currentScenarioId) {
      await loadScenario(appState.currentScenarioId);
    }
  });
  document.querySelector("#step-button").addEventListener("click", executeSelected);
  document.querySelector("#autoplay-button").addEventListener("click", autoplay);
  document.querySelector("#train-standard-button").addEventListener("click", async () => {
    await startTraining("standard");
  });
  document.querySelector("#train-smoke-button").addEventListener("click", async () => {
    await startTraining("smoke");
  });
}

async function refreshAll(initial = false) {
  const scenariosPayload = initial ? await apiFetch("/api/scenarios") : { scenarios: appState.scenarios };
  if (initial) {
    appState.scenarios = scenariosPayload.scenarios || [];
  }
  const previousTrainingStatus = trainingStage();
  await Promise.all([refreshModelInfo(), refreshState(), refreshTrainingStatus()]);
  await refreshRecommendations();
  renderAll();
  setStatusMessage(appState.currentState?.scenario?.talk_track || "");

  if (trainingIsActive(previousTrainingStatus) && appState.trainingStatus?.status === "finished") {
    await refreshModelInfo();
    await refreshRecommendations();
    renderAll();
    showToast("训练完成，页面已刷新到最新模型。", "success");
  }
}

function startPolling() {
  window.setInterval(async () => {
    appState.pollCounter += 1;
    try {
      const previousStatus = trainingStage();
      await refreshTrainingStatus();
      if (trainingIsActive()) {
        renderTrainingPanel();
        refreshButtonStates();
      }
      if (appState.pollCounter % 3 === 0) {
        await refreshModelInfo();
        renderTrainingPanel();
      }
      if (trainingIsActive(previousStatus) && appState.trainingStatus?.status === "finished") {
        await refreshModelInfo();
        await refreshRecommendations();
        renderAll();
      }
    } catch (error) {
      console.warn(error);
    }
  }, 2500);
}

async function bootstrap() {
  initTheme();
  bindControls();
  renderIconSlots();
  await refreshAll(true);
  startPolling();
}

bootstrap().catch((error) => {
  document.body.innerHTML = `
    <div class="app-shell">
      <div class="panel" style="padding:24px;">
        ${emptyState("演示页启动失败", error.message)}
      </div>
    </div>
  `;
});
