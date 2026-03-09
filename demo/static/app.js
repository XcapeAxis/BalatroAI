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
    signature: "",
  },
  busy: {
    action: false,
    training: false,
  },
  theme: "light",
  pollCounter: 0,
  ui: {
    inspectionEntry: null,
    viewMode: "overview",
    focusPanel: "preview",
  },
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
  layers: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m12 3 9 4.5-9 4.5-9-4.5z"/><path d="m3 12 9 4.5 9-4.5"/><path d="m3 16.5 9 4.5 9-4.5"/></svg>`,
  focus: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M9 3H5a2 2 0 0 0-2 2v4"/><path d="M15 3h4a2 2 0 0 1 2 2v4"/><path d="M21 15v4a2 2 0 0 1-2 2h-4"/><path d="M3 15v4a2 2 0 0 0 2 2h4"/><circle cx="12" cy="12" r="3.2"/></svg>`,
  scene: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="4" width="18" height="16" rx="3"/><path d="m7 14 2.8-3 3.1 3.7 2.2-2.5L19 16"/><circle cx="8.5" cy="8.5" r="1.2"/></svg>`,
  stage: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 20h18"/><path d="M6 20v-8l6-4 6 4v8"/><path d="M9.5 10.8 12 9l2.5 1.8"/></svg>`,
  risk: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3 2.8 19h18.4L12 3Z"/><path d="M12 9v4.5"/><circle cx="12" cy="16.8" r="0.9" fill="currentColor" stroke="none"/></svg>`,
  decision: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 4h14"/><path d="M12 4v16"/><path d="m12 8 5 4-5 4"/><path d="m12 8-5 4 5 4"/></svg>`,
  target: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="1.4" fill="currentColor" stroke="none"/></svg>`,
  compare: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 5h6v14H5z"/><path d="M13 5h6v14h-6z"/><path d="M8 9v6"/><path d="M16 8v8"/><path d="M15 12h2"/></svg>`,
  model: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 3 4 7v10l8 4 8-4V7l-8-4Z"/><path d="M12 12 4 7"/><path d="M12 12l8-5"/><path d="M12 12v9"/></svg>`,
  heuristic: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 12h8"/><path d="m8 8 4 4-4 4"/><path d="M14 6h6"/><path d="M14 12h6"/><path d="M14 18h6"/></svg>`,
  preview: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6Z"/><circle cx="12" cy="12" r="2.8"/></svg>`,
  result: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 19V9"/><path d="M12 19V5"/><path d="M19 19v-7"/></svg>`,
  timeline: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M12 6v12"/><path d="M7 10h10"/><circle cx="12" cy="6" r="1.8"/><circle cx="12" cy="18" r="1.8"/></svg>`,
  training: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M4 19h16"/><path d="M7 16V9"/><path d="M12 16V5"/><path d="M17 16v-3"/></svg>`,
  hand: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M7 11V5.5a1.5 1.5 0 0 1 3 0V10"/><path d="M10 10V4.8a1.4 1.4 0 0 1 2.8 0V10"/><path d="M12.8 10V5.6a1.4 1.4 0 1 1 2.8 0V12"/><path d="M7 11 5.2 9.8a1.6 1.6 0 0 0-2.3 2L6 18.5A3 3 0 0 0 8.7 20H15a3 3 0 0 0 3-3v-5"/></svg>`,
  joker: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M7 19c0-2 1.5-3.5 3.5-3.5S14 17 14 19"/><path d="M10.5 15.5V13"/><path d="M6 8c1.5 0 2.5-1 2.5-2.5S7.5 3 6 3C4.5 3 3.5 4 3.5 5.5S4.5 8 6 8Z"/><path d="M18 8c1.5 0 2.5-1 2.5-2.5S19.5 3 18 3c-1.5 0-2.5 1-2.5 2.5S16.5 8 18 8Z"/><path d="M12 5c1.5 0 2.5-1 2.5-2.5S13.5 0 12 0 9.5 1 9.5 2.5 10.5 5 12 5Z" transform="translate(0 3)"/><path d="M6 8c0 3 2.5 5 6 5s6-2 6-5"/></svg>`,
  resources: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 7h14"/><path d="M5 12h14"/><path d="M5 17h9"/><path d="M16.5 17.5 18 19l3-4"/></svg>`,
  chips: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="8"/><path d="M12 4v4"/><path d="M12 16v4"/><path d="M4 12h4"/><path d="M16 12h4"/></svg>`,
  blind: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M6 20V9"/><path d="M6 9c4 0 6-3 12-3v8c-6 0-8 3-12 3"/><path d="M6 6V4"/></svg>`,
  discard: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m7 7 10 10"/><path d="M17 7 7 17"/><rect x="4" y="4" width="16" height="16" rx="3"/></svg>`,
  money: `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><rect x="3" y="6" width="18" height="12" rx="2"/><circle cx="12" cy="12" r="2.2"/><path d="M7 12h.01M17 12h.01"/></svg>`,
};

function icon(name) {
  return ICONS[name] || "";
}

function qs(selector) {
  return document.querySelector(selector);
}

function actionSignature(action) {
  return JSON.stringify(action || {});
}

function safeNumber(value, fallback = 0) {
  const numeric = Number(value);
  return Number.isFinite(numeric) ? numeric : fallback;
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

function emptyState(title, description) {
  return `
    <div class="empty-state">
      <strong>${title}</strong>
      <p>${description}</p>
    </div>
  `;
}

function metricCard(label, value, subvalue = "", iconName = "trend") {
  const template = qs("#metric-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".metric-label").textContent = label;
  node.querySelector(".metric-value").textContent = value;
  node.querySelector(".metric-subvalue").textContent = subvalue;
  return node;
}

function showToast(message, type = "info") {
  const root = qs("#toast-root");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  root.appendChild(toast);
  setTimeout(() => toast.remove(), 3200);
}

function trimText(text, maxLength = 88) {
  const cleaned = String(text || "").trim();
  if (!cleaned) return "";
  return cleaned.length <= maxLength ? cleaned : `${cleaned.slice(0, maxLength - 1)}…`;
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

function stageText(stage) {
  const mapping = {
    idle: "空闲",
    queued: "等待开始",
    building_dataset: "构建数据集",
    sweep: "超参试跑",
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
    running: "运行中",
    finished: "已完成",
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

function renderIconSlots(root = document) {
  root.querySelectorAll("[data-icon]").forEach((node) => {
    node.innerHTML = icon(node.dataset.icon);
  });
}

function renderViewModeControls() {
  const app = qs("#app");
  if (!app) return;
  app.dataset.viewMode = "overview";
  app.dataset.focusPanel = appState.ui.focusPanel;
  document.querySelectorAll(".dock-tab[data-focus-panel]").forEach((button) => {
    button.classList.toggle("active", button.dataset.focusPanel === appState.ui.focusPanel);
  });
}

function setViewMode(mode) {
  appState.ui.viewMode = "overview";
  renderViewModeControls();
}

function setFocusPanel(panel, options = {}) {
  appState.ui.focusPanel = panel || "preview";
  if (options.viewMode === "focus") {
    appState.ui.viewMode = "focus";
  }
  renderViewModeControls();
}

function applyTheme(theme) {
  appState.theme = theme;
  document.body.dataset.theme = theme;
  localStorage.setItem("balatro_mvp_theme", theme);
  const toggle = qs("#theme-toggle");
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
  const node = qs("#status-message");
  if (node) {
    node.textContent = trimText(text || "", 80);
  }
}

function currentRunId() {
  const runDir = appState.modelInfo?.run_dir || "";
  if (!runDir) return "-";
  const normalized = String(runDir).replaceAll("/", "\\");
  return normalized.split("\\").pop() || "-";
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

function trainingStage(status = appState.trainingStatus?.status) {
  return status || "idle";
}

function trainingIsActive(status = appState.trainingStatus?.status) {
  return ["queued", "building_dataset", "training", "evaluating", "running"].includes(status || "");
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

function scenarioEvalRows() {
  return (
    appState.modelInfo?.scenario_eval?.results ||
    appState.modelInfo?.verdict?.scenario_eval?.results ||
    []
  );
}

function riskWeight(level) {
  const label = String(level || "");
  if (label.includes("高")) return 3;
  if (label.includes("中")) return 2;
  if (label.includes("低")) return 1;
  return 0;
}

function riskClass(level) {
  const label = String(level || "");
  if (label.includes("高")) return "high";
  if (label.includes("中")) return "medium";
  if (label.includes("低")) return "low";
  return "";
}

function sourceIconName(source) {
  return source === "heuristic" ? "heuristic" : "model";
}

function decisionIconName(text) {
  const value = String(text || "");
  if (value.includes("弃牌")) return "discard";
  if (value.includes("资源")) return "resources";
  if (value.includes("Joker")) return "joker";
  return "decision";
}

function scenarioIconName(scenario) {
  if (scenario?.has_jokers || String(scenario?.focus || "").includes("Joker")) return "joker";
  if (String(scenario?.focus || "").includes("弃牌")) return "discard";
  return "scene";
}

function riskText(level) {
  const value = String(level || "").trim();
  if (!value) return "-";
  return value.includes("风险") ? value : `${value}风险`;
}

function resourceIconName(label) {
  const mapping = {
    "当前盲注": "blind",
    "当前筹码": "chips",
    "距离目标": "target",
    "剩余出牌": "hand",
    "剩余弃牌": "discard",
    "现金": "money",
    "当前轮次": "timeline",
    "重掷成本": "money",
    "上一手牌型": "stage",
    "牌堆": "hand",
    "弃牌堆": "discard",
    "已打出": "play",
    "Joker": "joker",
    "回合筹码": "chips",
    "总筹码": "target",
    "出牌次数": "hand",
    "弃牌次数": "discard",
    "现金变化": "money",
    "牌堆变化": "hand",
    "弃牌堆变化": "discard",
    "已打出变化": "play",
    "当前运行 ID": "model",
    "模型状态": "model",
    "样本规模": "training",
    "最佳验证损失": "trend",
    "Top-1 命中": "target",
    "Top-3 命中": "compare",
    "训练设备": "training",
    "当前 checkpoint": "training",
  };
  return mapping[label] || "trend";
}

function expectedScore(recommendation) {
  return safeNumber(recommendation?.preview?.expected_score ?? recommendation?.preview?.reward);
}

function recommendationPhaseAfter(recommendation) {
  return recommendation?.preview?.phase_after_text || recommendation?.preview?.phase_after || "-";
}

function normalizeExplanation(text) {
  let value = String(text || "").trim();
  if (!value) return "";
  if (/^[a-z_]+_score=/i.test(value)) return "";
  value = value.replace(/教师策略也支持这个动作（[^）]+）/g, "教师策略也支持这一步。");
  value = value.replace(/教师策略会优先\s*/g, "教师策略会优先 ");
  value = value.replace(/（[a-z_]+_score=[^)]+）/gi, "");
  value = value.replace(/。。+/g, "。");
  value = value.replace(/\s+\./g, "。");
  value = value.replace(/\s+/g, " ").trim();
  return value;
}

function contrastReason(recommendation) {
  const cleaned = normalizeExplanation(recommendation?.why_not_next);
  if (cleaned) return cleaned;
  if (recommendation?.teacher_agrees) {
    return "下一名虽然也能走通，但当前这一步更完整地兑现了手里的价值。";
  }
  return "更稳的方案会把这一步的收益压低，所以这条动作更适合用来讲模型和基线的分歧。";
}

function getSelectedRecommendation() {
  const { source, index } = appState.selectedRecommendation;
  return (appState.compareRecommendations[source] || [])[index] || null;
}

function recommendationExistsBySignature(signature) {
  if (!signature) return null;
  for (const [source, list] of Object.entries(appState.compareRecommendations)) {
    const foundIndex = list.findIndex((entry) => actionSignature(entry.action) === signature);
    if (foundIndex >= 0) {
      return { source, index: foundIndex };
    }
  }
  return null;
}

function ensureSelectedRecommendation() {
  const current = getSelectedRecommendation();
  if (current) {
    appState.selectedRecommendation.signature = actionSignature(current.action);
    return;
  }
  const preferred = recommendationExistsBySignature(appState.selectedRecommendation.signature);
  if (preferred) {
    appState.selectedRecommendation = {
      ...preferred,
      signature: actionSignature(appState.compareRecommendations[preferred.source][preferred.index]?.action),
    };
    return;
  }
  if ((appState.compareRecommendations.model || []).length) {
    appState.selectedRecommendation = {
      source: "model",
      index: 0,
      signature: actionSignature(appState.compareRecommendations.model[0]?.action),
    };
    return;
  }
  if ((appState.compareRecommendations.heuristic || []).length) {
    appState.selectedRecommendation = {
      source: "heuristic",
      index: 0,
      signature: actionSignature(appState.compareRecommendations.heuristic[0]?.action),
    };
  }
}

function compareSummaryData() {
  const modelTop = appState.compareRecommendations.model?.[0];
  const heuristicTop = appState.compareRecommendations.heuristic?.[0];
  if (!modelTop && !heuristicTop) return null;
  if (!modelTop || !heuristicTop) {
    const single = modelTop || heuristicTop;
    return {
      title: "当前只有一条建议可用",
      lead: single?.label || "暂无建议",
      copy: "这个阶段暂时没有形成模型 vs 基线的直接对比，页面会先把可执行建议讲清楚。",
      same: true,
      sameText: "单路建议",
      scoreDiffText: `${formatNumber(expectedScore(single), 0)} 筹码`,
      postureText: "暂不比较",
      modelTop,
      heuristicTop,
    };
  }
  const same = actionSignature(modelTop.action) === actionSignature(heuristicTop.action);
  const modelScore = expectedScore(modelTop);
  const heuristicScore = expectedScore(heuristicTop);
  const diff = modelScore - heuristicScore;
  const modelRisk = riskWeight(modelTop.risk_level);
  const heuristicRisk = riskWeight(heuristicTop.risk_level);
  const modelAlt = appState.compareRecommendations.model?.[1];
  const heuristicAlt = appState.compareRecommendations.heuristic?.[1];
  let postureText = "风险取向接近";
  if (modelRisk > heuristicRisk) postureText = "模型更激进";
  else if (modelRisk < heuristicRisk) postureText = "模型更保守";
  else if (diff > 0) postureText = "模型更看重当下收益";
  else if (diff < 0) postureText = "启发式更看重当下收益";

  if (same) {
    const altDifferent =
      modelAlt &&
      heuristicAlt &&
      actionSignature(modelAlt.action) !== actionSignature(heuristicAlt.action);
    if (altDifferent) {
      const altModelRisk = riskWeight(modelAlt.risk_level);
      const altHeuristicRisk = riskWeight(heuristicAlt.risk_level);
      let altPosture = "备选思路不同";
      if (altModelRisk > altHeuristicRisk) altPosture = "模型的备选更激进";
      else if (altModelRisk < altHeuristicRisk) altPosture = "模型的备选更保守";
      else if (expectedScore(modelAlt) > expectedScore(heuristicAlt)) altPosture = "模型的备选更看收益";
      else if (expectedScore(modelAlt) < expectedScore(heuristicAlt)) altPosture = "启发式的备选更看收益";
      return {
        title: "首选一致，但两边的备选思路已经开始分开",
        lead: `${altPosture}。模型第二选择是“${modelAlt.label}”，基线第二选择是“${heuristicAlt.label}”。`,
        copy: "这很适合现场讲：即便两边都同意第一步怎么走，训练模型和基线对“如果第一步不能用”这件事的理解已经不一样了。",
        same,
        sameText: "首选一致",
        scoreDiffText: `备选收益差 ${formatNumber(expectedScore(modelAlt) - expectedScore(heuristicAlt), 0)}`,
        postureText: altPosture,
        modelTop,
        heuristicTop,
      };
    }
    return {
      title: "模型和基线现在给出同一个答案",
      lead: modelTop.label,
      copy: "这个场景足够清晰，演示重点可以直接放到执行后会发生什么，而不是谁赢谁输。",
      same,
      sameText: "一致",
      scoreDiffText: `收益差 ${formatNumber(diff, 0)}`,
      postureText,
      modelTop,
      heuristicTop,
    };
  }

  const diffDirection = diff > 0 ? "更愿意现在兑现收益" : diff < 0 ? "更像在保留后手" : "收益判断不同";
  return {
    title: "模型和基线出现分歧，这正是最值得讲的一步",
    lead: `${postureText}，${diffDirection}`,
    copy: `模型首选“${modelTop.label}”，基线首选“${heuristicTop.label}”。这块最适合讲为什么这一步不能只看眼前筹码。`,
    same,
    sameText: "不同",
    scoreDiffText: `模型比基线 ${diff >= 0 ? "多" : "少"}看 ${formatNumber(Math.abs(diff), 0)} 筹码`,
    postureText,
    modelTop,
    heuristicTop,
  };
}

function trainingNarrative(modelInfo, trainingStatus) {
  const evalRows = scenarioEvalRows();
  const teacherAgreements = evalRows.filter((row) => row.teacher_agrees).length;
  const totalRows = evalRows.length;
  const top1 = safeNumber(modelInfo?.metrics?.final?.val_acc1);
  const top3 = safeNumber(modelInfo?.metrics?.final?.val_acc3);
  const loaded = Boolean(modelInfo?.loaded);

  let title = loaded ? "这版模型已经接管演示推荐" : "当前仍在回退到启发式";
  let lead = loaded
    ? `现在页面里的“模型建议”来自真实 checkpoint：${modelInfo?.model_name || "-"}`
    : "当前推荐仍可展示，但如果要强调训练成果，建议先加载或训练模型。";
  let strengths = "目前没有足够的场景评估结果。";
  let limitation = "当前更适合做场景化演示，不适合直接讲成通用最优代理。";

  if (totalRows > 0) {
    strengths =
      teacherAgreements === totalRows
        ? `在 ${teacherAgreements}/${totalRows} 个演示场景里，模型都和教师策略对齐。`
        : `在 ${teacherAgreements}/${totalRows} 个演示场景里，模型能和教师策略对齐。`;
  }

  if (top3 >= 0.16) {
    limitation = "它已经能把不少合理动作推到候选前排，但第一名稳定性仍然有限。";
  }
  if (top1 >= 0.12) {
    limitation = "Top-1 已经开始稳定，但当前版本仍然主要覆盖手牌阶段。";
  }
  if (modelInfo?.verdict?.improved_over_previous === false) {
    limitation = "这版更适合展示训练闭环和场景能力，不适合宣称已经全面优于上一版。";
  }
  if (trainingIsActive(trainingStatus?.status)) {
    lead = "后台还有一轮新训练在跑，下面这块会继续实时刷新进度和最佳 checkpoint。";
  }

  return { title, lead, strengths, limitation };
}

function buildInspectEntry(recommendation) {
  if (!recommendation) return null;
  const reward = expectedScore(recommendation);
  const actionType = String(recommendation?.action?.action_type || "").toUpperCase();
  return {
    timestamp: new Date().toISOString(),
    kind: "inspect",
    kind_label: "正在观察",
    label: `查看：${recommendation.label}`,
    summary:
      actionType === "DISCARD"
        ? `这一步先不收分，重点是换后手。预计会把局面带到 ${recommendationPhaseAfter(recommendation)}。`
        : `预计单步带来 ${formatNumber(reward, 0)} 筹码，并进入 ${recommendationPhaseAfter(recommendation)}。`,
    highlight: true,
  };
}

function syncInspectionEntry() {
  appState.ui.inspectionEntry = buildInspectEntry(getSelectedRecommendation());
}

function setPanelLoading(panelNames, active) {
  panelNames.forEach((name) => {
    const panel = document.querySelector(`[data-panel="${name}"]`);
    if (panel) panel.classList.toggle("is-loading", active);
  });
}

function pulsePanels(panelNames) {
  panelNames.forEach((name) => {
    const panel = document.querySelector(`[data-panel="${name}"]`);
    if (!panel) return;
    panel.classList.remove("is-updated");
    window.requestAnimationFrame(() => {
      panel.classList.add("is-updated");
      window.setTimeout(() => panel.classList.remove("is-updated"), 620);
    });
  });
}

function topPolicyLine(recommendation) {
  if (!recommendation) return "先选一个场景，页面会自动把最值得看的动作顶出来。";
  const score = expectedScore(recommendation);
  const actionType = String(recommendation?.action?.action_type || "").toUpperCase();
  if (actionType === "DISCARD") {
    return `${recommendation.rank === 1 ? "当前首选" : "当前正在看"}先换后手：${recommendation.label}`;
  }
  return `${recommendation.rank === 1 ? "当前首选" : "当前正在看"}的动作预计带来 ${formatNumber(score, 0)} 筹码。`;
}

function previewHeadline(recommendation) {
  if (!recommendation) {
    return {
      title: "先点一条建议，再看执行后会发生什么",
      lead: "结果区会跟着当前选中的动作实时联动。",
      copy: "这块是整页里最适合讲“为什么值得”的位置。",
    };
  }
  const preview = recommendation.preview || {};
  const reward = expectedScore(recommendation);
  const delta = preview.delta || {};
  const scoreGap = safeNumber(currentInsights().score_gap);
  const afterTotal = safeNumber(appState.currentState?.resources?.score_chips) + safeNumber(delta.score_chips);
  const actionType = String(recommendation?.action?.action_type || "").toUpperCase();

  if (actionType === "DISCARD") {
    return {
      title: "这一步先不收分，核心是把下一手换好",
      lead: `会消耗 ${Math.abs(safeNumber(delta.discards_left))} 次弃牌，给后续高质量出牌腾空间。`,
      copy: normalizeExplanation(recommendation.reason),
    };
  }
  if (reward >= scoreGap && scoreGap > 0) {
    return {
      title: "这一步足以把当前分差补上",
      lead: `预计执行后总筹码来到 ${formatNumber(afterTotal, 0)}，可以直接越过当前目标。`,
      copy: normalizeExplanation(recommendation.reason),
    };
  }
  return {
    title: `这一步预计带来 ${formatNumber(reward, 0)} 筹码`,
    lead: `执行后会进入 ${recommendationPhaseAfter(recommendation)}，适合继续讲后续资源变化。`,
    copy: normalizeExplanation(recommendation.reason),
  };
}

function renderTopSummary() {
  const current = appState.currentState;
  if (!current) return;
  qs("#status-scenario").textContent = current.scenario.name;
  qs("#status-model").textContent = appState.modelInfo?.model_name || current.model_name || "-";
  qs("#status-run-id").textContent = `运行 ID：${currentRunId()}`;
  qs("#scenario-focus-inline").textContent = current.scenario.focus || "-";
}

function renderStatusStrip() {
  return;
}

function renderScenarios() {
  const container = qs("#scenario-list");
  container.innerHTML = "";
  appState.scenarios.forEach((scenario, index) => {
    const active = scenario.id === appState.currentScenarioId ? " active" : "";
    const node = document.createElement("div");
    node.className = `scenario-card${active}`;
    node.innerHTML = `
      <div class="scenario-header">
        <div>
          <span class="section-label">场景 ${index + 1}</span>
          <h4>${scenario.name}</h4>
        </div>
        <span class="chip ${riskClass(scenario.risk_label)}">${riskText(scenario.risk_label || "中")}</span>
      </div>
      <p class="summary-sentence">重点看：${scenario.focus}</p>
      <p class="plain-meta">手数 ${scenario.hands_left} · 弃牌 ${scenario.discards_left} · 目标 ${formatNumber(
        scenario.target_chips,
        0
      )}</p>
      <div class="inline-actions" style="margin-top:12px;">
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
  const current = appState.currentState;
  const insights = currentInsights();
  const root = qs("#insight-tags");
  if (!current) {
    root.textContent = "";
    return;
  }
  root.textContent =
    current.scenario.focus ||
    insights.decision_hint ||
    "先看当前局面，再对照右侧建议，最后看下方结果摘要。";
}

function renderStageSpotlight() {
  const root = qs("#stage-spotlight");
  const selected = getSelectedRecommendation();
  if (!selected) {
    root.innerHTML = emptyState("主舞台还没有选中动作", "点右侧任意推荐卡，中央局面会立即跟着它亮起来。");
    return;
  }
  const preview = previewHeadline(selected);
  root.innerHTML = `
    <span class="section-label">当前一句话判断</span>
    <div class="callout-title">${preview.title}</div>
    <p class="callout-lead">${preview.lead}</p>
    <p class="support-copy">${trimText(
      contrastReason(selected) || selected.risk_hint || normalizeExplanation(preview.copy),
      100
    )}</p>
    <div class="fact-list">
      <div class="fact-item">
        <strong>当前动作</strong>
        <span>${selected.label}</span>
      </div>
      <div class="fact-item">
        <strong>预计收益</strong>
        <span>${formatNumber(expectedScore(selected), 0)} 筹码</span>
      </div>
    </div>
  `;
}

function renderStateSummary() {
  const current = appState.currentState;
  const root = qs("#state-summary-card");
  if (!current) return;
  const insights = currentInsights();
  root.innerHTML = `
    <span class="section-label">当前局面判断</span>
    <h4>${insights.decision_type}</h4>
    <p class="callout-lead">${insights.decision_hint}</p>
    <p class="support-copy">${insights.risk_reason}</p>
    <div class="plain-meta">${insights.risk_level} · ${current.phase_label || current.phase_text || current.phase} · ${
      current.scenario.focus
    }</div>
  `;
}

function renderResources() {
  const current = appState.currentState;
  if (!current) return;
  const grid = qs("#resource-grid");
  grid.innerHTML = "";
  const resources = current.resources;
  [
    ["当前盲注", resources.blind_label || resources.blind_text || resources.blind, "这回合门槛"],
    [
      "当前筹码",
      `${formatNumber(resources.score_chips, 0)} / ${formatNumber(resources.target_chips, 0)}`,
      "离目标还有多远",
    ],
    ["距离目标", `${formatNumber(currentInsights().score_gap, 0)} 筹码`, currentInsights().score_gap <= 0 ? "已经够分" : "还要再补一段"],
    [
      "剩余资源",
      `${resources.hands_left} 出牌 / ${resources.discards_left} 弃牌`,
      resources.hands_left <= 1 || resources.discards_left <= 1 ? "已经偏紧" : "还有操作空间",
    ],
  ].forEach(([label, value, subvalue]) =>
    grid.appendChild(metricCard(label, value, subvalue, resourceIconName(label)))
  );

  const zoneCounts = qs("#zone-counts");
  if (!zoneCounts) return;
  zoneCounts.innerHTML = "";
  [
    ["牌堆", current.zones.deck_count, "还能抽多少牌"],
    ["弃牌堆", current.zones.discard.length, "已经送走多少牌"],
    ["已打出", current.zones.played.length, "本回合已经用掉多少牌"],
    ["Joker", current.jokers.length, "当前生效的机制数量"],
  ].forEach(([label, value, subvalue]) =>
    zoneCounts.appendChild(metricCard(label, `${value}`, subvalue, resourceIconName(label)))
  );
}

function renderHand() {
  const current = appState.currentState;
  if (!current) return;
  const selectedIndices = new Set((getSelectedRecommendation()?.action?.indices || []).map((value) => Number(value)));
  const hasSelection = selectedIndices.size > 0;
  qs("#hand-meta").textContent = `${current.zones.hand.length} 张可见手牌`;
  const container = qs("#hand-cards");
  container.innerHTML = "";
  current.zones.hand.forEach((card) => {
    const chosen = selectedIndices.has(Number(card.index));
    const node = document.createElement("div");
    node.className = `playing-card${chosen ? " selected" : ""}${hasSelection && !chosen ? " dimmed" : ""}`;
    node.innerHTML = `
      <span class="card-index">#${Number(card.index) + 1}</span>
      <div class="card-rank">${card.rank}</div>
      <div class="card-suit">${card.suit_text}</div>
      <div class="card-note">${card.status_text || "标准牌"}</div>
    `;
    container.appendChild(node);
  });

  const jokerList = qs("#joker-list");
  jokerList.innerHTML = "";
  if (!current.jokers.length) {
    jokerList.innerHTML = emptyState("当前没有 Joker", "这个场景更适合单纯讲出牌或弃牌判断。");
    return;
  }
  current.jokers.forEach((joker) => {
    const chip = document.createElement("div");
    chip.className = "note-card";
    chip.innerHTML = `
      <strong>${joker.label_zh || joker.label}</strong>
      <p>${joker.effect_text || "这个 Joker 会改变当前动作的价值计算。推荐会把这部分增益算进去。"}</p>
    `;
    jokerList.appendChild(chip);
  });
}

function recommendationSummaryLine(recommendation) {
  const score = expectedScore(recommendation);
  const actionType = String(recommendation?.action?.action_type || "").toUpperCase();
  if (actionType === "DISCARD") {
    return "先换后手，争取更好的下一手。";
  }
  return `预计单步带来 ${formatNumber(score, 0)} 筹码。`;
}

function recommendationCard(recommendation, source, index, compareState) {
  const isSelected =
    appState.selectedRecommendation.source === source && appState.selectedRecommendation.index === index;
  const differenceClass = compareState?.same ? "compare-same" : "compare-different";
  const rankLabel = recommendation.rank === 1 ? "首选动作" : `备选 ${recommendation.rank}`;
  const differenceText = compareState?.same ? "与另一侧首选一致" : "与另一侧首选不同";
  const compactClass = recommendation.rank > 1 ? " compact-alt" : "";
  const postureText =
    recommendation.risk_level && recommendation.risk_level.includes("高")
      ? "更激进"
      : recommendation.risk_level && recommendation.risk_level.includes("低")
        ? "更稳"
        : "中性";
  if (recommendation.rank > 1) {
    return `
      <div class="recommendation-card ${differenceClass} compact-alt${isSelected ? " active" : ""}" data-source="${source}" data-index="${index}">
        <div class="recommendation-header">
          <div>
            <span class="section-label">${rankLabel}</span>
            <h4>${recommendation.label}</h4>
          </div>
          <span class="chip">${formatNumber(expectedScore(recommendation), 0)} 筹码</span>
        </div>
        <p class="summary-sentence">${differenceText} · ${riskText(recommendation.risk_level)} · ${postureText}</p>
      </div>
    `;
  }
  return `
    <div class="recommendation-card ${differenceClass}${compactClass}${recommendation.rank === 1 ? " top-pick" : ""}${isSelected ? " active" : ""}" data-source="${source}" data-index="${index}">
      <div class="recommendation-header">
        <div>
          <span class="section-label">${rankLabel}</span>
          <h4>${recommendation.label}</h4>
        </div>
        <span class="chip">${differenceText}</span>
      </div>
      <p class="callout-lead">${recommendationSummaryLine(recommendation)}</p>
      <div class="recommendation-meta">
        <span class="recommendation-chip">${recommendation.source_text || recommendation.source_label || source}</span>
        <span class="recommendation-chip ${riskClass(recommendation.risk_level)}">${riskText(recommendation.risk_level)}</span>
        <span class="recommendation-chip">置信度 ${formatPct(recommendation.confidence)}</span>
        <span class="recommendation-chip">预计 ${formatNumber(expectedScore(recommendation), 0)} 筹码</span>
      </div>
      <p class="callout-copy">${trimText(normalizeExplanation(recommendation.reason), 92)}</p>
      <p class="support-copy">风格：${postureText} · ${recommendation.teacher_agrees_label}</p>
      <p class="decision-footnote">不选下一名：${trimText(contrastReason(recommendation), 68)}</p>
    </div>
  `;
}

function setSelectedRecommendation(source, index, options = {}) {
  const recommendation = (appState.compareRecommendations[source] || [])[index];
  if (!recommendation) return;
  appState.selectedRecommendation = {
    source,
    index,
    signature: actionSignature(recommendation.action),
  };
  setFocusPanel("preview");
  syncInspectionEntry();
  renderTopSummary();
  renderStageSpotlight();
  renderHand();
  renderRecommendations();
  renderSelectedActionCard();
  renderPreview();
  renderTimeline();
  if (options.announce) {
    setStatusMessage(`正在查看：${recommendation.label}`);
    pulsePanels(["board", "decision", "preview", "timeline"]);
  }
}

function renderCompareSummary() {
  const root = qs("#compare-summary");
  const summary = compareSummaryData();
  if (!summary) {
    root.innerHTML = emptyState("推荐还没准备好", "场景载入后，这里会先给出模型 vs 基线的一句话结论。");
    return;
  }
  root.innerHTML = `
    <span class="section-label">模型 vs 启发式</span>
    <div class="callout-title">${summary.title}</div>
    <p class="callout-lead">${summary.lead}</p>
    <div class="compare-highlight">
      <div class="fact-item">
        <strong>首选是否一致</strong>
        <span>${summary.sameText}</span>
      </div>
      <div class="fact-item">
        <strong>分歧方向</strong>
        <span>${summary.postureText} · ${summary.scoreDiffText}</span>
      </div>
    </div>
    <p class="support-copy">${trimText(summary.copy, 106)}</p>
  `;
}

function renderSelectedActionCard() {
  const root = qs("#selected-action-card");
  const recommendation = getSelectedRecommendation();
  if (!recommendation) {
    root.innerHTML = emptyState("先选中一条动作建议", "右侧点任意推荐卡，下面会立刻切到这一步的完整解释。");
    return;
  }
  root.innerHTML = `
    <span class="section-label">当前聚焦动作</span>
    <div class="callout-title">${recommendation.label}</div>
    <p class="callout-lead">${recommendationSummaryLine(recommendation)}</p>
    <div class="selected-summary">
      <div class="fact-item">
        <strong>建议来源</strong>
        <span>${recommendation.source_text || recommendation.source_label || recommendation.source}</span>
      </div>
      <div class="fact-item">
        <strong>预计收益</strong>
        <span>${formatNumber(expectedScore(recommendation), 0)} 筹码</span>
      </div>
      <div class="fact-item">
        <strong>风险判断</strong>
        <span>${riskText(recommendation.risk_level)}</span>
      </div>
      <div class="fact-item">
        <strong>执行后阶段</strong>
        <span>${recommendationPhaseAfter(recommendation)}</span>
      </div>
    </div>
    <p class="support-copy">${trimText(normalizeExplanation(recommendation.reason), 108)}</p>
    <p class="decision-footnote">点击这条动作后，中间牌面、下方结果预览和时间线会一起更新。</p>
  `;
}

function renderRecommendations() {
  const compareState = compareSummaryData();
  ["model", "heuristic"].forEach((source) => {
    const container = qs(`#recommendations-${source}`);
    const list = appState.compareRecommendations[source] || [];
    if (!list.length) {
      container.innerHTML = emptyState("当前阶段没有候选动作", "切换场景或执行一步后再试。");
      return;
    }
    container.innerHTML = list
      .map((entry, index) => recommendationCard(entry, source, index, compareState))
      .join("");
    container.querySelectorAll(".recommendation-card").forEach((node) => {
      node.addEventListener("click", () => {
        setSelectedRecommendation(node.dataset.source, Number(node.dataset.index), { announce: true });
      });
    });
  });
  renderCompareSummary();
}

function renderPreview() {
  const focusRoot = qs("#preview-focus");
  const summaryRoot = qs("#preview-summary");
  const deltaRoot = qs("#preview-delta");
  focusRoot.innerHTML = "";
  summaryRoot.innerHTML = "";
  deltaRoot.innerHTML = "";

  const recommendation = getSelectedRecommendation();
  if (!recommendation) {
    const empty = emptyState("先点一条建议", "结果区会把当前动作执行后的关键变化讲清楚。");
    focusRoot.innerHTML = empty;
    summaryRoot.innerHTML = empty;
    return;
  }

  const preview = recommendation.preview || {};
  const delta = preview.delta || {};
  const headline = previewHeadline(recommendation);

  focusRoot.innerHTML = `
    <span class="section-label">执行后最重要的变化</span>
    <div class="callout-title">${headline.title}</div>
    <p class="callout-lead">${headline.lead}</p>
    <p class="decision-footnote">${headline.copy}</p>
  `;

  summaryRoot.innerHTML = `
    <div class="plain-meta">${recommendation.source_text || recommendation.source_label || recommendation.source} · ${riskText(
      recommendation.risk_level
    )} · ${recommendationPhaseAfter(recommendation)}</div>
    <div class="preview-inline-grid">
      <div class="preview-metric">
        <strong>预计收益</strong>
        <span>${formatNumber(expectedScore(recommendation), 0)} 筹码</span>
      </div>
      <div class="preview-metric">
        <strong>回合筹码</strong>
        <span>${formatNumber(delta.round_chips || 0, 0)}</span>
      </div>
      <div class="preview-metric">
        <strong>出牌变化</strong>
        <span>${formatNumber(delta.hands_left || 0, 0)}</span>
      </div>
      <div class="preview-metric">
        <strong>弃牌变化</strong>
        <span>${formatNumber(delta.discards_left || 0, 0)}</span>
      </div>
    </div>
  `;

  const deltaRows = [
    ["回合筹码", formatNumber(delta.round_chips || 0, 0), "执行这一手后，当前回合会加多少"],
    ["总筹码", formatNumber(delta.score_chips || 0, 0), "整局筹码变化"],
    ["出牌次数", formatNumber(delta.hands_left || 0, 0), "负数表示会消耗出牌次数"],
    ["弃牌次数", formatNumber(delta.discards_left || 0, 0), "负数表示会消耗弃牌次数"],
    ["现金变化", formatNumber(delta.money || 0, 0), "经济层面的变化"],
    ["牌堆变化", formatNumber(delta.deck_count || 0, 0), "下一轮还能摸到多少牌"],
  ];
  const visibleRows =
    appState.ui.focusPanel === "preview" ? deltaRows : deltaRows.slice(0, 4);
  visibleRows.forEach(([label, value, subvalue]) => {
    deltaRoot.appendChild(metricCard(label, `${value}`, subvalue, resourceIconName(label)));
  });
}

function renderTimeline() {
  const container = qs("#timeline");
  const timeline = appState.currentState?.timeline || [];
  const entries = [];

  if (appState.ui.inspectionEntry) {
    entries.push(appState.ui.inspectionEntry);
  }
  if (trainingIsActive()) {
    entries.push({
      kind: "training",
      kind_label: "训练更新",
      label: readableStatusLabel(appState.trainingStatus?.status_label, appState.trainingStatus?.status),
      summary: appState.trainingStatus?.message || "后台训练正在刷新。",
      highlight: false,
    });
  }
  timeline
    .slice()
    .reverse()
    .forEach((entry) => entries.push(entry));

  if (!entries.length) {
    container.innerHTML = emptyState("时间线还是空的", "载入场景、切换观察动作或执行一步后，这里会自动出现记录。");
    return;
  }

  container.innerHTML = entries
    .slice(0, appState.ui.focusPanel === "timeline" ? 6 : 3)
    .map((entry, index) => {
      let kindClass = "action";
      if (entry.kind === "scenario_loaded") kindClass = "scene";
      else if (entry.kind === "inspect") kindClass = "inspect";
      else if (entry.kind === "training") kindClass = "training";
      const subtitle =
        entry.kind === "scenario_loaded"
          ? entry.summary || entry.focus || ""
          : entry.kind === "inspect" || entry.kind === "training"
            ? entry.summary || ""
            : `${entry.delta?.phase_before || entry.phase_before || "-"} → ${
                entry.delta?.phase_after || entry.phase_after || "-"
              } ｜ 收益 ${formatNumber(entry.reward || 0, 0)}`;
      return `
        <div class="timeline-entry${index === 0 ? " highlight" : ""}">
          <span class="timeline-kind ${kindClass}">${entry.kind_label || entry.kind}</span>
          <h4>${entry.label}</h4>
          <p>${trimText(subtitle, 88)}</p>
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
  const chartRoot = qs("#training-chart");
  if (!history.length) {
    chartRoot.innerHTML = emptyState("还没有训练曲线", "启动一次训练后，这里会实时展示 train / val loss 的变化。");
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
      <text x="${padding}" y="18" fill="currentColor" opacity="0.68" font-size="12">训练损失（黄） / 验证损失（绿）</text>
      <text x="${padding}" y="${height - 8}" fill="currentColor" opacity="0.65" font-size="12">开始</text>
      <text x="${width - 74}" y="${height - 8}" fill="currentColor" opacity="0.65" font-size="12">第 ${history.length} 轮</text>
      <text x="${width - 108}" y="${padding + 14}" fill="currentColor" opacity="0.55" font-size="11">损失越低越好</text>
    </svg>
  `;
}

function renderTrainingResultCard() {
  const root = qs("#training-result-card");
  const narrative = trainingNarrative(appState.modelInfo, appState.trainingStatus);
  root.innerHTML = `
    <span class="section-label">成果总结</span>
    <div class="callout-title">${narrative.title}</div>
    <p class="callout-lead">${narrative.lead}</p>
    <div class="fact-list">
      <div class="fact-item">
        <strong>当前模型</strong>
        <span>${appState.modelInfo?.model_name || "-"}</span>
      </div>
      <div class="fact-item">
        <strong>运行 ID</strong>
        <span>${currentRunId()}</span>
      </div>
      <div class="fact-item">
        <strong>样本规模</strong>
        <span>${appState.modelInfo?.dataset_stats?.total_records || "-"}</span>
      </div>
      <div class="fact-item">
        <strong>Top-3</strong>
        <span>${formatPct(appState.modelInfo?.metrics?.final?.val_acc3 || 0)}</span>
      </div>
    </div>
    <p class="support-copy">已经学会：${narrative.strengths}</p>
    <p class="decision-footnote">当前边界：${narrative.limitation}</p>
  `;
}

function renderTrainingStatusCard() {
  const training = appState.trainingStatus || {};
  const modelInfo = appState.modelInfo || {};
  const statusRoot = qs("#training-status-card");
  const progressValue = trainingProgressValue();
  const displayStatusLabel =
    training.status === "idle"
      ? statusText("idle")
      : readableStatusLabel(training.status_label, training.status || "idle");
  const displayMessage =
    training.status === "idle"
      ? "当前没有训练任务。你可以现场点一次快速烟雾训练，让面试官直接看到训练闭环。"
      : training.message || "训练状态已更新。";

  statusRoot.innerHTML = `
    <h4>${displayStatusLabel} / ${stageText(trainingStage())}</h4>
    <p>${displayMessage}</p>
    <div class="progress-shell" style="margin-top:12px;">
      <div class="progress-bar" style="width:${progressValue * 100}%"></div>
    </div>
    <div class="plain-meta" style="margin-top:12px;">
      任务 ${training.job_id || "-"} · 阶段 ${stageText(trainingStage())} · 剩余 ${formatDuration(
        training.training?.eta_sec || training.eta_sec
      )} · 最佳轮次 ${training.training?.best_epoch || modelInfo.metrics?.best_epoch || "-"} · 最佳损失 ${formatNumber(
        training.training?.best_val_loss || modelInfo.metrics?.best_val_loss || 0,
        3
      )}
    </div>
  `;
}

function renderModelMeta() {
  const modelMeta = qs("#model-meta");
  const modelInfo = appState.modelInfo || {};
  modelMeta.innerHTML = "";
  const rows = [
    ["当前运行 ID", currentRunId(), modelInfo.loaded ? "当前已经由页面加载" : "当前仍会回退到启发式"],
    ["模型状态", modelInfo.loaded ? "已加载" : "未加载", modelInfo.loaded ? "模型已成功接管推荐" : "建议先训练或加载 checkpoint"],
    ["样本规模", `${modelInfo.dataset_stats?.total_records || modelInfo.metrics?.train_samples || "-"}`, "训练这版模型用到的总样本数"],
    ["最佳验证损失", formatNumber(modelInfo.metrics?.best_val_loss || 0, 3), "越低越好"],
    ["Top-1 命中", formatPct(modelInfo.metrics?.final?.val_acc1 || 0), "第一推荐命中的比例"],
    ["Top-3 命中", formatPct(modelInfo.metrics?.final?.val_acc3 || 0), "前三候选里命中的比例"],
    ["训练设备", modelInfo.config?.device_used || "-", "最近一次训练跑在什么设备上"],
    ["当前 checkpoint", modelInfo.checkpoint_path ? "已保存" : "缺失", modelInfo.checkpoint_path || "还没有可用模型文件"],
  ];
  const visibleRows = appState.ui.focusPanel === "training" ? rows : rows.slice(0, 6);
  visibleRows.forEach(([label, value, subvalue]) =>
    modelMeta.appendChild(metricCard(label, value, subvalue, resourceIconName(label)))
  );
}

function renderTrainingNotes() {
  const root = qs("#training-side-notes");
  const modelInfo = appState.modelInfo || {};
  const evalRows = scenarioEvalRows();
  const narrative = trainingNarrative(modelInfo, appState.trainingStatus);
  const notes = [];

  notes.push(`
    <div class="note-card">
      <strong>这块为什么重要</strong>
      <p>这块用来证明页面里的建议不是写死规则，而是来自真实训练过的 checkpoint。</p>
    </div>
  `);
  notes.push(`
    <div class="note-card">
      <strong>它已经学会了什么</strong>
      <p>${narrative.strengths}</p>
    </div>
  `);
  notes.push(`
    <div class="note-card">
      <strong>现在的边界</strong>
      <p>${narrative.limitation}</p>
    </div>
  `);
  if (modelInfo.verdict?.improved_over_previous !== undefined) {
    notes.push(`
      <div class="note-card">
        <strong>和上一版的关系</strong>
        <p>${
          modelInfo.verdict?.improved_over_previous
            ? "这一版已经优于上一版，适合作为默认展示模型。"
            : "这一版没有全面超过上一版，更适合展示训练闭环和场景能力。"
        }</p>
      </div>
    `);
  }
  notes.push(
    ...evalRows.slice(0, appState.ui.focusPanel === "training" ? 3 : 2).map(
      (row) => `
        <div class="note-card">
          <strong>${row.scenario_name}</strong>
          <p>${row.top1_label || "暂无首选动作"}</p>
          <p class="muted">教师一致：${row.teacher_agrees ? "是" : "否"} ｜ 预计 ${formatNumber(row.expected_score, 0)}</p>
        </div>
      `
    )
  );
  root.innerHTML = notes.join("");
}

function renderTrainingPanel() {
  renderTrainingResultCard();
  renderTrainingStatusCard();
  renderModelMeta();
  renderTrainingChart(trainingHistoryRows());
  renderTrainingNotes();
}

function refreshButtonStates() {
  const selected = getSelectedRecommendation();
  const trainingRunning = trainingIsActive();
  qs("#step-button").disabled = appState.busy.action || !selected;
  qs("#autoplay-button").disabled = appState.busy.action;
  qs("#reload-button").disabled = appState.busy.action;
  qs("#train-standard-button").disabled = trainingRunning || appState.busy.training;
  qs("#train-smoke-button").disabled = trainingRunning || appState.busy.training;
}

function renderAll() {
  if (!appState.currentState) return;
  renderViewModeControls();
  renderTopSummary();
  renderStatusStrip();
  renderScenarios();
  renderInsightTags();
  renderStageSpotlight();
  renderStateSummary();
  renderResources();
  renderHand();
  renderRecommendations();
  renderSelectedActionCard();
  renderPreview();
  renderTimeline();
  renderTrainingPanel();
  refreshButtonStates();
  renderIconSlots();
  renderViewModeControls();
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
  setPanelLoading(["scenario", "board", "decision", "preview"], true);
  setStatusMessage("正在切换场景…");
  try {
    await apiFetch("/api/scenario/load", {
      method: "POST",
      body: JSON.stringify({ scenario_id: scenarioId }),
    });
    await refreshState();
    await refreshRecommendations();
    setFocusPanel("preview");
    syncInspectionEntry();
    renderAll();
    pulsePanels(["scenario", "board", "decision", "preview", "timeline"]);
    setStatusMessage(appState.currentState?.scenario?.talk_track || "");
    showToast(`已切换到场景：${appState.currentState?.scenario?.name}`, "success");
  } catch (error) {
    showToast(error.message, "error");
    throw error;
  } finally {
    appState.busy.action = false;
    setPanelLoading(["scenario", "board", "decision", "preview"], false);
    refreshButtonStates();
  }
}

async function executeSelected() {
  const recommendation = getSelectedRecommendation();
  if (!recommendation) return;
  appState.busy.action = true;
  refreshButtonStates();
  setPanelLoading(["board", "decision", "preview", "timeline"], true);
  setStatusMessage(`正在执行：${recommendation.label}`);
  try {
    await apiFetch("/api/step", {
      method: "POST",
      body: JSON.stringify({ policy: recommendation.source, action: recommendation.action }),
    });
    await refreshState();
    await refreshRecommendations();
    setFocusPanel("preview");
    syncInspectionEntry();
    renderAll();
    pulsePanels(["board", "decision", "preview", "timeline"]);
    showToast(`已执行：${recommendation.label}`, "success");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    appState.busy.action = false;
    setPanelLoading(["board", "decision", "preview", "timeline"], false);
    refreshButtonStates();
  }
}

async function autoplay() {
  appState.busy.action = true;
  refreshButtonStates();
  setPanelLoading(["board", "decision", "preview", "timeline"], true);
  setStatusMessage("正在自动演示 3 步…");
  try {
    await apiFetch("/api/autoplay", {
      method: "POST",
      body: JSON.stringify({ policy: "model", steps: 3 }),
    });
    await refreshState();
    await refreshRecommendations();
    setFocusPanel("preview");
    syncInspectionEntry();
    renderAll();
    pulsePanels(["board", "decision", "preview", "timeline"]);
    showToast("自动演示完成。", "success");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    appState.busy.action = false;
    setPanelLoading(["board", "decision", "preview", "timeline"], false);
    refreshButtonStates();
  }
}

async function startTraining(profile) {
  appState.busy.training = true;
  refreshButtonStates();
  setPanelLoading(["training"], true);
  setStatusMessage(profile === "smoke" ? "正在启动快速烟雾训练…" : "正在启动 2 小时训练…");
  try {
    const payload = await apiFetch("/api/training/start", {
      method: "POST",
      body: JSON.stringify({ profile }),
    });
    appState.trainingStatus = payload;
    renderTrainingPanel();
    refreshButtonStates();
    pulsePanels(["training", "timeline"]);
    showToast(profile === "smoke" ? "快速烟雾训练已启动。" : "2 小时训练已在后台启动。", "info");
  } catch (error) {
    showToast(error.message, "error");
  } finally {
    appState.busy.training = false;
    setPanelLoading(["training"], false);
    refreshButtonStates();
  }
}

function bindControls() {
  qs("#reload-button").addEventListener("click", async () => {
    if (appState.currentScenarioId) {
      await loadScenario(appState.currentScenarioId);
    }
  });
  document.querySelectorAll(".dock-tab[data-focus-panel]").forEach((button) => {
    button.addEventListener("click", () => {
      setFocusPanel(button.dataset.focusPanel);
      pulsePanels([button.dataset.focusPanel]);
    });
  });
  qs("#step-button").addEventListener("click", executeSelected);
  qs("#autoplay-button").addEventListener("click", autoplay);
  qs("#train-standard-button").addEventListener("click", async () => {
    await startTraining("standard");
  });
  qs("#train-smoke-button").addEventListener("click", async () => {
    await startTraining("smoke");
  });
}

async function refreshAll(initial = false) {
  if (initial) {
    const scenariosPayload = await apiFetch("/api/scenarios");
    appState.scenarios = scenariosPayload.scenarios || [];
  }
  const previousTrainingStatus = trainingStage();
  await Promise.all([refreshModelInfo(), refreshState(), refreshTrainingStatus()]);
  await refreshRecommendations();
  renderViewModeControls();
  syncInspectionEntry();
  renderAll();
  setStatusMessage(appState.currentState?.scenario?.talk_track || "");

  if (trainingIsActive(previousTrainingStatus) && appState.trainingStatus?.status === "finished") {
    await refreshModelInfo();
    await refreshRecommendations();
    syncInspectionEntry();
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
      if (trainingIsActive() || previousStatus !== appState.trainingStatus?.status) {
        renderTrainingPanel();
        renderTimeline();
        refreshButtonStates();
      }
      if (appState.pollCounter % 3 === 0) {
        await refreshModelInfo();
        renderTrainingPanel();
      }
      if (trainingIsActive(previousStatus) && appState.trainingStatus?.status === "finished") {
        await refreshModelInfo();
        await refreshRecommendations();
        syncInspectionEntry();
        renderAll();
        pulsePanels(["training", "decision", "preview"]);
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
  setStatusMessage("正在载入演示…");
  setPanelLoading(["hero", "scenario", "board", "decision", "preview", "training"], true);
  try {
    await refreshAll(true);
    startPolling();
  } finally {
    setPanelLoading(["hero", "scenario", "board", "decision", "preview", "training"], false);
  }
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
