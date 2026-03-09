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

function metricCard(label, value, subvalue = "") {
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
  qs("#status-message").textContent = text || "";
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
  qs("#status-mode").textContent = current.mode === "autoplay" ? "自动演示" : "手动";
  qs("#status-phase").textContent = current.phase_label || current.phase_text || current.phase;
  qs("#scenario-focus-inline").textContent = current.scenario.focus || "-";
  qs("#hero-description").textContent =
    current.scenario.talk_track || "左边选场景，中间看局面，右边看推荐分歧，再执行一步。";

  const selected = getSelectedRecommendation();
  const focusRoot = qs("#hero-focus-card");
  if (!selected) {
    focusRoot.innerHTML = emptyState("正在准备推荐动作", "场景载入后，这里会先给出一句最值得讲的结论。");
    return;
  }

  focusRoot.innerHTML = `
    <span class="section-label">当前一句话</span>
    <div class="callout-title">${topPolicyLine(selected)}</div>
    <p class="callout-copy">${normalizeExplanation(selected.reason)}</p>
    <div class="flow-kpis" style="margin-top:14px;">
      <span class="kpi-chip">${selected.source_text || selected.source_label || selected.source}</span>
      <span class="kpi-chip">置信度 ${formatPct(selected.confidence)}</span>
      <span class="tone-chip ${riskClass(selected.risk_level)}">${selected.risk_level}风险</span>
      <span class="kpi-chip">预计 ${formatNumber(expectedScore(selected), 0)} 筹码</span>
    </div>
  `;
}

function renderStatusStrip() {
  const current = appState.currentState;
  if (!current) return;
  const insights = currentInsights();
  qs("#status-risk").textContent = insights.risk_level;
  qs("#status-decision-type").textContent = insights.decision_type;
  qs("#status-gap").textContent = `${formatNumber(insights.score_gap, 0)} 筹码`;
  qs("#status-training").textContent =
    readableStatusLabel(appState.trainingStatus?.status_label, appState.trainingStatus?.status);
}

function renderScenarios() {
  const container = qs("#scenario-list");
  container.innerHTML = "";
  appState.scenarios.forEach((scenario) => {
    const active = scenario.id === appState.currentScenarioId ? " active" : "";
    const tags = (scenario.tags || []).map((tag) => `<span class="chip">${tag}</span>`).join("");
    const node = document.createElement("div");
    node.className = `scenario-card${active}`;
    node.innerHTML = `
      <h4>${scenario.name}</h4>
      <p class="callout-lead">这场主要看：${scenario.focus}</p>
      <p>${scenario.summary}</p>
      <div class="scenario-meta">
        <span>手数 ${scenario.hands_left} / 弃牌 ${scenario.discards_left}</span>
        <span>目标 ${formatNumber(scenario.target_chips, 0)} 筹码</span>
      </div>
      <div class="tag-row">${tags}</div>
      <p class="muted">${scenario.talk_track || ""}</p>
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
  const tags = [
    ...(insights.tags || []),
    current?.scenario?.focus ? `主看：${current.scenario.focus}` : "",
  ].filter(Boolean);
  root.innerHTML = tags.map((tag) => `<span class="chip">${tag}</span>`).join("");
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
    <span class="section-label">这一步最值得盯住什么</span>
    <div class="callout-title">${preview.title}</div>
    <p class="callout-lead">${preview.lead}</p>
    <p class="callout-copy">${contrastReason(selected) || selected.risk_hint || normalizeExplanation(preview.copy)}</p>
    <div class="flow-kpis" style="margin-top:14px;">
      <span class="kpi-chip">动作：${selected.label}</span>
      <span class="kpi-chip">后续阶段：${recommendationPhaseAfter(selected)}</span>
      <span class="kpi-chip">${selected.teacher_agrees_label}</span>
    </div>
  `;
}

function renderStateSummary() {
  const current = appState.currentState;
  const root = qs("#state-summary-card");
  if (!current) return;
  const insights = currentInsights();
  root.innerHTML = `
    <h4>${insights.decision_type}</h4>
    <p class="callout-lead">${insights.decision_hint}</p>
    <p class="callout-copy">${insights.risk_reason}</p>
  `;
}

function renderResources() {
  const current = appState.currentState;
  if (!current) return;
  const grid = qs("#resource-grid");
  grid.innerHTML = "";
  const resources = current.resources;
  [
    ["当前盲注", resources.blind_label || resources.blind_text || resources.blind, "这是本回合要跨过去的门槛"],
    [
      "当前筹码",
      `${formatNumber(resources.score_chips, 0)} / ${formatNumber(resources.target_chips, 0)}`,
      "越接近目标，越能直接结束这一手的讲解",
    ],
    ["剩余出牌", `${resources.hands_left}`, resources.hands_left <= 1 ? "已经偏紧，要慎重" : "还有试错空间"],
    ["剩余弃牌", `${resources.discards_left}`, resources.discards_left <= 1 ? "弃牌机会很贵" : "还能继续优化手牌"],
    ["现金", `$${formatNumber(resources.money, 0)}`, resources.money <= 2 ? "经济偏紧" : "后续还有调整余地"],
    ["当前轮次", `Ante ${resources.ante} / Round ${resources.round_num}`, "方便解释这是哪一个压力阶段"],
    ["重掷成本", `$${formatNumber(resources.reroll_cost, 0)}`, "主要和商店决策相关"],
    ["上一手牌型", current.score.last_hand_type || "无", "帮助解释当前局面的节奏来源"],
  ].forEach(([label, value, subvalue]) => grid.appendChild(metricCard(label, value, subvalue)));

  const zoneCounts = qs("#zone-counts");
  zoneCounts.innerHTML = "";
  [
    ["牌堆", current.zones.deck_count, "还能抽多少牌"],
    ["弃牌堆", current.zones.discard.length, "已经送走多少牌"],
    ["已打出", current.zones.played.length, "本回合已经用掉多少牌"],
    ["Joker", current.jokers.length, "当前生效的机制数量"],
  ].forEach(([label, value, subvalue]) => zoneCounts.appendChild(metricCard(label, `${value}`, subvalue)));
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
  return `
    <div class="recommendation-card ${differenceClass}${isSelected ? " active" : ""}" data-source="${source}" data-index="${index}">
      <span class="section-label">#${recommendation.rank} ${recommendation.rank === 1 ? "首选" : "候选"}</span>
      <h4>${recommendation.label}</h4>
      <p class="callout-lead">${recommendationSummaryLine(recommendation)}</p>
      <div class="recommendation-meta">
        <span class="recommendation-chip">${recommendation.source_text || recommendation.source_label || source}</span>
        <span class="recommendation-chip ${riskClass(recommendation.risk_level)}">${recommendation.risk_level}风险</span>
        <span class="recommendation-chip">置信度 ${formatPct(recommendation.confidence)}</span>
        <span class="recommendation-chip">预计 ${formatNumber(expectedScore(recommendation), 0)} 筹码</span>
        <span class="recommendation-chip">${recommendation.teacher_agrees_label}</span>
      </div>
      <p>${normalizeExplanation(recommendation.reason)}</p>
      <div class="recommendation-tags" style="margin-top:10px;">
        ${(recommendation.tags || []).map((tag) => `<span class="chip">${tag}</span>`).join("")}
      </div>
      <p class="muted" style="margin-top:10px;">没选下一名：${contrastReason(recommendation)}</p>
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
    <span class="section-label">模型 vs 基线首选</span>
    <div class="callout-title">${summary.title}</div>
    <p class="callout-lead">${summary.lead}</p>
    <div class="duo-metrics">
      <div class="duo-metric">
        <span class="section-label">是否一致</span>
        <strong>${summary.sameText}</strong>
        <p class="muted">一眼看模型和基线是不是同路。</p>
      </div>
      <div class="duo-metric">
        <span class="section-label">分歧方向</span>
        <strong>${summary.postureText}</strong>
        <p class="muted">${summary.scoreDiffText}</p>
      </div>
    </div>
    <p class="callout-copy">${summary.copy}</p>
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
    <div class="action-summary-grid">
      <span class="kpi-chip">${recommendation.source_text || recommendation.source_label || recommendation.source}</span>
      <span class="kpi-chip">后续阶段：${recommendationPhaseAfter(recommendation)}</span>
      <span class="tone-chip ${riskClass(recommendation.risk_level)}">${recommendation.risk_level}风险</span>
      <span class="kpi-chip">置信度 ${formatPct(recommendation.confidence)}</span>
      <span class="kpi-chip">预计 ${formatNumber(expectedScore(recommendation), 0)} 筹码</span>
    </div>
    <p class="callout-copy">${normalizeExplanation(recommendation.reason)}</p>
    <p class="help-text">这张卡是整页的主卖点。点“执行当前动作”后，下方结果区和时间线会同步变化。</p>
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
    <p class="callout-copy">${headline.copy}</p>
  `;

  summaryRoot.innerHTML = `
    <h4>${recommendation.label}</h4>
    <p>${normalizeExplanation(recommendation.reason)}</p>
    <div class="pill-row" style="margin-top:12px;">
      <span class="chip">${recommendation.source_text || recommendation.source_label || recommendation.source}</span>
      <span class="chip">执行后阶段：${recommendationPhaseAfter(recommendation)}</span>
      <span class="chip">预计牌型：${preview.expected_hand_type || "-"}</span>
      <span class="chip">预计收益：${formatNumber(expectedScore(recommendation), 0)}</span>
    </div>
    <div class="preview-highlight">
      <span class="button-icon">${icon("trend")}</span>
      ${contrastReason(recommendation)}
    </div>
  `;

  [
    ["回合筹码", formatNumber(delta.round_chips || 0, 0), "执行这一手后，当前回合会加多少"],
    ["总筹码", formatNumber(delta.score_chips || 0, 0), "整局筹码变化"],
    ["出牌次数", formatNumber(delta.hands_left || 0, 0), "负数表示会消耗出牌次数"],
    ["弃牌次数", formatNumber(delta.discards_left || 0, 0), "负数表示会消耗弃牌次数"],
    ["现金变化", formatNumber(delta.money || 0, 0), "经济层面的变化"],
    ["牌堆变化", formatNumber(delta.deck_count || 0, 0), "下一轮还能摸到多少牌"],
    ["弃牌堆变化", formatNumber(delta.discard_count || 0, 0), "会有多少牌进入弃牌堆"],
    ["已打出变化", formatNumber(delta.played_count || 0, 0), "会有多少牌被真正打出去"],
  ].forEach(([label, value, subvalue]) => {
    deltaRoot.appendChild(metricCard(label, `${value}`, subvalue));
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
    .slice(0, 7)
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
      <text x="${padding}" y="18" fill="currentColor" opacity="0.65" font-size="12">训练损失（黄） / 验证损失（绿）</text>
      <text x="${padding}" y="${height - 8}" fill="currentColor" opacity="0.65" font-size="12">第 1 轮</text>
      <text x="${width - 104}" y="${height - 8}" fill="currentColor" opacity="0.65" font-size="12">第 ${history.length} 轮</text>
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
    <div class="training-kpis">
      <span class="kpi-chip">运行 ID：${currentRunId()}</span>
      <span class="kpi-chip">样本数 ${appState.modelInfo?.dataset_stats?.total_records || "-"}</span>
      <span class="kpi-chip">最佳损失 ${formatNumber(appState.modelInfo?.metrics?.best_val_loss || 0, 3)}</span>
      <span class="kpi-chip">Top-3 ${formatPct(appState.modelInfo?.metrics?.final?.val_acc3 || 0)}</span>
    </div>
    <p class="callout-copy">已经学会：${narrative.strengths}</p>
    <p class="help-text">当前边界：${narrative.limitation}</p>
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
    <div class="pill-row" style="margin-top:12px;">
      <span class="chip">任务 ID：${training.job_id || "-"}</span>
      <span class="chip">阶段：${stageText(trainingStage())}</span>
      <span class="chip">预计剩余：${formatDuration(training.training?.eta_sec || training.eta_sec)}</span>
      <span class="chip">最佳轮次：${training.training?.best_epoch || modelInfo.metrics?.best_epoch || "-"}</span>
      <span class="chip">最佳损失：${formatNumber(training.training?.best_val_loss || modelInfo.metrics?.best_val_loss || 0, 3)}</span>
    </div>
  `;
}

function renderModelMeta() {
  const modelMeta = qs("#model-meta");
  const modelInfo = appState.modelInfo || {};
  modelMeta.innerHTML = "";
  [
    ["当前运行 ID", currentRunId(), modelInfo.loaded ? "当前已经由页面加载" : "当前仍会回退到启发式"],
    ["模型状态", modelInfo.loaded ? "已加载" : "未加载", modelInfo.loaded ? "模型已成功接管推荐" : "建议先训练或加载 checkpoint"],
    ["样本规模", `${modelInfo.dataset_stats?.total_records || modelInfo.metrics?.train_samples || "-"}`, "训练这版模型用到的总样本数"],
    ["最佳验证损失", formatNumber(modelInfo.metrics?.best_val_loss || 0, 3), "越低越好"],
    ["Top-1 命中", formatPct(modelInfo.metrics?.final?.val_acc1 || 0), "第一推荐命中的比例"],
    ["Top-3 命中", formatPct(modelInfo.metrics?.final?.val_acc3 || 0), "前三候选里命中的比例"],
    ["训练设备", modelInfo.config?.device_used || "-", "最近一次训练跑在什么设备上"],
    ["当前 checkpoint", modelInfo.checkpoint_path ? "已保存" : "缺失", modelInfo.checkpoint_path || "还没有可用模型文件"],
  ].forEach(([label, value, subvalue]) => modelMeta.appendChild(metricCard(label, value, subvalue)));
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
      <p>训练面板不是监控台，而是用来证明“这页里的模型建议真的是训练出来的”。</p>
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
    ...evalRows.slice(0, 3).map(
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
