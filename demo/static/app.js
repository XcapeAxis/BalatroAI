const state = {
  scenarios: [],
  currentScenarioId: null,
  currentPolicy: "model",
  currentState: null,
  recommendations: [],
  selectedRecommendationIndex: 0,
  modelInfo: null,
};

async function apiFetch(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${path}`);
  }
  return payload;
}

function metricCard(label, value) {
  const template = document.querySelector("#metric-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".metric-label").textContent = label;
  node.querySelector(".metric-value").textContent = value;
  return node;
}

function formatNumber(value) {
  const numeric = Number(value || 0);
  return Number.isInteger(numeric) ? `${numeric}` : numeric.toFixed(1);
}

function renderScenarios() {
  const container = document.querySelector("#scenario-list");
  container.innerHTML = "";
  state.scenarios.forEach((scenario) => {
    const card = document.createElement("div");
    card.className = "scenario-card";
    if (scenario.id === state.currentScenarioId) {
      card.classList.add("active");
    }
    card.innerHTML = `
      <h3>${scenario.name}</h3>
      <p>${scenario.summary}</p>
      <div class="pill-line">
        <span class="pill">${scenario.focus}</span>
        <span class="pill">${scenario.phase}</span>
      </div>
      <button class="secondary">Load Scenario</button>
    `;
    card.querySelector("button").addEventListener("click", async () => {
      await loadScenario(scenario.id);
    });
    container.appendChild(card);
  });
}

function renderStatus() {
  const current = state.currentState;
  if (!current) return;
  document.querySelector("#status-scenario").textContent = current.scenario.name;
  document.querySelector("#status-model").textContent = current.model_name || "-";
  document.querySelector("#status-phase").textContent = current.phase;
  document.querySelector("#status-mode").textContent = current.mode;
}

function renderResources() {
  const current = state.currentState;
  if (!current) return;
  const grid = document.querySelector("#resource-grid");
  grid.innerHTML = "";
  const resources = current.resources;
  [
    ["Blind", resources.blind],
    ["Chips", `${formatNumber(resources.score_chips)} / ${formatNumber(resources.target_chips)}`],
    ["Hands Left", resources.hands_left],
    ["Discards Left", resources.discards_left],
    ["Money", `$${formatNumber(resources.money)}`],
    ["Round", `${resources.ante}-${resources.round_num}`],
    ["Reroll Cost", `$${formatNumber(resources.reroll_cost)}`],
    ["Last Hand", current.score.last_hand_type || "none"],
  ].forEach(([label, value]) => grid.appendChild(metricCard(label, value)));

  const zoneCounts = document.querySelector("#zone-counts");
  zoneCounts.innerHTML = "";
  [
    ["Deck", current.zones.deck_count],
    ["Discard", current.zones.discard.length],
    ["Played", current.zones.played.length],
    ["Jokers", current.jokers.length],
  ].forEach(([label, value]) => zoneCounts.appendChild(metricCard(label, value)));
}

function renderHand() {
  const current = state.currentState;
  if (!current) return;
  document.querySelector("#state-summary").textContent = current.scenario.talk_track || current.scenario.summary;
  document.querySelector("#hand-meta").textContent = `${current.zones.hand.length} visible cards`;
  const container = document.querySelector("#hand-cards");
  container.innerHTML = "";
  const selectedIndices = new Set(
    (state.recommendations[state.selectedRecommendationIndex]?.action?.indices || []).map((value) => Number(value))
  );
  current.zones.hand.forEach((card) => {
    const node = document.createElement("div");
    node.className = "playing-card";
    if (selectedIndices.has(Number(card.index))) {
      node.classList.add("selected");
    }
    node.innerHTML = `
      <span class="index">#${Number(card.index) + 1}</span>
      <div class="rank">${card.rank}</div>
      <div class="suit">${card.suit}</div>
      <div class="zone-meta">${card.effect_text || (card.modifier_tags[0] || "standard")}</div>
    `;
    container.appendChild(node);
  });

  const jokerList = document.querySelector("#joker-list");
  jokerList.innerHTML = "";
  if (!current.jokers.length) {
    jokerList.innerHTML = `<div class="empty">No jokers in this scenario.</div>`;
  } else {
    current.jokers.forEach((joker) => {
      const tag = document.createElement("div");
      tag.className = "tag";
      tag.textContent = joker.label;
      jokerList.appendChild(tag);
    });
  }
}

function renderRecommendations() {
  const container = document.querySelector("#recommendations");
  container.innerHTML = "";
  if (!state.recommendations.length) {
    container.innerHTML = `<div class="empty">No recommendations available in the current phase.</div>`;
    return;
  }
  state.recommendations.forEach((recommendation, index) => {
    const card = document.createElement("div");
    card.className = "recommendation-card";
    if (index === state.selectedRecommendationIndex) {
      card.classList.add("active");
    }
    card.innerHTML = `
      <h3>#${recommendation.rank} ${recommendation.label}</h3>
      <div class="recommendation-meta">
        <span>source: ${recommendation.source}</span>
        <span>score: ${formatNumber(recommendation.score)}</span>
        <span>confidence: ${formatNumber(recommendation.confidence)}</span>
      </div>
      <p>${recommendation.reason}</p>
      <button class="secondary">Select</button>
    `;
    card.querySelector("button").addEventListener("click", () => {
      state.selectedRecommendationIndex = index;
      renderRecommendations();
      renderHand();
      renderPreview();
    });
    container.appendChild(card);
  });
}

function renderPreview() {
  const previewRoot = document.querySelector("#preview-summary");
  const deltaRoot = document.querySelector("#preview-delta");
  previewRoot.innerHTML = "";
  deltaRoot.innerHTML = "";
  const recommendation = state.recommendations[state.selectedRecommendationIndex];
  if (!recommendation) {
    previewRoot.innerHTML = `<div class="empty">Pick a recommendation to inspect the next-step delta.</div>`;
    return;
  }
  const preview = recommendation.preview || {};
  previewRoot.innerHTML = `
    <strong>${recommendation.label}</strong>
    <p>${recommendation.reason}</p>
    <div class="pill-line">
      <span class="pill">phase after: ${preview.phase_after || "-"}</span>
      <span class="pill">reward: ${formatNumber(preview.reward || 0)}</span>
      <span class="pill">expected score: ${preview.expected_score ? formatNumber(preview.expected_score) : "-"}</span>
      <span class="pill">hand type: ${preview.expected_hand_type || "-"}</span>
    </div>
  `;
  const delta = preview.delta || {};
  [
    ["Hands", delta.hands_left || 0],
    ["Discards", delta.discards_left || 0],
    ["Round Chips", delta.round_chips || 0],
    ["Money", delta.money || 0],
    ["Deck", delta.deck_count || 0],
    ["Discard", delta.discard_count || 0],
    ["Played", delta.played_count || 0],
    ["Jokers", delta.joker_count || 0],
  ].forEach(([label, value]) => deltaRoot.appendChild(metricCard(label, formatNumber(value))));
}

function renderTimeline() {
  const container = document.querySelector("#timeline");
  container.innerHTML = "";
  const timeline = state.currentState?.timeline || [];
  if (!timeline.length) {
    container.innerHTML = `<div class="empty">Timeline will populate after scenario loads and actions.</div>`;
    return;
  }
  timeline
    .slice()
    .reverse()
    .forEach((entry) => {
      const node = document.createElement("div");
      node.className = "timeline-entry";
      if (entry.kind === "scenario_loaded") {
        node.innerHTML = `<strong>${entry.label}</strong><p>${entry.summary}</p>`;
      } else {
        node.innerHTML = `
          <strong>${entry.label}</strong>
          <p>${entry.phase_before} -> ${entry.phase_after} | reward ${formatNumber(entry.reward)}</p>
        `;
      }
      container.appendChild(node);
    });
}

function renderModelPanel() {
  const metaRoot = document.querySelector("#model-meta");
  const chartRoot = document.querySelector("#model-chart");
  metaRoot.innerHTML = "";
  chartRoot.innerHTML = "";
  if (!state.modelInfo) return;
  const datasetStats = state.modelInfo.dataset_stats || {};
  const metrics = state.modelInfo.metrics || {};
  const finalMetrics = metrics.final || {};
  [
    ["Loaded", state.modelInfo.loaded ? "yes" : "no"],
    ["Run", state.modelInfo.run_dir ? state.modelInfo.run_dir.split("\\").pop() : "-"],
    ["Samples", datasetStats.total_records || metrics.train_samples || "-"],
    ["Best Val Loss", metrics.best_val_loss ? formatNumber(metrics.best_val_loss) : "-"],
    ["Val Acc@1", finalMetrics.val_acc1 ? formatNumber(finalMetrics.val_acc1) : "-"],
    ["Val Acc@3", finalMetrics.val_acc3 ? formatNumber(finalMetrics.val_acc3) : "-"],
  ].forEach(([label, value]) => metaRoot.appendChild(metricCard(label, value)));

  const history = state.modelInfo.history || [];
  if (!history.length) {
    chartRoot.innerHTML = `<div class="empty">No training curve available yet.</div>`;
    return;
  }
  const width = 520;
  const height = 180;
  const padding = 16;
  const losses = history.map((point) => Number(point.val_loss || point.loss || 0));
  const maxLoss = Math.max(...losses, 1);
  const minLoss = Math.min(...losses, 0);
  const xScale = (index) => padding + (index / Math.max(history.length - 1, 1)) * (width - padding * 2);
  const yScale = (value) => height - padding - ((value - minLoss) / Math.max(maxLoss - minLoss, 1e-6)) * (height - padding * 2);
  const points = history.map((point, index) => `${xScale(index)},${yScale(Number(point.val_loss || point.loss || 0))}`).join(" ");
  chartRoot.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      <rect x="0" y="0" width="${width}" height="${height}" rx="20" fill="rgba(255,255,255,0.72)"></rect>
      <polyline fill="none" stroke="#d3a046" stroke-width="3" points="${points}"></polyline>
      <text x="${padding}" y="${padding + 8}" fill="#6f665c" font-size="12">Validation loss per epoch</text>
      <text x="${padding}" y="${height - 10}" fill="#6f665c" font-size="11">epoch 1</text>
      <text x="${width - 70}" y="${height - 10}" fill="#6f665c" font-size="11">epoch ${history.length}</text>
    </svg>
  `;
}

async function refreshState() {
  state.currentState = await apiFetch("/api/state");
  state.currentScenarioId = state.currentState.scenario.id;
  renderStatus();
  renderResources();
  renderHand();
  renderTimeline();
}

async function refreshRecommendations() {
  const payload = await apiFetch("/api/recommend", {
    method: "POST",
    body: JSON.stringify({ policy: state.currentPolicy, topk: 3 }),
  });
  state.recommendations = payload.recommendations || [];
  state.selectedRecommendationIndex = 0;
  renderRecommendations();
  renderPreview();
  renderHand();
}

async function refreshModelInfo() {
  state.modelInfo = await apiFetch("/api/model_info");
  renderModelPanel();
  renderStatus();
}

async function loadScenario(scenarioId) {
  await apiFetch("/api/scenario/load", {
    method: "POST",
    body: JSON.stringify({ scenario_id: scenarioId }),
  });
  state.currentScenarioId = scenarioId;
  await refreshState();
  await refreshRecommendations();
  renderScenarios();
}

async function executeSelected() {
  const recommendation = state.recommendations[state.selectedRecommendationIndex];
  const action = recommendation ? recommendation.action : null;
  await apiFetch("/api/step", {
    method: "POST",
    body: JSON.stringify({ policy: state.currentPolicy, action }),
  });
  await refreshState();
  await refreshRecommendations();
}

async function autoplay() {
  await apiFetch("/api/autoplay", {
    method: "POST",
    body: JSON.stringify({ policy: state.currentPolicy, steps: 3 }),
  });
  await refreshState();
  await refreshRecommendations();
}

function bindControls() {
  document.querySelector("#reload-button").addEventListener("click", async () => {
    if (state.currentScenarioId) {
      await loadScenario(state.currentScenarioId);
    }
  });
  document.querySelector("#step-button").addEventListener("click", executeSelected);
  document.querySelector("#autoplay-button").addEventListener("click", autoplay);
  document.querySelectorAll(".toggle-button").forEach((button) => {
    button.addEventListener("click", async () => {
      state.currentPolicy = button.dataset.policy;
      document.querySelectorAll(".toggle-button").forEach((node) => node.classList.remove("active"));
      button.classList.add("active");
      await refreshRecommendations();
      renderStatus();
    });
  });
}

async function bootstrap() {
  bindControls();
  const scenariosPayload = await apiFetch("/api/scenarios");
  state.scenarios = scenariosPayload.scenarios || [];
  renderScenarios();
  await refreshModelInfo();
  await refreshState();
  await refreshRecommendations();
}

bootstrap().catch((error) => {
  document.body.innerHTML = `<div class="app-shell"><div class="panel empty">Failed to start demo UI: ${error.message}</div></div>`;
});
