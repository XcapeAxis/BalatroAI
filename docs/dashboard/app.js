(function () {
  "use strict";

  function asNumber(value, fallback) {
    var n = Number(value);
    return Number.isFinite(n) ? n : fallback;
  }

  function fmt(value, digits) {
    if (value === null || value === undefined || value === "") {
      return "-";
    }
    var n = Number(value);
    if (!Number.isFinite(n)) {
      return String(value);
    }
    return n.toFixed(digits);
  }

  function pct(value) {
    var n = Number(value);
    if (!Number.isFinite(n)) {
      return "-";
    }
    return (n * 100).toFixed(2) + "%";
  }

  function esc(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;");
  }

  function sparkline(values) {
    if (!Array.isArray(values) || values.length === 0) {
      return "-";
    }
    var ticks = "▁▂▃▄▅▆▇█";
    var nums = values.map(function (v) { return Number(v); }).filter(function (v) { return Number.isFinite(v); });
    if (nums.length === 0) {
      return "-";
    }
    var min = Math.min.apply(null, nums);
    var max = Math.max.apply(null, nums);
    if (max - min < 1e-12) {
      return "▅".repeat(nums.length);
    }
    return nums.map(function (v) {
      var idx = Math.round(((v - min) / (max - min)) * (ticks.length - 1));
      return ticks[Math.max(0, Math.min(ticks.length - 1, idx))];
    }).join("");
  }

  function tokenClass(status) {
    var t = String(status || "").toLowerCase();
    if (t.indexOf("pass") >= 0 || t.indexOf("good") >= 0 || t.indexOf("improv") >= 0 || t === "stable") {
      return "good";
    }
    if (t.indexOf("fail") >= 0 || t.indexOf("regress") >= 0 || t === "bad") {
      return "bad";
    }
    if (t.indexOf("noisy") >= 0 || t.indexOf("hold") >= 0 || t.indexOf("warn") >= 0) {
      return "warn";
    }
    return "";
  }

  function renderTable(el, headers, rows) {
    var head = "<thead><tr>" + headers.map(function (h) { return "<th>" + esc(h) + "</th>"; }).join("") + "</tr></thead>";
    var bodyRows = rows.map(function (r) {
      return "<tr>" + r.map(function (c) { return "<td>" + c + "</td>"; }).join("") + "</tr>";
    }).join("");
    el.innerHTML = "<table>" + head + "<tbody>" + bodyRows + "</tbody></table>";
  }

  function loadData() {
    if (window.DASHBOARD_DATA && typeof window.DASHBOARD_DATA === "object") {
      return Promise.resolve(window.DASHBOARD_DATA);
    }
    return fetch("./data/latest.json").then(function (r) {
      if (!r.ok) {
        throw new Error("failed to load data/latest.json: " + r.status);
      }
      return r.json();
    });
  }

  function render(data) {
    var latestGate = (data && data.latest_gate) || {};
    var alerts = (data && data.regression_alerts) || {};
    var summary = alerts.summary || {};
    var trendMetrics = (data && data.trend_metrics) || {};
    var avg = trendMetrics.avg_ante_reached || {};
    var med = trendMetrics.median_ante_reached || {};
    var win = trendMetrics.win_rate || {};
    var trendSignal = data.trend_signal || "unknown";
    var repo = data.repo || {};

    document.getElementById("build-meta").textContent =
      "Generated: " + (data.generated_at || "-") + " | Branch: " + (repo.branch || "-");
    document.getElementById("latest-gate").innerHTML =
      "<span class=\"" + tokenClass(latestGate.status) + "\">" + esc((latestGate.gate_name || "-") + " " + (latestGate.status || "-")) + "</span>";
    document.getElementById("trend-signal").innerHTML =
      "<span class=\"" + tokenClass(trendSignal) + "\">" + esc(trendSignal) + "</span>";
    document.getElementById("hard-regressions").textContent = String(summary.hard_regression || 0);
    document.getElementById("total-series").textContent = String(summary.total_series || 0);

    var chips = [
      { label: "Mainline", value: repo.on_mainline ? "yes" : "no" },
      { label: "Tree Clean", value: repo.working_tree_clean ? "yes" : "no" },
      { label: "Candidate", value: ((data.candidate || {}).decision || "n/a") },
      { label: "Release", value: ((data.release_state || {}).action || "n/a") }
    ];
    var chipsHtml = chips.map(function (chip) {
      var cls = tokenClass(chip.value);
      return "<span class=\"chip " + cls + "\">" + esc(chip.label + ": " + chip.value) + "</span>";
    }).join("");
    document.getElementById("status-chips").innerHTML = chipsHtml;

    renderTable(
      document.getElementById("trend-table"),
      ["Metric", "Latest", "Baseline", "Delta", "Pct", "Sparkline"],
      [
        ["avg_ante_reached", fmt(avg.latest_value, 4), fmt(avg.baseline_median, 4), fmt(avg.delta, 4), pct(avg.pct_change), esc(sparkline(avg.sparkline_values))],
        ["median_ante_reached", fmt(med.latest_value, 4), fmt(med.baseline_median, 4), fmt(med.delta, 4), pct(med.pct_change), esc(sparkline(med.sparkline_values))],
        ["win_rate", fmt(win.latest_value, 4), fmt(win.baseline_median, 4), fmt(win.delta, 4), pct(win.pct_change), esc(sparkline(win.sparkline_values))]
      ]
    );

    var alertPairs = [
      ["hard_regression", summary.hard_regression || 0],
      ["soft_regression", summary.soft_regression || 0],
      ["noisy_needs_more_data", summary.noisy_needs_more_data || 0],
      ["improvement", summary.improvement || 0],
      ["no_signal", summary.no_signal || 0]
    ];
    document.getElementById("alert-list").innerHTML = alertPairs.map(function (pair) {
      var cls = tokenClass(pair[0]);
      return "<li><span class=\"" + cls + "\">" + esc(pair[0]) + "</span>: " + esc(pair[1]) + "</li>";
    }).join("");

    var champion = data.champion || {};
    var candidate = data.candidate || {};
    var cc = [
      "champion.exp_id: " + (champion.exp_id || "n/a"),
      "champion.status: " + (champion.status || "n/a"),
      "candidate.decision: " + (candidate.decision || "n/a"),
      "candidate.top: " + (candidate.top_candidate_exp_id || "n/a"),
      "candidate.reason: " + (candidate.reason || "n/a")
    ];
    document.getElementById("cc-list").innerHTML = cc.map(function (row) {
      return "<li>" + esc(row) + "</li>";
    }).join("");

    var runs = Array.isArray(data.recent_runs) ? data.recent_runs.slice(0, 15) : [];
    renderTable(
      document.getElementById("runs-table"),
      ["Run ID", "Milestone", "Gate", "Avg Ante", "Median", "Win Rate", "Timestamp"],
      runs.map(function (run) {
        var gate = String(run.gate_status || "UNKNOWN");
        return [
          esc(run.run_id || "-"),
          esc(run.milestone || "-"),
          "<span class=\"" + tokenClass(gate) + "\">" + esc(gate) + "</span>",
          esc(fmt(run.avg_ante_reached, 4)),
          esc(fmt(run.median_ante_reached, 4)),
          esc(fmt(run.win_rate, 4)),
          esc(run.timestamp || "-")
        ];
      })
    );

    var gates = Array.isArray(data.gate_history) ? data.gate_history.slice(0, 20) : [];
    renderTable(
      document.getElementById("gates-table"),
      ["Gate", "Status", "Run ID", "Milestone", "Timestamp"],
      gates.map(function (row) {
        var status = String(row.status || "UNKNOWN");
        return [
          esc(row.gate_name || "-"),
          "<span class=\"" + tokenClass(status) + "\">" + esc(status) + "</span>",
          esc(row.run_id || "-"),
          esc(row.milestone || "-"),
          esc(row.timestamp || "-")
        ];
      })
    );
  }

  function renderError(err) {
    var message = String((err && err.message) || err || "Unknown dashboard error");
    document.getElementById("build-meta").textContent = message;
    document.getElementById("trend-signal").textContent = "error";
  }

  loadData().then(render).catch(renderError);
})();
