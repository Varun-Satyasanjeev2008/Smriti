const API_BASE_URL = (typeof window !== "undefined" && window.location.port === "8000")
  ? ""
  : "http://localhost:8000";

let isMockMode = false;
let isBackendReachable = false;

document.addEventListener("DOMContentLoaded", () => {
  initHealthCheck();
  initTabs();
  initFormControls();
  initResearchForm();
  initReflectionTab();
  initLogsTab();
});


async function initHealthCheck() {
  const statusDot = document.getElementById("statusDot");
  const statusText = document.getElementById("statusText");
  const mockToggle = document.getElementById("mockToggle");
  const toggleText = document.getElementById("toggleText");

  async function check() {
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 2000);

      const res = await fetch(`${API_BASE_URL}/health`, { signal: controller.signal });
      clearTimeout(timeoutId);

      if (res.ok) {
        const data = await res.json();
        isBackendReachable = true;
        statusDot.className = "status-dot dot-online";
        statusText.textContent = `Backend Online (${data.service || "FastAPI"})`;

        // Automatically switch mock mode OFF when real backend is detected
        if (!mockToggle.dataset.userModified && isMockMode) {
          setMockMode(false);
        }
      } else {
        throw new Error("Bad response");
      }
    } catch {
      isBackendReachable = false;
      statusDot.className = "status-dot dot-offline";
      statusText.textContent = "Backend Offline (:8000)";

      // Auto-enable mock mode if user hasn't explicitly set it yet
      if (!mockToggle.dataset.userModified) {
        setMockMode(true);
      }
    }
  }

  // Handle Mock Mode Toggle
  mockToggle.addEventListener("change", (e) => {
    mockToggle.dataset.userModified = "true";
    setMockMode(e.target.checked);
  });

  function setMockMode(active) {
    isMockMode = active;
    mockToggle.checked = active;
    toggleText.textContent = active ? "Mock Mode: ON" : "Mock Mode: Off";
    if (active) {
      statusText.textContent = isBackendReachable
        ? "Mock Mode Active"
        : "Mock Mode Active (Backend Offline)";
    }
  }

  // Initial check and periodic polling every 10 seconds
  await check();
  setInterval(check, 10000);
}

function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  const contents = document.querySelectorAll(".tab-content");

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => {
        b.classList.remove("active");
        b.setAttribute("aria-selected", "false");
      });
      contents.forEach((c) => (c.style.display = "none"));

      btn.classList.add("active");
      btn.setAttribute("aria-selected", "true");

      const targetId = btn.dataset.tab;
      const targetContent = document.getElementById(targetId);
      if (targetContent) {
        targetContent.style.display = "block";
      }

      // Auto-load logs if user switches to logs tab
      if (targetId === "logsTab") {
        fetchAuditLogs();
      }
    });
  });
}

function initFormControls() {
  // Advanced filters toggle
  const toggleBtn = document.getElementById("toggleFiltersBtn");
  const filtersDiv = document.getElementById("advancedFilters");
  const arrow = document.getElementById("filterArrow");

  toggleBtn.addEventListener("click", () => {
    const isHidden = filtersDiv.style.display === "none";
    filtersDiv.style.display = isHidden ? "grid" : "none";
    arrow.textContent = isHidden ? "▲" : "▼";
  });

  // Max patents slider value display
  const maxSlider = document.getElementById("maxPatentsInput");
  const maxVal = document.getElementById("maxPatentsVal");
  maxSlider.addEventListener("input", (e) => {
    maxVal.textContent = e.target.value;
  });

  // Example Query loader
  const exampleBtn = document.getElementById("exampleBtn");
  const queryInput = document.getElementById("queryInput");
  const cpcInput = document.getElementById("cpcInput");

  exampleBtn.addEventListener("click", () => {
    queryInput.value = "Neural network image classification using convolutional layers and residual skip connections";
    cpcInput.value = "G06N3/04";
    maxSlider.value = 4;
    maxVal.textContent = "4";
    queryInput.focus();
  });
}

// -----------------------------------------------------------------------------
// POST /research — Prior Art Research
// -----------------------------------------------------------------------------

function initResearchForm() {
  const form = document.getElementById("researchForm");
  const submitBtn = document.getElementById("submitBtn");
  const btnText = submitBtn.querySelector(".btn-text");
  const spinner = submitBtn.querySelector(".spinner");

  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const errorTitle = document.getElementById("errorTitle");
  const errorMsg = document.getElementById("errorMsg");
  const resultsSection = document.getElementById("resultsSection");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const query = document.getElementById("queryInput").value.trim();
    const cpcClass = document.getElementById("cpcInput").value.trim() || null;
    const maxPatents = parseInt(document.getElementById("maxPatentsInput").value, 10) || 5;
    const dateFrom = document.getElementById("dateFromInput").value || null;
    const dateTo = document.getElementById("dateToInput").value || null;

    if (query.length < 5) {
      alert("Research query must be at least 5 characters long.");
      return;
    }

    // UI Loading state
    submitBtn.disabled = true;
    btnText.textContent = "Analyzing Patents...";
    spinner.style.display = "inline-block";
    loadingState.style.display = "block";
    errorState.style.display = "none";
    resultsSection.style.display = "none";

    const payload = {
      query,
      cpc_class: cpcClass,
      max_patents: maxPatents,
      date_from: dateFrom,
      date_to: dateTo,
    };

    try {
      let data;
      if (isMockMode) {
        // Simulate network latency for mock response
        await new Promise((resolve) => setTimeout(resolve, 1200));
        data = generateMockResearchResponse(payload);
      } else {
        const response = await fetch(`${API_BASE_URL}/research`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.detail || `Server responded with HTTP ${response.status}`);
        }
        data = await response.json();
      }

      // Render the response
      renderResearchResults(data);
      resultsSection.style.display = "block";
      resultsSection.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      errorTitle.textContent = isMockMode ? "Mock Simulation Failed" : "Research Pipeline Failed";
      errorMsg.textContent = `${err.message}. If the backend server isn't running on port 8000, you can toggle "Mock Mode" at the top right to explore the interface.`;
      errorState.style.display = "flex";
    } finally {
      submitBtn.disabled = false;
      btnText.textContent = "Run Prior-Art Research";
      spinner.style.display = "none";
      loadingState.style.display = "none";
    }
  });
}

// -----------------------------------------------------------------------------
// Render Results (Summary, Memories, Findings)
// -----------------------------------------------------------------------------

function renderResearchResults(data) {
  // 1. Summary Banner
  const summaryText = document.getElementById("summaryText");
  const statPills = document.getElementById("statPills");
  summaryText.textContent = data.summary || "Research completed successfully.";

  const findings = data.findings || [];
  const acceptedCount = findings.filter((f) => f.verdict === "ACCEPTED").length;
  const rejectedCount = findings.length - acceptedCount;

  statPills.innerHTML = `
    <span class="stat-pill stat-pill-accepted">✓ ${acceptedCount} Accepted Prior-Art</span>
    <span class="stat-pill stat-pill-rejected">✕ ${rejectedCount} Rejected</span>
  `;

  // 2. Recalled Context (Hindsight Memory)
  const recalledSection = document.getElementById("recalledSection");
  const memoryList = document.getElementById("memoryList");
  const recalledCount = document.getElementById("recalledCount");
  const memories = data.recalled_context || [];

  // Backend memories may be plain strings or objects, and the text field name
  // varies (Hindsight returns `text`; the mock uses `content`).
  const getMemoryText = (m) => {
    if (typeof m === "string") return m;
    if (!m || typeof m !== "object") return "";
    const t = m.content ?? m.text ?? m.memory ?? m.summary ?? m.reasoning;
    return t != null && t !== "" ? String(t) : JSON.stringify(m, null, 2);
  };

  if (memories.length > 0) {
    console.log("Recalled context sample:", memories[0]);
    recalledCount.textContent = memories.length;
    memoryList.innerHTML = memories
      .map((m) => {
        const score = typeof m === "object" && m ? (m.score ?? m.relevance) : null;
        return `
        <div class="memory-item">
          ${score ? `<span class="memory-score">Match Score: ${(score * 100).toFixed(0)}%</span>` : ""}
          <p>${escapeHtml(getMemoryText(m))}</p>
        </div>
      `;
      })
      .join("");
    recalledSection.style.display = "block";
  } else {
    recalledSection.style.display = "none";
  }

  // 3. Evaluated Patent Findings
  const findingsCount = document.getElementById("findingsCount");
  const findingsList = document.getElementById("findingsList");
  findingsCount.textContent = findings.length;

  if (findings.length === 0) {
    findingsList.innerHTML = `
      <div class="card text-center">
        <p>No candidate patents found matching this query in the configured public patent sources.</p>
      </div>
    `;
    return;
  }

  findingsList.innerHTML = findings
    .map((patent) => {
      const isAccepted = patent.verdict === "ACCEPTED";
      const verdictBadge = isAccepted
        ? `<span class="badge badge-accepted">✓ Accepted Prior-Art</span>`
        : `<span class="badge badge-rejected">✕ Rejected</span>`;

      const confClass = `badge-confidence-${(patent.confidence || "MEDIUM").toLowerCase()}`;
      const confidenceBadge = `<span class="badge ${confClass}">Confidence: ${patent.confidence}</span>`;

      const cpcTags = (patent.cpc_codes || [])
        .map((cpc) => `<span class="cpc-tag">${escapeHtml(cpc)}</span>`)
        .join("");
      const patentId = String(patent.patent_id || "");
      // Display the patent ID exactly as returned by the backend.
      // Do NOT prepend "US " -- the backend already includes the country prefix
      // for all jurisdictions (US, EP, CN, WO, etc.).
      const patentIdLabel = patentId;

      return `
        <article class="patent-card">
          <div class="patent-card-header">
            <div class="patent-id-title">
              <span class="patent-id-badge">${escapeHtml(patentIdLabel)}</span>
              <h4 class="patent-title">${escapeHtml(patent.title)}</h4>
            </div>
            <div class="badge-group">
              ${verdictBadge}
              ${confidenceBadge}
            </div>
          </div>

          <div class="patent-meta">
            ${patent.assignee ? `<span class="meta-item">Assignee: <strong>${escapeHtml(patent.assignee)}</strong></span>` : ""}
            ${patent.filing_date ? `<span class="meta-item">Filing: <strong>${escapeHtml(patent.filing_date)}</strong></span>` : ""}
            ${patent.grant_date ? `<span class="meta-item">Grant: <strong>${escapeHtml(patent.grant_date)}</strong></span>` : ""}
            ${patent.publication_date ? `<span class="meta-item">Publication: <strong>${escapeHtml(patent.publication_date)}</strong></span>` : ""}
            ${patent.source ? `<span class="meta-item">Source: <strong>${escapeHtml(patent.source)}</strong></span>` : ""}
          </div>

          ${cpcTags ? `<div class="cpc-tags">${cpcTags}</div>` : ""}

          ${
            patent.abstract
              ? `
              <div class="abstract-box">
                <strong>Abstract:</strong> ${escapeHtml(patent.abstract)}
              </div>
            `
              : ""
          }

          <div class="reasoning-box">
            <div class="reasoning-label">💡 Agent Evaluation Reasoning</div>
            <p class="reasoning-text">"${escapeHtml(patent.reasoning)}"</p>
          </div>
        </article>
      `;
    })
    .join("");
}

// -----------------------------------------------------------------------------
// TAB 2: GET /reflect — Cross-Research Patterns
// -----------------------------------------------------------------------------

function initReflectionTab() {
  const runBtn = document.getElementById("runReflectBtn");
  const queryInput = document.getElementById("reflectQueryInput");
  const resultBox = document.getElementById("reflectResult");
  const content = document.getElementById("reflectionContent");
  const btnText = runBtn.querySelector(".btn-text");
  const spinner = runBtn.querySelector(".spinner");

  runBtn.addEventListener("click", async () => {
    runBtn.disabled = true;
    btnText.textContent = "Reflecting...";
    spinner.style.display = "inline-block";
    resultBox.style.display = "none";

    const focusQuery = queryInput.value.trim();
    const endpoint = focusQuery
      ? `${API_BASE_URL}/reflect?query=${encodeURIComponent(focusQuery)}`
      : `${API_BASE_URL}/reflect`;

    try {
      let reflectionText = "";
      if (isMockMode) {
        await new Promise((r) => setTimeout(r, 900));
        reflectionText = focusQuery
          ? `Based on 42 accumulated patent evaluations for query "${focusQuery}": Most relevant prior-art clusters in CPC class G06N3 (neural networks) with an 82% acceptance rate. The primary reason for rejection is non-analogous art (lacking trainable skip-connections).`
          : `Synthesised memory across 58 patent research sessions: Common prior-art overlaps frequently occur between G06N3/04 and G06V10/82. Inventions with explicit mathematical claim formulations have 35% higher prior-art rejection rates due to novelty over earlier academic patents.`;
      } else {
        const res = await fetch(endpoint);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        reflectionText = data.reflection || "No reflection available.";
      }

      content.textContent = reflectionText;
      resultBox.style.display = "block";
    } catch (err) {
      alert(`Could not fetch reflection: ${err.message}`);
    } finally {
      runBtn.disabled = false;
      btnText.textContent = "Generate Reflection";
      spinner.style.display = "none";
    }
  });
}

// -----------------------------------------------------------------------------
// TAB 3: GET /logs — Audit Logs
// -----------------------------------------------------------------------------

function initLogsTab() {
  const refreshBtn = document.getElementById("refreshLogsBtn");
  refreshBtn.addEventListener("click", fetchAuditLogs);
}

async function fetchAuditLogs() {
  const tbody = document.getElementById("logsTableBody");
  tbody.innerHTML = `<tr><td colspan="8" class="text-center">Loading logs from database...</td></tr>`;

  try {
    let logs = [];
    if (isMockMode) {
      await new Promise((r) => setTimeout(r, 400));
      logs = [
        {
          id: 1,
          query: "Convolutional neural network for image recognition",
          cpc_class: "G06N3/04",
          patents_searched: 4,
          patents_accepted: 3,
          patents_rejected: 1,
          recalled_memories: 2,
          latency_ms: 3840,
          created_at: new Date().toISOString(),
        },
        {
          id: 2,
          query: "Reinforcement learning robotic arm trajectory planning",
          cpc_class: "B25J9/16",
          patents_searched: 5,
          patents_accepted: 1,
          patents_rejected: 4,
          recalled_memories: 0,
          latency_ms: 4120,
          created_at: new Date(Date.now() - 3600000).toISOString(),
        },
      ];
    } else {
      const res = await fetch(`${API_BASE_URL}/logs?limit=20`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      logs = data.logs || [];
    }

    if (logs.length === 0) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-center">No research audit logs found in database.</td></tr>`;
      return;
    }

    tbody.innerHTML = logs
      .map(
        (log) => `
        <tr>
          <td>${formatDate(log.created_at)}</td>
          <td><strong>${escapeHtml(log.query)}</strong></td>
          <td><code>${log.cpc_class ? escapeHtml(log.cpc_class) : "—"}</code></td>
          <td>${log.patents_searched}</td>
          <td><span style="color: var(--success); font-weight: 600;">${log.patents_accepted}</span></td>
          <td><span style="color: var(--danger); font-weight: 600;">${log.patents_rejected}</span></td>
          <td>${log.recalled_memories}</td>
          <td>${log.latency_ms ? `${Math.round(log.latency_ms)}ms` : "—"}</td>
        </tr>
      `
      )
      .join("");
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8" class="text-center" style="color: var(--danger);">Failed to load logs: ${err.message}</td></tr>`;
  }
}

// -----------------------------------------------------------------------------
// Mock Response Generator (for Demo / Standalone Testing)
// -----------------------------------------------------------------------------

function generateMockResearchResponse(req) {
  return {
    query: req.query,
    summary: `Research complete in 3840ms. Hindsight memory recalled 1 previous investigation (reusing past verdict & reasoning). Evaluated ${req.max_patents} candidate patent(s): 2 accepted as prior art, ${Math.max(0, req.max_patents - 2)} rejected.`,
    recalled_context: [
      {
        content: `PATENT PRIOR ART RESEARCH (Retained from previous session)\nResearch Query: "${req.query}"\nPatent ID: US 9,876,543 (Deep residual CNN architectures)\nVerdict: ACCEPTED (Relevance Score: 94%)\nReasoning: Previous investigation confirmed direct anticipation in Claim 1 regarding residual shortcut connections bypassing convolutional layers. Conclusion reused — redundant public patent-source re-evaluation skipped.`,
        score: 0.94,
        metadata: { patent_id: "9,876,543", verdict: "ACCEPTED" },
      },
    ],
    findings: [
      {
        patent_id: "10,410,105",
        title: "Deep convolutional neural network for image recognition",
        abstract:
          "Systems and methods for image recognition using deep neural network architectures comprising hierarchical convolutional layers, pooling operations, and batch normalization layers to classify high-resolution images.",
        assignee: "Acme Cognitive Systems Corp",
        cpc_codes: [req.cpc_class || "G06N3/04", "G06V10/82"],
        filing_date: "2017-03-15",
        grant_date: "2019-09-10",
        verdict: "ACCEPTED",
        confidence: "HIGH",
        reasoning:
          "This patent directly discloses a convolutional neural network architecture for image classification that anticipates the queried invention. Claim 1 recites 'a plurality of convolutional layers' applied to image data, which squarely reads on the query.",
      },
      {
        patent_id: "10,123,456",
        title: "Method and apparatus for training multi-layer neural network models",
        abstract:
          "An accelerated training procedure utilizing mini-batch stochastic gradient descent with momentum adjustments across distributed computing nodes for deep visual classifiers.",
        assignee: "Global AI Technologies Ltd",
        cpc_codes: ["G06N3/08", "G06F17/16"],
        filing_date: "2018-06-20",
        grant_date: "2020-04-14",
        verdict: "ACCEPTED",
        confidence: "MEDIUM",
        reasoning:
          "Provides relevant prior art for optimization and layer configuration, though focused primarily on the distributed training loop rather than runtime classification.",
      },
      {
        patent_id: "9,654,321",
        title: "Database indexing method using hashed feature representations",
        abstract:
          "Techniques for fast retrieval of unstructured multimedia documents using hash-based index tables and proximity distance calculations.",
        assignee: "DataCloud Systems Inc",
        cpc_codes: ["G06F16/245", "G06F16/58"],
        filing_date: "2014-11-12",
        grant_date: "2017-01-22",
        verdict: "REJECTED",
        confidence: "HIGH",
        reasoning:
          "Non-analogous art. The patent concerns general database index structures for media storage and lacks trainable neural network weights or convolutional feature extraction layers.",
      },
    ].slice(0, req.max_patents),
  };
}

// -----------------------------------------------------------------------------
// Utilities
// -----------------------------------------------------------------------------

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function formatDate(isoStr) {
  if (!isoStr) return "—";
  try {
    const d = new Date(isoStr);
    return d.toLocaleString(undefined, {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return isoStr;
  }
}