// ══════════════════════════════════════════════════════════════════════════════
// LegalRAG - Document Comparison Application
// ══════════════════════════════════════════════════════════════════════════════

// ── Global State ──────────────────────────────────────────────────────────────
let fileA = null;
let fileB = null;
let currentFilter = 'all';
let previewUrls = { a: null, b: null };

// ── Configuration ─────────────────────────────────────────────────────────────
const STEP_DEFINITIONS = [
  { title: "Document Structure Analysis" },
  { title: "Vector Embedding Generation" },
  { title: "Semantic Matching" },
  { title: "LLM Analysis" },
  { title: "Report Generation" },
];

const API_ENDPOINT = "/api/compare";

// ── Citation Store ─────────────────────────────────────────────────────────────
// Key: citation id (article label), Value: {type, similarity, text_a, text_b, page_a, page_b}
const citationStore = {};

// ── Report Store ──────────────────────────────────────────────────────────────
const reportStore = {};
let reportCounter = 0;

// ── Utility Functions ─────────────────────────────────────────────────────────

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

/**
 * Hide empty state from chat area
 */
function hideEmptyState() {
  const el = document.getElementById("empty-state");
  if (el) el.remove();
}

/**
 * Append message element to chat area with smooth scroll
 */
function appendMessage(element) {
  const chat = document.getElementById("chat");
  chat.appendChild(element);
  element.scrollIntoView({ behavior: "smooth", block: "end" });
}

// ── File Upload Management ───────────────────────────────────────────────────

/**
 * Setup file input handlers
 */
function setupFileInput(inputId, slotId, nameId, checkId, hintId, varSetter) {
  const input = document.getElementById(inputId);
  const slot = document.getElementById(slotId);
  const nameEl = document.getElementById(nameId);
  const checkEl = document.getElementById(checkId);
  const hintEl = document.getElementById(hintId);

  input.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (!file) return;

    varSetter(file);
    nameEl.textContent = file.name;
    nameEl.style.display = "block";
    hintEl.style.display = "none";
    slot.classList.add("has-file");
    checkEl.style.display = "flex";
    updateCompareButton();
  });
}

/**
 * Update compare button state based on file selection
 */
function updateCompareButton() {
  const btn = document.getElementById("btn-compare");
  btn.disabled = !(fileA && fileB);
}

// Initialize file inputs
setupFileInput("input-a", "slot-a", "name-a", "check-a", "hint-a", (f) => (fileA = f));
setupFileInput("input-b", "slot-b", "name-b", "check-b", "hint-b", (f) => (fileB = f));

// ── Progress Card Management ──────────────────────────────────────────────────

/**
 * Create progress tracking card
 */
function createProgressCard(nameA, nameB) {
  const card = document.createElement("div");
  card.className = "msg";
  card.id = "progress-card";

  const stepsHtml = STEP_DEFINITIONS.map(
    (s, i) => `
    <div class="step waiting" id="step-${i + 1}">
      <span class="step-indicator">${i + 1}</span>
      <div class="step-content">
        <div class="step-title">${s.title}</div>
        <div class="step-detail" id="step-detail-${i + 1}">Waiting...</div>
      </div>
    </div>
  `
  ).join("");

  card.innerHTML = `
    <div class="progress-card">
      <div class="progress-card-header">
        <div class="ai-badge">Processing</div>
        <div class="progress-info">
          <div class="progress-title">
            Analyzing Documents
            <span class="thinking-indicator"><span></span><span></span><span></span></span>
          </div>
          <div class="progress-subtitle">${escapeHtml(nameA)} → ${escapeHtml(nameB)}</div>
        </div>
      </div>
      <div class="steps" id="steps-list">
        ${stepsHtml}
      </div>
    </div>
  `;

  return card;
}

/**
 * Update step status and detail
 */
function updateStep(stepNumber, status, detail) {
  const el = document.getElementById(`step-${stepNumber}`);
  const detailEl = document.getElementById(`step-detail-${stepNumber}`);
  if (!el || !detailEl) return;

  el.className = `step ${status}`;
  detailEl.textContent = detail || "";
  el.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

/**
 * Display statistics summary
 */
function showStatistics(stats) {
  const card = document.getElementById("progress-card")?.querySelector(".progress-card");
  if (!card) return;

  // Remove thinking indicator
  const dots = card.querySelector(".thinking-indicator");
  if (dots) dots.remove();

  // Update badge to complete state
  const badge = card.querySelector(".ai-badge");
  if (badge) {
    badge.textContent = "Complete";
    badge.style.background = "rgba(34, 197, 94, 0.1)";
    badge.style.borderColor = "var(--green)";
    badge.style.color = "var(--green)";
  }

  // Create stats row
  const statsEl = document.createElement("div");
  statsEl.className = "stats-row";
  statsEl.innerHTML = `
    <div class="stat-card modified" data-filter-type="MODIFIED" onclick="applyFilter('MODIFIED')" title="Click để lọc điều khoản sửa đổi">
      <div class="stat-label">Modified</div>
      <div class="stat-num">${stats.modified}</div>
    </div>
    <div class="stat-card added" data-filter-type="ADDED" onclick="applyFilter('ADDED')" title="Click để lọc điều khoản thêm mới">
      <div class="stat-label">Added</div>
      <div class="stat-num">${stats.added}</div>
    </div>
    <div class="stat-card deleted" data-filter-type="DELETED" onclick="applyFilter('DELETED')" title="Click để lọc điều khoản bị xóa">
      <div class="stat-label">Deleted</div>
      <div class="stat-num">${stats.deleted}</div>
    </div>
    <div class="stat-card unchanged" data-filter-type="all" onclick="applyFilter('all')" title="Click để xem tất cả">
      <div class="stat-label">Unchanged</div>
      <div class="stat-num">${stats.unchanged}</div>
    </div>
  `;
  card.appendChild(statsEl);
}

// ── Citation Management ───────────────────────────────────────────────────────

/**
 * Store citation data received from server
 */
function storeCitations(items) {
  items.forEach(item => {
    // Use the article id as key (may have duplicates for different types; use type+id)
    const key = `${item.type}::${item.id}`;
    citationStore[key] = item;
  });
}

/**
 * Open source panel with citation data
 */
function openCitationPanel(key) {
  const data = citationStore[key];
  if (!data) return;

  const panel = document.getElementById("source-panel");
  const label = document.getElementById("source-panel-label");
  const badge = document.getElementById("source-badge");
  const blockA = document.getElementById("source-block-a");
  const blockB = document.getElementById("source-block-b");
  const textA = document.getElementById("source-text-a");
  const textB = document.getElementById("source-text-b");
  const pageA = document.getElementById("source-page-a");
  const pageB = document.getElementById("source-page-b");
  const nameA = document.getElementById("source-doc-name-a");
  const nameB = document.getElementById("source-doc-name-b");

  label.textContent = `Điều khoản: ${data.id}`;
  badge.textContent = data.type;
  badge.className = `source-badge type-${data.type.toLowerCase()}`;

  // File names from global state
  nameA.textContent = data.filename_a || (fileA ? fileA.name : "Tài liệu cũ");
  nameB.textContent = data.filename_b || (fileB ? fileB.name : "Tài liệu mới");

  // Show/hide blocks based on type
  if (data.type === "ADDED") {
    blockA.style.display = "none";
    blockB.style.display = "flex";
  } else if (data.type === "DELETED") {
    blockA.style.display = "flex";
    blockB.style.display = "none";
  } else {
    blockA.style.display = "flex";
    blockB.style.display = "flex";
  }

  // Populate text with highlight
  if (data.text_a) {
    textA.innerHTML = highlightText(escapeHtml(data.text_a));
  }
  if (data.text_b) {
    textB.innerHTML = highlightText(escapeHtml(data.text_b));
  }

  // Page info
  pageA.textContent = data.page_a ? `Trang ${data.page_a}` : "";
  pageB.textContent = data.page_b ? `Trang ${data.page_b}` : "";

  // Activate panel
  panel.classList.add("open");
  document.getElementById("content-area").classList.add("panel-open");

  // Load PDF into frames with page numbers
  if (previewUrls.a) {
    const pageHash = data.page_a ? `#page=${data.page_a}` : "";
    document.getElementById("pdf-frame-a").src = `${previewUrls.a}${pageHash}`;
  }
  if (previewUrls.b) {
    const pageHash = data.page_b ? `#page=${data.page_b}` : "";
    document.getElementById("pdf-frame-b").src = `${previewUrls.b}${pageHash}`;
  }

  // Switch to relevant tab based on change type
  if (data.type === "ADDED") {
    switchSourceTab('v2');
  } else {
    switchSourceTab('v1');
  }

  // Mark active citation chip
  document.querySelectorAll(".citation-chip").forEach(c => c.classList.remove("active"));
  const chip = document.querySelector(`.citation-chip[data-key="${CSS.escape(key)}"]`);
  if (chip) chip.classList.add("active");
}

/**
 * Switch between V1 and V2 tabs in source panel
 */
function switchSourceTab(tab) {
  const tabV1 = document.getElementById("tab-v1");
  const tabV2 = document.getElementById("tab-v2");
  const containerV1 = document.getElementById("viewer-container-a");
  const containerV2 = document.getElementById("viewer-container-b");

  if (tab === 'v1') {
    tabV1.classList.add("active");
    tabV2.classList.remove("active");
    containerV1.style.display = "flex";
    containerV2.style.display = "none";
  } else {
    tabV1.classList.remove("active");
    tabV2.classList.add("active");
    containerV1.style.display = "none";
    containerV2.style.display = "flex";
  }
}

/**
 * Close source panel
 */
function closeSourcePanel() {
  const panel = document.getElementById("source-panel");
  panel.classList.remove("open");
  document.getElementById("content-area").classList.remove("panel-open");
  document.querySelectorAll(".citation-chip").forEach(c => c.classList.remove("active"));
}
function highlightText(htmlText) {
  return `<mark class="source-highlight">${htmlText.replace(/\n\n/g, '</mark><br><br><mark class="source-highlight">').replace(/\n/g, '<br>')}</mark>`;
}
function injectCitationChips(container) {
  const headings = container.querySelectorAll("h3");
  headings.forEach(h3 => {
    const text = h3.textContent.trim();
    const matchedKey = findCitationKey(text);
    if (matchedKey) {
      const chip = document.createElement("button");
      chip.className = "citation-chip";
      chip.dataset.key = matchedKey;
      chip.title = "Click để xem văn bản gốc";
      chip.innerHTML = `<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg> Xem nội dung gốc`;
      chip.onclick = () => openCitationPanel(matchedKey);
      h3.appendChild(chip);
    }
  });

  const blockquotes = container.querySelectorAll("blockquote");
  blockquotes.forEach(bq => {
    const text = bq.textContent;
    const matchedKey = findCitationKey(text);
    if (matchedKey) {
      bq.classList.add("has-citation");
      bq.style.cursor = "pointer";
      bq.onclick = () => openCitationPanel(matchedKey);
      bq.title = "Click để xem văn bản gốc";
    }
  });
}
function findCitationKey(text) {
  for (const key of Object.keys(citationStore)) {
    const id = citationStore[key].id;
    if (id && text.includes(id)) {
      return key;
    }
  }
  return null;
}
function showReport(markdown) {
  const wrap = document.createElement("div");
  wrap.className = "msg report-wrapper";
  wrap.dataset.type = "all";

  const renderedHtml = DOMPurify.sanitize(marked.parse(markdown));
  const reportId = ++reportCounter;
  reportStore[reportId] = markdown;

  wrap.innerHTML = `
    <div class="report-card">
      <div class="report-header">
        <div class="report-title">Comparison Report</div>
        <div class="report-actions">
          <button class="btn-icon" id="btn-expand-all" title="Mở rộng tất cả" onclick="toggleAllSections(true)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="7 13 12 18 17 13"></polyline>
              <polyline points="7 6 12 11 17 6"></polyline>
            </svg>
          </button>
          <button class="btn-icon" id="btn-collapse-all" title="Thu gọn tất cả" onclick="toggleAllSections(false)">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <polyline points="7 11 12 6 17 11"></polyline>
              <polyline points="7 18 12 13 17 18"></polyline>
            </svg>
          </button>
          <button class="btn-icon" title="Copy Markdown" onclick="copyReport(this, ${reportId})">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect>
              <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
          </button>
          <button class="btn-icon" title="Download" onclick="downloadReport(${reportId})">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
              <polyline points="7 10 12 15 17 10"></polyline>
              <line x1="12" y1="15" x2="12" y2="3"></line>
            </svg>
          </button>
        </div>
      </div>
      <div class="report-body">
        <div class="md-content" id="md-content-${reportId}">${renderedHtml}</div>
      </div>
    </div>
  `;

  appendMessage(wrap);

  // After DOM insertion: inject citation chips & color badges
  const mdContent = wrap.querySelector(`#md-content-${reportId}`);
  if (mdContent) {
    postProcessMarkdown(mdContent);
    makeReportCollapsible(mdContent);
    injectCitationChips(mdContent);
  }

  document.getElementById("filter-section").style.display = "block";
}

/**
 * Convert [TAGS] into colored badges
 */
function postProcessMarkdown(container) {
  const walk = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
  let node;
  const nodesToReplace = [];

  while (node = walk.nextNode()) {
    if (node.textContent.includes('[') && node.textContent.includes(']')) {
      nodesToReplace.push(node);
    }
  }

  nodesToReplace.forEach(textNode => {
    const parent = textNode.parentNode;
    if (!parent) return;

    const html = textNode.textContent.replace(/\[(THỰC CHẤT|HÌNH THỨC|Sửa đổi|Bổ sung|Loại bỏ|Số liệu|Thời hạn|Lỗi chính tả|Định dạng|Ngôn ngữ chuyên môn|Cấu trúc danh sách|Khoảng trắng)\]/g, (match, tag) => {
      const className = tag === 'THỰC CHẤT' ? 'badge-substantial' : 
                        tag === 'HÌNH THỨC' ? 'badge-formal' : 'badge-detail';
      return `<span class="inline-badge ${className}">${tag}</span>`;
    });

    if (html !== textNode.textContent) {
      const span = document.createElement('span');
      span.innerHTML = html;
      parent.replaceChild(span, textNode);
    }
  });
}

function makeReportCollapsible(container) {
  const h2List = container.querySelectorAll("h2");
  h2List.forEach(h2 => {
    h2.classList.add("collapsible-heading");
    h2.setAttribute("data-collapsed", "false");

    // Collect all sibling elements until next h2
    const siblings = [];
    let el = h2.nextElementSibling;
    while (el && el.tagName !== "H2") {
      siblings.push(el);
      el = el.nextElementSibling;
    }

    if (siblings.length > 0) {
      const wrapper = document.createElement("div");
      wrapper.className = "collapsible-body";
      siblings[0].parentNode.insertBefore(wrapper, siblings[0]);
      siblings.forEach(s => wrapper.appendChild(s));

      const icon = document.createElement("span");
      icon.className = "collapse-icon";
      icon.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>`;
      h2.prepend(icon);

      h2.onclick = () => {
        const isCollapsed = h2.getAttribute("data-collapsed") === "true";
        h2.setAttribute("data-collapsed", !isCollapsed);
        wrapper.style.display = isCollapsed ? "block" : "none";
        icon.style.transform = isCollapsed ? "rotate(0deg)" : "rotate(-90deg)";
      };
    }
  });
}

function toggleAllSections(expand) {
  const headings = document.querySelectorAll(".collapsible-heading");
  headings.forEach(h => {
    const isCollapsed = h.getAttribute("data-collapsed") === "true";
    if (expand && isCollapsed) h.click();
    else if (!expand && !isCollapsed) h.click();
  });
}
function copyReport(btnEl, reportId) {
  const markdown = reportStore[reportId] || "";
  navigator.clipboard.writeText(markdown).then(() => {
    const originalHTML = btnEl.innerHTML;
    btnEl.innerHTML = '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    setTimeout(() => (btnEl.innerHTML = originalHTML), 1500);
  });
}
function downloadReport(reportId) {
  const markdown = reportStore[reportId] || "";
  const blob = new Blob([markdown], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `legal_comparison_${Date.now()}.md`;
  a.click();
  URL.revokeObjectURL(url);
}
function applyFilter(filter) {
  currentFilter = filter;
  document.querySelectorAll(".filter-chip").forEach(c => {
    c.classList.toggle("active", c.dataset.filter === filter);
  });
  document.querySelectorAll(".stat-card").forEach(c => {
    const t = c.dataset.filterType;
    c.classList.toggle("stat-active", t === filter);
  });

  document.querySelectorAll(".citation-chip").forEach(chip => {
    const key = chip.dataset.key;
    const data = citationStore[key];
    if (!data) return;
    const visible = filter === "all" || data.type === filter;
    chip.style.display = visible ? "" : "none";
  });

  const mdContents = document.querySelectorAll(".md-content");
  mdContents.forEach(content => {
    const h3List = content.querySelectorAll("h3");
    h3List.forEach(h3 => {
      const chip = h3.querySelector(".citation-chip");
      if (!chip) return;
      const key = chip.dataset.key;
      const data = citationStore[key];
      if (!data) return;

      const matchesFilter = filter === "all" || data.type === filter;
      let el = h3;
      while (el) {
        el.style.display = matchesFilter ? "" : "none";
        el = el.nextElementSibling;
        if (el && (el.tagName === "H3" || el.tagName === "H2")) break;
      }
    });
  });
}

function showError(message) {
  const wrap = document.createElement("div");
  wrap.className = "msg";
  wrap.innerHTML = `
    <div class="error-msg">
      <span class="error-icon">⚠</span>
      <div><strong>Error:</strong><br>${escapeHtml(message)}</div>
    </div>
  `;
  appendMessage(wrap);
}
function handleSSEEvent(event, payload) {
  switch (event) {
    case "previews":
      previewUrls.a = payload.url_a;
      previewUrls.b = payload.url_b;
      // Update tab labels with real filenames
      document.getElementById("tab-v1").textContent = payload.name_a;
      document.getElementById("tab-v2").textContent = payload.name_b;
      break;
    case "progress":
      updateStep(payload.step, payload.status, payload.detail);
      if (payload.status === "running") {
        document.getElementById("status-text").textContent = payload.title || "Processing";
        document.getElementById("status-dot").style.background = "var(--accent)";
      }
      break;
    case "stats":
      showStatistics(payload);
      break;
    case "citations":
      storeCitations(payload.items || []);
      break;
    case "report":
      showReport(payload.markdown);
      document.getElementById("status-text").textContent = "Complete";
      document.getElementById("status-dot").style.background = "var(--green)";
      break;
    case "done":
      document.getElementById("btn-reset").style.display = "block";
      break;
    case "error":
      showError(payload.message);
      document.getElementById("btn-reset").style.display = "block";
      document.getElementById("status-text").textContent = "Error";
      document.getElementById("status-dot").style.background = "var(--red)";
      break;
  }
}
async function startComparison() {
  if (!fileA || !fileB) return;

  hideEmptyState();
  closeSourcePanel();

  // Clear old citations
  Object.keys(citationStore).forEach(k => delete citationStore[k]);

  // Update UI state
  const btn = document.getElementById("btn-compare");
  btn.disabled = true;
  document.getElementById("btn-text").textContent = "Processing...";
  document.getElementById("status-text").textContent = "Processing";
  document.getElementById("status-dot").style.background = "var(--yellow)";

  // Create and display progress card
  const progressCard = createProgressCard(fileA.name, fileB.name);
  appendMessage(progressCard);

  // Prepare form data
  const formData = new FormData();
  formData.append("file_a", fileA);
  formData.append("file_b", fileB);

  try {
    // Make API request
    const response = await fetch(API_ENDPOINT, {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Unknown error" }));
      showError(error.detail || "Server error");
      return;
    }

    // Process SSE stream
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      // Parse SSE frames
      const frames = buffer.split("\n\n");
      buffer = frames.pop() || "";

      for (const frame of frames) {
        if (!frame.trim()) continue;

        const lines = frame.split("\n");
        let event = "message";
        let data = "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            event = line.slice(7).trim();
          } else if (line.startsWith("data: ")) {
            data += (data ? "\n" : "") + line.slice(6).trim();
          }
        }

        if (!data) continue;

        try {
          const payload = JSON.parse(data);
          handleSSEEvent(event, payload);
        } catch (e) {
          console.error("Failed to parse SSE data:", e);
        }
      }
    }
  } catch (error) {
    showError(`Connection failed: ${error.message}`);
  } finally {
    // Reset button state
    btn.disabled = false;
    document.getElementById("btn-text").textContent = "So sánh văn bản";
  }
}

// ── Reset Functionality ───────────────────────────────────────────────────────

/**
 * Reset application to initial state
 */
function resetComparison() {
  // Clear file state
  fileA = null;
  fileB = null;
  currentFilter = "all";
  previewUrls = { a: null, b: null };

  // Restore tab placeholder labels
  document.getElementById("tab-v1").textContent = "File A";
  document.getElementById("tab-v2").textContent = "File B";

  // Clear PDF frames
  const frameA = document.getElementById("pdf-frame-a");
  const frameB = document.getElementById("pdf-frame-b");
  if (frameA) frameA.src = "";
  if (frameB) frameB.src = "";

  // Clear citations
  Object.keys(citationStore).forEach(k => delete citationStore[k]);

  // Close source panel
  closeSourcePanel();

  // Hide filter section
  document.getElementById("filter-section").style.display = "none";

  // Reset file slots
  const slots = [
    ["slot-a", "name-a", "check-a", "hint-a", "input-a"],
    ["slot-b", "name-b", "check-b", "hint-b", "input-b"],
  ];

  for (const [slot, name, check, hint, input] of slots) {
    document.getElementById(slot).classList.remove("has-file");
    document.getElementById(name).style.display = "none";
    document.getElementById(check).style.display = "none";
    document.getElementById(hint).style.display = "block";
    document.getElementById(input).value = "";
  }

  // Restore empty state
  const chat = document.getElementById("chat");
  chat.innerHTML = `
    <div class="empty-state" id="empty-state">
      <h2>Hệ thống So sánh Văn bản Pháp lý</h2>
      <p>Tải lên hai phiên bản văn bản để hệ thống tự động phân tích và so sánh từng điều khoản bằng công nghệ AI.</p>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-title">Document Processing</div>
          <div class="feature-desc">Hỗ trợ PDF &amp; DOCX với phân tích cấu trúc tự động</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">Semantic Analysis</div>
          <div class="feature-desc">So sánh ngữ nghĩa sử dụng vector embedding</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">LLM Comparison</div>
          <div class="feature-desc">Phân tích điều khoản chi tiết bằng AI</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">Click Citations</div>
          <div class="feature-desc">Xem trực tiếp nội dung gốc bôi vàng khi click trích dẫn</div>
        </div>
      </div>
    </div>
  `;

  // Hide reset button and disable compare button
  document.getElementById("btn-reset").style.display = "none";
  document.getElementById("btn-compare").disabled = true;
  document.getElementById("status-text").textContent = "Ready";
  document.getElementById("status-dot").style.background = "var(--green)";
}

// ── Keyboard Navigation ───────────────────────────────────────────────────────
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    closeSourcePanel();
  }
});

// ── Drag & Drop Support ───────────────────────────────────────────────────────

(function initializeDragDrop() {
  const overlay = document.getElementById("drag-overlay");
  let dragTarget = null;

  document.addEventListener("dragenter", (e) => {
    if (e.dataTransfer.types.includes("Files")) {
      dragTarget = e.target;
      overlay.classList.add("active");
    }
  });

  document.addEventListener("dragleave", (e) => {
    if (e.target === dragTarget) {
      overlay.classList.remove("active");
    }
  });

  document.addEventListener("dragover", (e) => e.preventDefault());

  document.addEventListener("drop", (e) => {
    e.preventDefault();
    overlay.classList.remove("active");

    // Filter valid files
    const files = Array.from(e.dataTransfer.files).filter(
      (f) => f.name.endsWith(".pdf") || f.name.endsWith(".docx")
    );

    // Assign files to slots
    if (files[0]) {
      fileA = files[0];
      const nameEl = document.getElementById("name-a");
      nameEl.textContent = files[0].name;
      nameEl.style.display = "block";
      document.getElementById("hint-a").style.display = "none";
      document.getElementById("slot-a").classList.add("has-file");
      document.getElementById("check-a").style.display = "flex";
    }

    if (files[1]) {
      fileB = files[1];
      const nameEl = document.getElementById("name-b");
      nameEl.textContent = files[1].name;
      nameEl.style.display = "block";
      document.getElementById("hint-b").style.display = "none";
      document.getElementById("slot-b").classList.add("has-file");
      document.getElementById("check-b").style.display = "flex";
    }

    updateCompareButton();
  });
})();

// ── Initialize Markdown Parser ───────────────────────────────────────────────

marked.setOptions({
  breaks: true,
  gfm: true,
});
