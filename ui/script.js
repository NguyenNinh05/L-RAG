let fileA = null;
let fileB = null;
let currentFilter = "all";
let currentSessionId = null;
let currentSummaryCounts = {};
let citationStore = {};
let analysisStore = null;
let reportStore = {};
let reportCounter = 0;
let currentReportId = null;
let currentSourceMode = "report";
let currentFileNames = { a: null, b: null };
let activeAssistantStream = null;
let isComparing = false;

const API_COMPARE = "/api/compare";
const API_SESSIONS = "/api/sessions";
const STEP_DEFINITIONS = [
  { title: "Phân tích cấu trúc tài liệu" },
  { title: "Tạo vector embedding" },
  { title: "Đối sánh ngữ nghĩa" },
  { title: "Phân tích bằng LLM" },
  { title: "Lưu phiên làm việc" },
];

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function sanitizeStem(value) {
  return String(value || "")
    .replace(/\.[^.]+$/, "")
    .replace(/[^a-zA-Z0-9._-]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 40) || "document";
}

function setCurrentFileNames(nameA = null, nameB = null) {
  currentFileNames = { a: nameA || null, b: nameB || null };
}

function getResolvedFileNames() {
  return {
    a: currentFileNames.a || fileA?.name || null,
    b: currentFileNames.b || fileB?.name || null,
  };
}

function buildReportFileName(nameA = null, nameB = null) {
  const resolved = {
    a: nameA || currentFileNames.a || fileA?.name || "doc_a",
    b: nameB || currentFileNames.b || fileB?.name || "doc_b",
  };
  return `bao_cao_so_sanh_${sanitizeStem(resolved.a)}_vs_${sanitizeStem(resolved.b)}.md`;
}

function formatCounts(summary = {}) {
  if (!summary || Object.keys(summary).length === 0) {
    return "Tải lên hai tài liệu để bắt đầu so sánh.";
  }
  return `${summary.atomic_changes || 0} thay đổi · ${summary.substantive || 0} thực chất · ${summary.formal || 0} hình thức`;
}

function hideEmptyState() {
  const empty = document.getElementById("empty-state");
  if (empty) empty.remove();
}

function emptyStateMarkup() {
  return `
    <div class="empty-state" id="empty-state">
      <h2>Trợ lý So sánh Văn bản Pháp lý</h2>
      <p>Hệ thống giữ nguyên luồng so sánh hiện tại, sau đó cho phép hỏi đáp trực tiếp trên phiên đã hoàn tất.</p>
      <div class="feature-grid">
        <div class="feature-card">
          <div class="feature-title">So sánh có dẫn chứng</div>
          <div class="feature-desc">Báo cáo và trích dẫn được khóa bởi lớp deterministic.</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">Lịch sử phiên</div>
          <div class="feature-desc">Mở lại báo cáo, trích dẫn và lịch sử chat ngay trên ứng dụng cục bộ.</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">Hỏi đáp sau so sánh</div>
          <div class="feature-desc">Chỉ bật hỏi đáp sau khi phiên so sánh hoàn tất.</div>
        </div>
        <div class="feature-card">
          <div class="feature-title">Trích dẫn ổn định</div>
          <div class="feature-desc">Bấm trích dẫn để mở lại đúng panel bằng chứng sau khi tải lại trang.</div>
        </div>
      </div>
    </div>
  `;
}

function clearChatArea() {
  document.getElementById("chat").innerHTML = "";
}

function appendMessage(element) {
  const chat = document.getElementById("chat");
  chat.appendChild(element);
  element.scrollIntoView({ behavior: "smooth", block: "end" });
}

function setStatus(text, color) {
  document.getElementById("status-text").textContent = text;
  document.getElementById("status-dot").style.background = color;
}

function updateTopbar(fileAName = null, fileBName = null, summary = {}) {
  const title = document.getElementById("topbar-title");
  const subtitle = document.getElementById("topbar-subtitle");
  if (fileAName && fileBName) {
    title.textContent = `So sánh ${fileAName} và ${fileBName}`;
    subtitle.textContent = formatCounts(summary);
    return;
  }
  title.textContent = "Phiên so sánh mới";
  subtitle.textContent = "Tải lên hai tài liệu để bắt đầu so sánh.";
}

function setChatAvailability(enabled, hintText) {
  const input = document.getElementById("chat-input");
  const send = document.getElementById("btn-send");
  const hint = document.getElementById("chat-input-hint");
  input.disabled = !enabled;
  send.disabled = !enabled;
  input.placeholder = enabled ? "Hỏi về các thay đổi trong phiên này..." : hintText;
  hint.textContent = hintText;
}

function updateCompareButton() {
  document.getElementById("btn-compare").disabled = !(fileA && fileB) || isComparing;
}

function setupFileInput(inputId, slotId, nameId, checkId, hintId, setter) {
  const input = document.getElementById(inputId);
  input.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (!file) return;
    const maxSize = 50 * 1024 * 1024;
    if (file.size > maxSize) {
      alert(`Tài liệu "${file.name}" vượt quá giới hạn 50MB.`);
      input.value = "";
      return;
    }
    setter(file);
    document.getElementById(nameId).textContent = file.name;
    document.getElementById(nameId).style.display = "block";
    document.getElementById(hintId).style.display = "none";
    document.getElementById(slotId).classList.add("has-file");
    document.getElementById(checkId).style.display = "flex";
    updateCompareButton();
  });
}

function setSlotFile(slotKey, file) {
  if (!file) return;
  const isA = slotKey === "A";
  if (isA) {
    fileA = file;
  } else {
    fileB = file;
  }
  document.getElementById(isA ? "name-a" : "name-b").textContent = file.name;
  document.getElementById(isA ? "name-a" : "name-b").style.display = "block";
  document.getElementById(isA ? "hint-a" : "hint-b").style.display = "none";
  document.getElementById(isA ? "slot-a" : "slot-b").classList.add("has-file");
  document.getElementById(isA ? "check-a" : "check-b").style.display = "flex";
  updateCompareButton();
}

function inferDropSlot(target) {
  const slot = target?.closest?.(".upload-slot");
  if (!slot) return null;
  if (slot.id === "slot-a") return "A";
  if (slot.id === "slot-b") return "B";
  return null;
}

function createProgressCard(nameA, nameB) {
  const card = document.createElement("div");
  card.className = "msg";
  card.id = "progress-card";
  card.innerHTML = `
    <div class="progress-card">
      <div class="progress-card-header">
        <div class="ai-badge">Đang xử lý</div>
        <div class="progress-info">
          <div class="progress-title">
            Tiến trình so sánh
            <span class="thinking-indicator"><span></span><span></span><span></span></span>
          </div>
          <div class="progress-subtitle">${escapeHtml(nameA)} → ${escapeHtml(nameB)}</div>
        </div>
      </div>
      <div class="progress-inline">
        <span class="progress-inline-label">Bước hiện tại</span>
        <div class="progress-inline-body">
          <div class="progress-inline-step" id="progress-current-step">Đang chờ bắt đầu xử lý...</div>
          <div class="progress-inline-detail" id="progress-current-detail">Đang chuẩn bị phiên so sánh cục bộ.</div>
        </div>
      </div>
    </div>
  `;
  return card;
}

function updateStep(stepNumber, status, detail) {
  const stepEl = document.getElementById("progress-current-step");
  const detailEl = document.getElementById("progress-current-detail");
  if (!stepEl || !detailEl) return;
  const stepTitle = STEP_DEFINITIONS[stepNumber - 1]?.title || `Bước ${stepNumber}`;
  stepEl.textContent = stepTitle;
  detailEl.textContent = detail || (status === "done" ? "Đã hoàn tất." : "Đang chạy...");
  stepEl.dataset.status = status;
}

function showStatistics(stats) {
  currentSummaryCounts = stats;
  const names = getResolvedFileNames();
  updateTopbar(names.a, names.b, stats);
  const card = document.getElementById("progress-card")?.querySelector(".progress-card");
  if (!card) return;

  const badge = card.querySelector(".ai-badge");
  if (badge) {
    badge.textContent = "Hoàn tất";
    badge.style.background = "rgba(34, 197, 94, 0.1)";
    badge.style.borderColor = "var(--green)";
    badge.style.color = "var(--green)";
  }

  const dots = card.querySelector(".thinking-indicator");
  if (dots) dots.remove();
  const currentStep = document.getElementById("progress-current-step");
  const currentDetail = document.getElementById("progress-current-detail");
  if (currentStep) {
    currentStep.textContent = "So sánh hoàn tất";
    currentStep.dataset.status = "done";
  }
  if (currentDetail) {
    currentDetail.textContent = formatCounts(stats);
  }
}

function getCitationKey(item) {
  return item.citation_id || `${item.citation_type || item.change_kind || "citation"}::${item.clause_id || item.id}`;
}

function storeCitations(items) {
  (items || []).forEach((item) => {
    const normalized = { ...item };
    normalized.citation_id = getCitationKey(item);
    citationStore[normalized.citation_id] = normalized;
  });
}

function findClauseCitationKey(text) {
  return (
    Object.keys(citationStore).find((key) => {
      const item = citationStore[key];
      return item.citation_type === "clause" && item.clause_id && text.includes(item.clause_id);
    }) || null
  );
}

function highlightText(htmlText) {
  return `<mark class="source-highlight">${htmlText}</mark>`;
}

function setSourceHeader(label, badgeText, badgeType = "report") {
  document.getElementById("source-panel-label").textContent = label;
  const badge = document.getElementById("source-badge");
  badge.textContent = badgeText;
  badge.className = `source-badge type-${String(badgeType).toLowerCase()}`;
}

function switchSourceMode(mode) {
  currentSourceMode = mode === "evidence" ? "evidence" : "report";
  const panel = document.getElementById("source-panel");
  panel.dataset.mode = currentSourceMode;
  document.getElementById("source-tab-report").classList.toggle("active", currentSourceMode === "report");
  document.getElementById("source-tab-evidence").classList.toggle("active", currentSourceMode === "evidence");
}

function clearEvidenceView() {
  document.getElementById("source-doc-name-a").textContent = "";
  document.getElementById("source-doc-name-b").textContent = "";
  document.getElementById("source-page-a").textContent = "";
  document.getElementById("source-page-b").textContent = "";
  document.getElementById("source-text-a").innerHTML = highlightText(escapeHtml("Chọn trích dẫn để xem bản cũ."));
  document.getElementById("source-text-b").innerHTML = highlightText(escapeHtml("Chọn trích dẫn để xem bản mới."));
}

function clearDrawerReport() {
  document.getElementById("drawer-report-empty").style.display = "block";
  document.getElementById("drawer-report-content").innerHTML = "";
  document.getElementById("drawer-report-filename").textContent = "bao_cao_so_sanh.md";
  document.getElementById("drawer-report-subtitle").textContent = "Mở tệp báo cáo để xem nội dung.";
}

function openDrawer() {
  document.getElementById("source-panel").classList.add("open");
  document.getElementById("content-area").classList.add("panel-open");
}

function closeSourcePanel() {
  document.getElementById("source-panel").classList.remove("open");
  document.getElementById("content-area").classList.remove("panel-open");
  document.querySelectorAll(".citation-chip").forEach((chip) => chip.classList.remove("active"));
}

function openCitationPanel(citationId) {
  const data = citationStore[citationId];
  if (!data) return;

  switchSourceMode("evidence");
  setSourceHeader(
    `Điều khoản: ${data.clause_id || "Nguồn trích"}`,
    data.change_kind || data.citation_type || "TRÍCH DẪN",
    data.change_kind || data.citation_type || "citation",
  );

  document.getElementById("source-doc-name-a").textContent = data.filename_a || "Văn bản cũ";
  document.getElementById("source-doc-name-b").textContent = data.filename_b || "Văn bản mới";
  document.getElementById("source-page-a").textContent = data.page_a
    ? `Trang ${data.page_a}${data.chunk_id_a ? ` · ${data.chunk_id_a}` : ""}`
    : data.chunk_id_a || "";
  document.getElementById("source-page-b").textContent = data.page_b
    ? `Trang ${data.page_b}${data.chunk_id_b ? ` · ${data.chunk_id_b}` : ""}`
    : data.chunk_id_b || "";

  const oldText = data.text_a || data.excerpt_a || "(Không có ở bản cũ)";
  const newText = data.text_b || data.excerpt_b || (data.text_a ? "(Không có ở bản mới)" : data.excerpt || "(Không có dữ liệu)");
  document.getElementById("source-text-a").innerHTML = highlightText(escapeHtml(oldText));
  document.getElementById("source-text-b").innerHTML = highlightText(escapeHtml(newText));

  openDrawer();
  document.querySelectorAll(".citation-chip").forEach((chip) => chip.classList.remove("active"));
  const chip = document.querySelector(`.citation-chip[data-key="${CSS.escape(citationId)}"]`);
  if (chip) chip.classList.add("active");
}

function injectCitationChips(container) {
  container.querySelectorAll("h3").forEach((heading) => {
    const key = findClauseCitationKey(heading.textContent.trim());
    if (!key || heading.querySelector(".citation-chip")) return;
    const chip = document.createElement("button");
    chip.className = "citation-chip";
    chip.dataset.key = key;
    chip.textContent = "Xem nguồn";
    chip.onclick = () => openCitationPanel(key);
    heading.appendChild(chip);
  });
}

function postProcessMarkdown(container) {
  const badgePattern = /\[(THỰC CHẤT|HÌNH THỨC|THAY ĐỔI TỪ NGỮ|KHÔNG ĐỔI NGỮ NGHĨA|Sửa đổi|Thay thế|Bổ sung|Loại bỏ|Số liệu|Thời hạn|Địa điểm|Nhân sự|Chủ thể|Chính tả|Định dạng|Ngôn ngữ chuyên môn|Cấu trúc danh sách|Khoảng trắng)\]/g;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, null, false);
  const nodesToReplace = [];
  let node;

  while ((node = walker.nextNode())) {
    if (node.textContent.includes("[") && node.textContent.includes("]")) {
      nodesToReplace.push(node);
    }
  }

  nodesToReplace.forEach((textNode) => {
    const parent = textNode.parentNode;
    if (!parent) return;
    const html = textNode.textContent.replace(badgePattern, (_, tag) => {
      const className = tag === "THỰC CHẤT" ? "badge-substantial" : tag === "HÌNH THỨC" ? "badge-formal" : "badge-detail";
      return `<span class="inline-badge ${className}">${tag}</span>`;
    });
    if (html === textNode.textContent) return;
    const span = document.createElement("span");
    span.innerHTML = html;
    parent.replaceChild(span, textNode);
  });
}

function makeReportCollapsible(container) {
  container.querySelectorAll("h2").forEach((heading) => {
    heading.classList.add("collapsible-heading");
    heading.setAttribute("data-collapsed", "false");

    const siblings = [];
    let cursor = heading.nextElementSibling;
    while (cursor && cursor.tagName !== "H2") {
      siblings.push(cursor);
      cursor = cursor.nextElementSibling;
    }
    if (!siblings.length) return;

    const wrapper = document.createElement("div");
    wrapper.className = "collapsible-body";
    siblings[0].parentNode.insertBefore(wrapper, siblings[0]);
    siblings.forEach((sibling) => wrapper.appendChild(sibling));

    const icon = document.createElement("span");
    icon.className = "collapse-icon";
    icon.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>';
    heading.prepend(icon);

    heading.onclick = () => {
      const collapsed = heading.getAttribute("data-collapsed") === "true";
      heading.setAttribute("data-collapsed", String(!collapsed));
      wrapper.style.display = collapsed ? "block" : "none";
      icon.style.transform = collapsed ? "rotate(0deg)" : "rotate(-90deg)";
    };
  });
}

function applyFilterToReportContent() {
  const content = document.getElementById("drawer-report-content");
  if (!content || !content.children.length) return;

  content.querySelectorAll("h3").forEach((heading) => {
    const chip = heading.querySelector(".citation-chip");
    if (!chip) return;
    const citation = citationStore[chip.dataset.key];
    const visible = currentFilter === "all" || citation?.change_kind === currentFilter;

    let element = heading;
    while (element) {
      element.style.display = visible ? "" : "none";
      element = element.nextElementSibling;
      if (element && (element.tagName === "H3" || element.tagName === "H2")) break;
    }
  });
}

function getReportEntry(reportId) {
  return reportStore[reportId] || null;
}

function registerReport(markdown, metadata = {}) {
  const reportId = ++reportCounter;
  reportStore[reportId] = {
    markdown,
    fileName: metadata.fileName || buildReportFileName(),
    subtitle: metadata.subtitle || formatCounts(currentSummaryCounts),
  };
  currentReportId = reportId;
  return reportId;
}

function renderReportIntoDrawer(reportId) {
  const report = getReportEntry(reportId);
  if (!report) return;

  const content = document.getElementById("drawer-report-content");
  document.getElementById("drawer-report-empty").style.display = "none";
  document.getElementById("drawer-report-filename").textContent = report.fileName;
  document.getElementById("drawer-report-subtitle").textContent = report.subtitle;
  content.innerHTML = DOMPurify.sanitize(marked.parse(report.markdown || ""));
  postProcessMarkdown(content);
  makeReportCollapsible(content);
  injectCitationChips(content);
  applyFilterToReportContent();
}

function openReportPanel(reportId) {
  const report = getReportEntry(reportId);
  if (!report) return;
  currentReportId = reportId;
  switchSourceMode("report");
  setSourceHeader(report.fileName, "BÁO CÁO", "report");
  renderReportIntoDrawer(reportId);
  openDrawer();
}

function copyWithFeedback(button, text) {
  if (!text) return;
  navigator.clipboard.writeText(text).then(() => {
    if (!button) return;
    const original = button.innerHTML;
    button.innerHTML = "✓";
    setTimeout(() => {
      button.innerHTML = original;
    }, 1000);
  });
}

function copyReport(button, reportId) {
  const report = getReportEntry(reportId);
  if (!report) return;
  copyWithFeedback(button, report.markdown);
}

function copyActiveReport(button) {
  if (!currentReportId) return;
  copyReport(button, currentReportId);
}

function downloadReport(reportId) {
  const report = getReportEntry(reportId);
  if (!report) return;
  const blob = new Blob([report.markdown || ""], { type: "text/markdown" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = report.fileName;
  link.click();
  URL.revokeObjectURL(url);
}

function downloadActiveReport() {
  if (!currentReportId) return;
  downloadReport(currentReportId);
}

function showReportArtifact(markdown, metadata = {}) {
  const reportId = registerReport(markdown, metadata);
  const report = getReportEntry(reportId);
  const wrapper = document.createElement("div");
  wrapper.className = "msg report-artifact-wrapper";
  wrapper.innerHTML = `
    <div class="report-artifact-card" onclick="openReportPanel(${reportId})">
      <div class="report-artifact-icon">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
          <polyline points="14 2 14 8 20 8"></polyline>
        </svg>
      </div>
      <div class="report-artifact-meta">
        <div class="report-artifact-name">${escapeHtml(report.fileName)}</div>
        <div class="report-artifact-subtitle">${escapeHtml(report.subtitle)}</div>
      </div>
      <div class="report-artifact-actions">
        <button class="btn-artifact-action" onclick="event.stopPropagation(); openReportPanel(${reportId})">Mở</button>
        <button class="btn-artifact-action" onclick="event.stopPropagation(); copyReport(this, ${reportId})">Sao chép</button>
        <button class="btn-artifact-action" onclick="event.stopPropagation(); downloadReport(${reportId})">Tải xuống</button>
      </div>
    </div>
  `;
  appendMessage(wrapper);
  document.getElementById("filter-section").style.display = "block";
}

function toggleAllSections(expand) {
  const content = document.getElementById("drawer-report-content");
  if (!content) return;
  content.querySelectorAll(".collapsible-heading").forEach((heading) => {
    const collapsed = heading.getAttribute("data-collapsed") === "true";
    if ((expand && collapsed) || (!expand && !collapsed)) {
      heading.click();
    }
  });
}

function showError(message) {
  const wrapper = document.createElement("div");
  wrapper.className = "msg";
  wrapper.innerHTML = `
    <div class="error-msg">
      <span class="error-icon">!</span>
      <div><strong>Lỗi:</strong><br>${escapeHtml(message)}</div>
    </div>
  `;
  appendMessage(wrapper);
}

function storeAnalysis(payload) {
  analysisStore = payload;
}

function applyFilter(filter) {
  currentFilter = filter;
  document.querySelectorAll(".filter-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.filter === filter);
  });
  document.querySelectorAll(".stat-card").forEach((card) => {
    card.classList.toggle("stat-active", card.dataset.filterType === filter);
  });
  applyFilterToReportContent();
}

function renderCitationChips(citationIds) {
  if (!citationIds || !citationIds.length) return "";
  return citationIds
    .filter((citationId) => citationStore[citationId])
    .map((citationId) => {
      const citation = citationStore[citationId];
      const label = escapeHtml(citation.clause_id || citation.citation_type || "Nguồn trích");
      return `<button class="citation-chip" data-key="${citationId}" onclick="openCitationPanel('${citationId}')">${label}</button>`;
    })
    .join("");
}

function beginAssistantStream() {
  if (activeAssistantStream?.wrapper?.isConnected) return activeAssistantStream;
  const wrapper = document.createElement("div");
  wrapper.className = "msg chat-message assistant";
  wrapper.innerHTML = `
    <div class="chat-card assistant">
      <div class="chat-role">Hệ thống</div>
      <div class="chat-body chat-body-streaming"></div>
    </div>
  `;
  appendMessage(wrapper);
  activeAssistantStream = {
    wrapper,
    body: wrapper.querySelector(".chat-body"),
    content: "",
  };
  return activeAssistantStream;
}

function appendAssistantStreamDelta(chunk) {
  if (!chunk) return;
  const stream = beginAssistantStream();
  stream.content += chunk;
  stream.body.textContent = stream.content;
}

function finalizeAssistantStream(content, citationIds = []) {
  const stream = beginAssistantStream();
  const body = DOMPurify.sanitize(marked.parse(content || ""));
  stream.wrapper.innerHTML = `
    <div class="chat-card assistant">
      <div class="chat-role">Hệ thống</div>
      <div class="chat-body">${body}</div>
      ${citationIds.length ? `<div class="message-citations">${renderCitationChips(citationIds)}</div>` : ""}
    </div>
  `;
  activeAssistantStream = null;
}

function clearAssistantStream() {
  if (activeAssistantStream?.wrapper?.isConnected) {
    activeAssistantStream.wrapper.remove();
  }
  activeAssistantStream = null;
}

function renderChatMessage(role, content, citationIds = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg chat-message ${role}`;
  const body =
    role === "assistant" ? DOMPurify.sanitize(marked.parse(content || "")) : `<p>${escapeHtml(content || "")}</p>`;
  wrapper.innerHTML = `
    <div class="chat-card ${role}">
      <div class="chat-role">${role === "assistant" ? "Hệ thống" : "Bạn"}</div>
      <div class="chat-body">${body}</div>
      ${citationIds.length ? `<div class="message-citations">${renderCitationChips(citationIds)}</div>` : ""}
    </div>
  `;
  appendMessage(wrapper);
}

function hydrateCitationsFromAnswer(answer) {
  const missing = (answer.used_citation_ids || []).filter((citationId) => !citationStore[citationId]);
  if (!missing.length) return Promise.resolve();
  return Promise.all(
    missing.map((citationId) =>
      fetch(`${API_SESSIONS}/${currentSessionId}/citations/${citationId}`)
        .then((response) => (response.ok ? response.json() : null))
        .then((citation) => {
          if (citation) storeCitations([citation]);
      }),
    ),
  );
}

function renderHistoryList(items) {
  const list = document.getElementById("history-list");
  const empty = document.getElementById("history-empty");
  list.innerHTML = "";

  if (!items.length) {
    list.appendChild(empty);
    empty.style.display = "block";
    return;
  }

  items.forEach((session) => {
    const item = document.createElement("button");
    item.className = `history-item ${session.session_id === currentSessionId ? "active" : ""}`;
    item.innerHTML = `
      <div class="history-item-top">
        <div class="history-title">${escapeHtml(session.file_a_name)} → ${escapeHtml(session.file_b_name)}</div>
        <span class="history-delete" title="Xóa phiên" onclick="event.stopPropagation(); deleteSession('${session.session_id}')">×</span>
      </div>
      <div class="history-meta">${formatCounts(session.summary_counts || {})}</div>
    `;
    item.onclick = () => loadSession(session.session_id);
    list.appendChild(item);
  });
}

async function loadHistory() {
  const response = await fetch(API_SESSIONS);
  if (!response.ok) return;
  const payload = await response.json();
  renderHistoryList(payload.items || []);
}

async function loadSession(sessionId) {
  const response = await fetch(`${API_SESSIONS}/${sessionId}`);
  if (!response.ok) {
    showError("Không thể tải phiên đã lưu.");
    return;
  }

  const session = await response.json();
  currentSessionId = session.session_id;
  currentSummaryCounts = session.summary_counts || {};
  setCurrentFileNames(session.file_a_name, session.file_b_name);
  citationStore = {};
  reportStore = {};
  reportCounter = 0;
  currentReportId = null;
  storeCitations(session.citations || []);
  analysisStore = session.analysis || null;

  clearChatArea();
  hideEmptyState();
  clearAssistantStream();
  closeSourcePanel();
  clearDrawerReport();
  clearEvidenceView();

  showReportArtifact(session.report_markdown || "", {
    fileName: buildReportFileName(session.file_a_name, session.file_b_name),
    subtitle: formatCounts(currentSummaryCounts),
  });

  (session.messages || []).forEach((message) => {
    renderChatMessage(message.role, message.content, message.citation_ids || []);
  });

  updateTopbar(session.file_a_name, session.file_b_name, currentSummaryCounts);
  setStatus("Phiên đã hoàn tất", "var(--green)");
  setChatAvailability(true, "Hỏi về các thay đổi trong phiên này...");
  document.getElementById("btn-reset").style.display = "block";
  loadHistory();
}

async function deleteSession(sessionId) {
  await fetch(`${API_SESSIONS}/${sessionId}`, { method: "DELETE" });
  if (sessionId === currentSessionId) {
    resetComparison();
  }
  loadHistory();
}

async function clearAllSessions() {
  await fetch(API_SESSIONS, { method: "DELETE" });
  if (currentSessionId) resetComparison();
  loadHistory();
}

function resetComparison() {
  fileA = null;
  fileB = null;
  currentSessionId = null;
  currentFilter = "all";
  currentSummaryCounts = {};
  analysisStore = null;
  citationStore = {};
  reportStore = {};
  reportCounter = 0;
  currentReportId = null;
  setCurrentFileNames(null, null);

  clearChatArea();
  clearAssistantStream();
  document.getElementById("chat").innerHTML = emptyStateMarkup();
  updateTopbar();
  setStatus("Sẵn sàng", "var(--green)");
  setChatAvailability(false, "Vui lòng hoàn tất so sánh để bắt đầu hỏi đáp.");
  document.getElementById("filter-section").style.display = "none";
  document.getElementById("btn-reset").style.display = "none";

  document.querySelectorAll(".filter-chip").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.filter === "all");
  });

  [
    ["slot-a", "name-a", "check-a", "hint-a", "input-a"],
    ["slot-b", "name-b", "check-b", "hint-b", "input-b"],
  ].forEach(([slot, name, check, hint, input]) => {
    document.getElementById(slot).classList.remove("has-file");
    document.getElementById(name).style.display = "none";
    document.getElementById(check).style.display = "none";
    document.getElementById(hint).style.display = "block";
    document.getElementById(input).value = "";
  });

  clearDrawerReport();
  clearEvidenceView();
  switchSourceMode("report");
  closeSourcePanel();
  updateCompareButton();
  loadHistory();
}

async function hydrateCurrentSession() {
  if (!currentSessionId) return;
  const response = await fetch(`${API_SESSIONS}/${currentSessionId}`);
  if (!response.ok) return;
  const session = await response.json();
  storeCitations(session.citations || []);
}

function handleCompareEvent(event, payload) {
  switch (event) {
    case "session":
      currentSessionId = payload.session_id;
      break;
    case "previews":
      setCurrentFileNames(payload.name_a, payload.name_b);
      updateTopbar(payload.name_a, payload.name_b, currentSummaryCounts);
      break;
    case "progress":
      updateStep(payload.step, payload.status, payload.detail);
      if (payload.status === "running") {
        setStatus(payload.title || "Đang xử lý", "var(--yellow)");
      }
      break;
    case "stats":
      showStatistics(payload);
      break;
    case "analysis":
      storeAnalysis(payload);
      break;
    case "citations":
      storeCitations(payload.items || []);
      break;
    case "report":
      showReportArtifact(payload.markdown, {
        fileName: buildReportFileName(),
        subtitle: formatCounts(currentSummaryCounts),
      });
      setStatus("Hoàn tất", "var(--green)");
      break;
    case "done":
      isComparing = false;
      updateCompareButton();
      document.getElementById("btn-text").textContent = "So sánh văn bản";
      document.getElementById("btn-reset").style.display = "block";
      setChatAvailability(true, "Hỏi về các thay đổi trong phiên này...");
      hydrateCurrentSession();
      loadHistory();
      break;
    case "error":
      isComparing = false;
      updateCompareButton();
      document.getElementById("btn-text").textContent = "So sánh văn bản";
      showError(payload.message || "Đã xảy ra lỗi.");
      setStatus("Lỗi", "var(--red)");
      break;
    default:
      break;
  }
}

async function consumeSSE(response, handler) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const frames = buffer.split("\n\n");
    buffer = frames.pop() || "";

    for (const frame of frames) {
      if (!frame.trim()) continue;
      let event = "message";
      let data = "";
      frame.split("\n").forEach((line) => {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        if (line.startsWith("data: ")) data += (data ? "\n" : "") + line.slice(6).trim();
      });
      if (!data) continue;
      await handler(event, JSON.parse(data));
    }
  }
}

async function startComparison() {
  if (!fileA || !fileB || isComparing) return;

  hideEmptyState();
  clearChatArea();
  clearAssistantStream();
  closeSourcePanel();
  clearDrawerReport();
  clearEvidenceView();
  switchSourceMode("report");

  citationStore = {};
  analysisStore = null;
  reportStore = {};
  reportCounter = 0;
  currentReportId = null;
  currentSummaryCounts = {};
  currentSessionId = null;
  currentFilter = "all";
  setCurrentFileNames(fileA.name, fileB.name);

  isComparing = true;
  updateCompareButton();
  document.getElementById("btn-text").textContent = "Đang xử lý...";
  setStatus("Đang xử lý", "var(--yellow)");
  setChatAvailability(false, "Vui lòng hoàn tất so sánh để bắt đầu hỏi đáp.");
  appendMessage(createProgressCard(fileA.name, fileB.name));

  const formData = new FormData();
  formData.append("file_a", fileA);
  formData.append("file_b", fileB);

  try {
    const response = await fetch(API_COMPARE, { method: "POST", body: formData });
    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Lỗi máy chủ" }));
      showError(error.detail || "Lỗi máy chủ");
      isComparing = false;
      updateCompareButton();
      return;
    }
    await consumeSSE(response, handleCompareEvent);
  } catch (error) {
    isComparing = false;
    updateCompareButton();
    showError(`Kết nối thất bại: ${error.message}`);
    setStatus("Lỗi", "var(--red)");
  }
}

async function sendChatQuestion() {
  const input = document.getElementById("chat-input");
  const question = input.value.trim();
  if (!question || !currentSessionId) return;

  renderChatMessage("user", question);
  input.value = "";
  setChatAvailability(false, "Đang tạo câu trả lời có dẫn chứng...");
  clearAssistantStream();

  try {
    const response = await fetch(`${API_SESSIONS}/${currentSessionId}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (response.status === 409) {
      const payload = await response.json();
      showError(payload.detail);
      setChatAvailability(false, "Vui lòng hoàn tất so sánh để bắt đầu hỏi đáp.");
      return;
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: "Yêu cầu hỏi đáp thất bại" }));
      showError(error.detail || "Yêu cầu hỏi đáp thất bại");
      setChatAvailability(true, "Hỏi về các thay đổi trong phiên này...");
      return;
    }

    await consumeSSE(response, async (sseEvent, payload) => {
      if (sseEvent === "delta") {
        appendAssistantStreamDelta(payload.content || "");
      }
      if (sseEvent === "message") {
        await hydrateCitationsFromAnswer(payload);
        finalizeAssistantStream(payload.answer_markdown, payload.used_citation_ids || []);
      }
      if (sseEvent === "error") {
        clearAssistantStream();
        showError(payload.message);
      }
    });
  } catch (error) {
    clearAssistantStream();
    showError(`Hỏi đáp thất bại: ${error.message}`);
  } finally {
    setChatAvailability(
      true,
      currentSessionId ? "Hỏi về các thay đổi trong phiên này..." : "Vui lòng hoàn tất so sánh để bắt đầu hỏi đáp.",
    );
  }
}

function initializeDragDrop() {
  const overlay = document.getElementById("drag-overlay");
  let dragTarget = null;

  document.addEventListener("dragenter", (event) => {
    if (event.dataTransfer.types.includes("Files")) {
      dragTarget = event.target;
      overlay.classList.add("active");
    }
  });

  document.addEventListener("dragleave", (event) => {
    if (event.target === dragTarget) {
      overlay.classList.remove("active");
    }
  });

  document.addEventListener("dragover", (event) => event.preventDefault());
  document.addEventListener("drop", (event) => {
    event.preventDefault();
    overlay.classList.remove("active");

    const files = Array.from(event.dataTransfer.files).filter(
      (file) => file.name.endsWith(".pdf") || file.name.endsWith(".docx"),
    );
    if (!files.length) return;

    const slot = inferDropSlot(event.target);
    if (slot === "A") {
      setSlotFile("A", files[0]);
      if (files[1]) setSlotFile("B", files[1]);
      return;
    }
    if (slot === "B") {
      setSlotFile("B", files[0]);
      if (files[1]) setSlotFile("A", files[1]);
      return;
    }
    if (files.length === 1) {
      if (!fileA) setSlotFile("A", files[0]);
      else if (!fileB) setSlotFile("B", files[0]);
      else setSlotFile("A", files[0]);
      return;
    }
    setSlotFile("A", files[0]);
    setSlotFile("B", files[1]);
  });
}

function installSynchronousScrolling() {
  const sourceA = document.getElementById("source-text-a");
  const sourceB = document.getElementById("source-text-b");
  if (!sourceA || !sourceB) return;

  let syncingFrom = null;

  function syncScroll(master, slave, sourceKey) {
    const masterScrollable = master.scrollHeight - master.clientHeight;
    const slaveScrollable = slave.scrollHeight - slave.clientHeight;
    if (masterScrollable <= 0 || slaveScrollable <= 0) return;

    const percentage = master.scrollTop / masterScrollable;
    syncingFrom = sourceKey;
    slave.scrollTop = percentage * slaveScrollable;
  }

  sourceA.addEventListener(
    "scroll",
    () => {
      if (syncingFrom === "B") {
        syncingFrom = null;
        return;
      }
      syncScroll(sourceA, sourceB, "A");
    },
    { passive: true },
  );

  sourceB.addEventListener(
    "scroll",
    () => {
      if (syncingFrom === "A") {
        syncingFrom = null;
        return;
      }
      syncScroll(sourceB, sourceA, "B");
    },
    { passive: true },
  );
}

function installEventHandlers() {
  setupFileInput("input-a", "slot-a", "name-a", "check-a", "hint-a", (file) => {
    fileA = file;
    setCurrentFileNames(fileA?.name || null, fileB?.name || null);
    updateTopbar(fileA?.name || null, fileB?.name || null, currentSummaryCounts);
  });

  setupFileInput("input-b", "slot-b", "name-b", "check-b", "hint-b", (file) => {
    fileB = file;
    setCurrentFileNames(fileA?.name || null, fileB?.name || null);
    updateTopbar(fileA?.name || null, fileB?.name || null, currentSummaryCounts);
  });

  document.getElementById("btn-send").addEventListener("click", sendChatQuestion);
  document.getElementById("chat-input").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendChatQuestion();
    }
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSourcePanel();
  });

  window.addEventListener("beforeunload", (event) => {
    if (isComparing) {
      event.returnValue = "Phiên so sánh đang chạy. Nếu tải lại trang, stream hiện tại sẽ bị ngắt.";
      return event.returnValue;
    }
    return undefined;
  });
}

window.applyFilter = applyFilter;
window.clearAllSessions = clearAllSessions;
window.closeSourcePanel = closeSourcePanel;
window.copyActiveReport = copyActiveReport;
window.copyReport = copyReport;
window.deleteSession = deleteSession;
window.downloadActiveReport = downloadActiveReport;
window.downloadReport = downloadReport;
window.loadSession = loadSession;
window.openCitationPanel = openCitationPanel;
window.openReportPanel = openReportPanel;
window.resetComparison = resetComparison;
window.startComparison = startComparison;
window.switchSourceMode = switchSourceMode;
window.toggleAllSections = toggleAllSections;

marked.setOptions({ breaks: true, gfm: true });

document.addEventListener("DOMContentLoaded", () => {
  installEventHandlers();
  initializeDragDrop();
  installSynchronousScrolling();
  clearDrawerReport();
  clearEvidenceView();
  switchSourceMode("report");
  setChatAvailability(false, "Vui lòng hoàn tất so sánh để bắt đầu hỏi đáp.");
  updateTopbar();
  loadHistory();
});
