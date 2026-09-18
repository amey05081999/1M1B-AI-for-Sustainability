/* ── Constants ─────────────────────────────────────────────────────────────── */
const API_BASE = "";   // same origin; change to http://localhost:8000 for dev

/* ── DOM refs ──────────────────────────────────────────────────────────────── */
const chatWindow      = document.getElementById("chatWindow");
const inputForm       = document.getElementById("inputForm");
const userInput       = document.getElementById("userInput");
const sendBtn         = document.getElementById("sendBtn");
const typingIndicator = document.getElementById("typingIndicator");
const charCount       = document.getElementById("charCount");
const sourcesInfo     = document.getElementById("sourcesInfo");
const topicList       = document.getElementById("topicList");
const resetBtn        = document.getElementById("resetBtn");
const statusDot       = document.getElementById("statusDot");
const menuToggle      = document.getElementById("menuToggle");
const sidebar         = document.getElementById("sidebar");

/* ── State ─────────────────────────────────────────────────────────────────── */
let isLoading = false;

/* ── Init ──────────────────────────────────────────────────────────────────── */
window.addEventListener("DOMContentLoaded", () => {
  checkHealth();
  loadTopics();
  autoResizeTextarea();
  userInput.focus();
});

/* ── Health check ──────────────────────────────────────────────────────────── */
async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (res.ok) {
      statusDot.classList.add("ok");
      statusDot.title = "API online";
    } else {
      statusDot.classList.add("error");
      statusDot.title = "API error";
    }
  } catch {
    statusDot.classList.add("error");
    statusDot.title = "API unreachable";
  }
}

/* ── Load topics into sidebar ──────────────────────────────────────────────── */
async function loadTopics() {
  try {
    const res = await fetch(`${API_BASE}/api/topics`);
    if (!res.ok) return;
    const topics = await res.json();
    topicList.innerHTML = "";
    topics.forEach(({ topic }) => {
      const li = document.createElement("li");
      li.textContent = topic;
      topicList.appendChild(li);
    });
  } catch {
    topicList.innerHTML = '<li class="topic-loading">Could not load topics</li>';
  }
}

/* ── Textarea auto-resize ──────────────────────────────────────────────────── */
function autoResizeTextarea() {
  userInput.addEventListener("input", () => {
    userInput.style.height = "auto";
    userInput.style.height = Math.min(userInput.scrollHeight, 120) + "px";
    charCount.textContent = `${userInput.value.length} / 2000`;
  });
}

/* ── Send message ──────────────────────────────────────────────────────────── */
inputForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const text = userInput.value.trim();
  if (!text || isLoading) return;
  sendMessage(text);
});

userInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    inputForm.dispatchEvent(new Event("submit"));
  }
});

async function sendMessage(text) {
  isLoading = true;
  sendBtn.disabled = true;
  sourcesInfo.textContent = "";

  // Append user bubble
  appendBubble("user", text);
  userInput.value = "";
  userInput.style.height = "auto";
  charCount.textContent = "0 / 2000";

  // Show typing indicator
  typingIndicator.classList.remove("hidden");
  scrollToBottom();

  try {
    const res = await fetch(`${API_BASE}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text }),
    });

    typingIndicator.classList.add("hidden");

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: "Unknown error" }));
      appendBubble("assistant", `⚠️ Error: ${err.detail || "Something went wrong."}`, []);
      return;
    }

    const data = await res.json();
    appendBubble("assistant", data.answer, data.sources, data.response_time_ms);

    if (data.sources && data.sources.length > 0) {
      sourcesInfo.textContent = `📚 ${data.sources.length} source(s) used · ${data.response_time_ms}ms`;
    }
  } catch (err) {
    typingIndicator.classList.add("hidden");
    appendBubble("assistant", "⚠️ Could not reach the API. Make sure the server is running on port 8000.", []);
  } finally {
    isLoading = false;
    sendBtn.disabled = false;
    userInput.focus();
    scrollToBottom();
  }
}

/* ── Append a chat bubble ──────────────────────────────────────────────────── */
function appendBubble(role, rawText, sources = [], ms = null) {
  const row = document.createElement("div");
  row.className = `msg-row ${role}`;

  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "👤" : "🌿";

  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.innerHTML = renderMarkdown(rawText);

  // Attach sources tag for assistant
  if (role === "assistant" && sources && sources.length > 0) {
    const tag = document.createElement("div");
    tag.className = "sources-tag";
    tag.innerHTML = "📚 Sources: " + sources.map(s =>
      `<span>${s.replace(".txt", "").replace(/_/g, " ")}</span>`
    ).join("");
    bubble.appendChild(tag);
  }

  row.appendChild(avatar);
  row.appendChild(bubble);
  chatWindow.appendChild(row);
  scrollToBottom();
}

/* ── Lightweight Markdown renderer ────────────────────────────────────────── */
function renderMarkdown(text) {
  // Escape HTML first
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Bold: **text**
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic: *text* or _text_
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");

  // Inline code: `code`
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Headers: ### or ##
  html = html.replace(/^### (.+)$/gm, "<h3>$1</h3>");
  html = html.replace(/^## (.+)$/gm, "<h3>$1</h3>");

  // Unordered lists: lines starting with - or *
  html = html.replace(/^[*\-] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>(\n|$))+/g, (match) => `<ul>${match}</ul>`);

  // Numbered lists
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Horizontal rule
  html = html.replace(/^---+$/gm, "<hr/>");

  // Paragraphs: double newline → <p>
  html = html
    .split(/\n{2,}/)
    .map(block => {
      block = block.trim();
      if (!block) return "";
      if (/^<(h[1-6]|ul|ol|li|hr)/.test(block)) return block;
      return `<p>${block.replace(/\n/g, "<br/>")}</p>`;
    })
    .join("\n");

  return html;
}

/* ── Scroll ────────────────────────────────────────────────────────────────── */
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatWindow.scrollTop = chatWindow.scrollHeight;
  });
}

/* ── Quick question buttons ────────────────────────────────────────────────── */
document.querySelectorAll(".quick-btn").forEach(btn => {
  btn.addEventListener("click", () => {
    const q = btn.dataset.q;
    if (q && !isLoading) {
      closeSidebar();
      sendMessage(q);
    }
  });
});

/* ── Reset conversation ────────────────────────────────────────────────────── */
resetBtn.addEventListener("click", async () => {
  try {
    await fetch(`${API_BASE}/api/reset`, { method: "POST" });
  } catch { /* ignore */ }

  // Clear chat UI
  chatWindow.innerHTML = "";
  sourcesInfo.textContent = "";

  // Re-add welcome message
  appendBubble("assistant",
    "Conversation reset! 🌿 I'm ready for your next sustainability question. What would you like to explore?"
  );
  closeSidebar();
});

/* ── Mobile sidebar toggle ─────────────────────────────────────────────────── */
menuToggle.addEventListener("click", () => {
  sidebar.classList.toggle("open");
});

function closeSidebar() {
  sidebar.classList.remove("open");
}

// Close sidebar when clicking outside on mobile
document.addEventListener("click", (e) => {
  if (window.innerWidth <= 680 &&
      sidebar.classList.contains("open") &&
      !sidebar.contains(e.target) &&
      e.target !== menuToggle) {
    closeSidebar();
  }
});
