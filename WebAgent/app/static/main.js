/**
 * app/static/main.js
 * ==================
 * Handles all frontend behaviour for WebAgent:
 *   1. Health check on page load
 *   2. Character counter for the goal input
 *   3. SSE streaming from POST /research
 *   4. Progressive report rendering
 *   5. UI state transitions (idle → running → done / error)
 *   6. Copy report to clipboard
 */

/* ── 1. Element references ──────────────────────────────────────────────── */

const goalInput     = document.getElementById("goalInput");
const runBtn        = document.getElementById("runBtn");
const charCount     = document.getElementById("charCount");
const pulseRing     = document.getElementById("pulseRing");
const statusBar     = document.getElementById("statusBar");
const statusBarText = document.getElementById("statusBarText");
const progressFill  = document.getElementById("progressFill");
const reportSection = document.getElementById("reportSection");
const reportBody    = document.getElementById("reportBody");
const errorSection  = document.getElementById("errorSection");
const errorMessage  = document.getElementById("errorMessage");
const copyBtn       = document.getElementById("copyBtn");
const statusDot     = document.getElementById("statusDot");
const statusLabel   = document.getElementById("statusLabel");


/* ── 2. Health check on page load ───────────────────────────────────────── */

async function checkHealth() {
  statusDot.className   = "status-dot checking";
  statusLabel.textContent = "Checking systems…";

  try {
    const response = await fetch("/health");
    const data     = await response.json();

    if (data.status === "ok") {
      statusDot.className     = "status-dot ok";
      statusLabel.textContent = "All systems operational";
    } else {
      statusDot.className     = "status-dot error";
      statusLabel.textContent = "One or more providers degraded";
    }
  } catch {
    statusDot.className     = "status-dot error";
    statusLabel.textContent = "Cannot reach backend";
  }
}


/* ── 3. Character counter ───────────────────────────────────────────────── */

function updateCharCount() {
  const count = goalInput.value.length;
  charCount.textContent = `${count} character${count !== 1 ? "s" : ""}`;
}

goalInput.addEventListener("input", updateCharCount);


/* ── 4. UI state machine ────────────────────────────────────────────────── */

function setStateIdle() {
  runBtn.disabled             = false;
  runBtn.querySelector(".btn-text").textContent = "Run Research";
  pulseRing.classList.remove("pulsing");
  statusBar.hidden            = true;
  progressFill.style.width    = "0%";
}

function setStateRunning() {
  runBtn.disabled             = true;
  runBtn.querySelector(".btn-text").textContent = "Researching…";
  pulseRing.classList.add("pulsing");
  reportSection.hidden        = true;
  errorSection.hidden         = true;
  reportBody.textContent      = "";
  statusBar.hidden            = false;
  statusBarText.textContent   = "Initialising agent…";
  progressFill.style.width    = "8%";
}

function setStateDone() {
  runBtn.disabled             = false;
  runBtn.querySelector(".btn-text").textContent = "Run Research";
  pulseRing.classList.remove("pulsing");
  progressFill.style.width    = "100%";
  setTimeout(() => {
    statusBar.hidden          = true;
    progressFill.style.width  = "0%";
  }, 1200);
}

function setStateError(message) {
  setStateIdle();
  statusBar.hidden            = true;
  errorSection.hidden         = false;
  errorMessage.textContent    = message;
}


/* ── 5. Progress status messages ────────────────────────────────────────── */

// Cycle through status messages while the agent is working
const STATUS_MESSAGES = [
  "Searching the web…",
  "Fetching pages…",
  "Extracting content…",
  "Storing findings…",
  "Analysing results…",
  "Building intelligence report…",
];

let statusIndex    = 0;
let statusInterval = null;

function startStatusCycle() {
  statusIndex = 0;
  statusInterval = setInterval(() => {
    statusIndex = (statusIndex + 1) % STATUS_MESSAGES.length;
    statusBarText.textContent = STATUS_MESSAGES[statusIndex];

    // Advance the progress bar gradually
    const currentWidth = parseFloat(progressFill.style.width) || 8;
    const nextWidth    = Math.min(currentWidth + 12, 88); // cap at 88% until done
    progressFill.style.width = `${nextWidth}%`;
  }, 5000);
}

function stopStatusCycle() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}


/* ── 6. Simple markdown renderer for the report ─────────────────────────── */

function renderMarkdown(text) {
  // Convert ## headings
  text = text.replace(/^## (.+)$/gm, "<h2>$1</h2>");
  // Convert **bold**
  text = text.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  // Convert URLs to links
  text = text.replace(
    /(https?:\/\/[^\s,\)]+)/g,
    '<a href="$1" target="_blank" rel="noopener">$1</a>'
  );
  // Convert line breaks to paragraphs
  const lines = text.split(/\n{2,}/);
  return lines
    .map(line => line.trim())
    .filter(Boolean)
    .map(line => (line.startsWith("<h2>") ? line : `<p>${line}</p>`))
    .join("\n");
}


/* ── 7. SSE streaming ────────────────────────────────────────────────────── */

async function runResearch() {
  const goal = goalInput.value.trim();

  if (!goal) {
    goalInput.focus();
    return;
  }

  setStateRunning();
  startStatusCycle();

  let accumulatedText = "";

  try {
    const response = await fetch("/research", {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify({ goal }),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.error || "Server returned an error.");
    }

    const reader  = response.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";

    // Read the SSE stream
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop(); // keep incomplete last line

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;

        let event;
        try {
          event = JSON.parse(line.slice(6));
        } catch {
          continue;
        }

        if (event.type === "chunk" && event.content) {
          accumulatedText = event.content; // agent sends full content, not deltas
          reportSection.hidden  = false;
          reportBody.innerHTML  = renderMarkdown(accumulatedText);
          reportBody.scrollIntoView({ behavior: "smooth", block: "nearest" });
        }

        if (event.type === "done") {
          stopStatusCycle();
          setStateDone();
        }

        if (event.type === "error") {
          throw new Error(event.content || "Agent encountered an error.");
        }
      }
    }

    // Fallback: if stream ended without a "done" event
    if (accumulatedText) {
      stopStatusCycle();
      setStateDone();
    } else {
      throw new Error("The agent completed but produced no output. Please try again.");
    }

  } catch (err) {
    stopStatusCycle();
    setStateError(err.message);
  }
}


/* ── 8. Copy report to clipboard ─────────────────────────────────────────── */

function copyReport() {
  const text = reportBody.innerText;
  navigator.clipboard.writeText(text).then(() => {
    copyBtn.querySelector("span").textContent = "Copied!";
    setTimeout(() => {
      copyBtn.querySelector("span").textContent = "Copy";
    }, 2000);
  }).catch(() => {
    copyBtn.querySelector("span").textContent = "Failed";
    setTimeout(() => {
      copyBtn.querySelector("span").textContent = "Copy";
    }, 2000);
  });
}


/* ── 9. Enter key shortcut (Ctrl/Cmd + Enter) ────────────────────────────── */

goalInput.addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
    runResearch();
  }
});


/* ── 10. Event listeners ────────────────────────────────────────────────── */

runBtn.addEventListener("click",  runResearch);
copyBtn.addEventListener("click", copyReport);


/* ── 11. Initialise ─────────────────────────────────────────────────────── */

checkHealth();
updateCharCount();
