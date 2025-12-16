// Global state
let currentSentenceId = null;
let sentenceData = null;
let allSentences = [];

// Punctuation characters to ignore
const PUNCT_CHARS = new Set(".,!?;:\"'[]()،؟«»-:…/\\");

// Check if a token is punctuation
function isPunctuation(token) {
  if (!token || !token.trim()) return false;
  const cleaned = token.trim().replace(/\s/g, "");
  if (!cleaned) return false;
  return cleaned.split("").every((ch) => PUNCT_CHARS.has(ch));
}

// Initialize dashboard
document.addEventListener("DOMContentLoaded", function () {
  loadSentences();

  // Search functionality
  document
    .getElementById("sentence-search")
    .addEventListener("input", function (e) {
      filterSentences(e.target.value);
    });
});

// Load all sentence IDs
async function loadSentences() {
  try {
    const response = await fetch("/api/sentences");
    const data = await response.json();
    allSentences = data.sentence_ids;
    renderSentenceList(allSentences);
  } catch (error) {
    console.error("Error loading sentences:", error);
    document.getElementById("sentence-list").innerHTML =
      '<p class="error">Error loading sentences</p>';
  }
}

// Render sentence list
function renderSentenceList(sentences) {
  const listEl = document.getElementById("sentence-list");
  if (sentences.length === 0) {
    listEl.innerHTML = "<p>No sentences found</p>";
    return;
  }

  listEl.innerHTML = sentences
    .map(
      (id) =>
        `<div class="sentence-item" onclick="loadSentence('${id}')">${id}</div>`
    )
    .join("");
}

// Filter sentences
function filterSentences(query) {
  const filtered = allSentences.filter((id) =>
    id.toLowerCase().includes(query.toLowerCase())
  );
  renderSentenceList(filtered);
}

// Load sentence data
async function loadSentence(sentenceId) {
  currentSentenceId = sentenceId;

  // Update UI
  document.getElementById(
    "current-sentence-id"
  ).textContent = `Sentence ID: ${sentenceId}`;
  document.querySelectorAll(".sentence-item").forEach((el) => {
    el.classList.remove("active");
    if (el.textContent === sentenceId) {
      el.classList.add("active");
    }
  });

  // Show loading
  document.getElementById("visualizations").innerHTML =
    '<div class="loading">Loading sentence data...</div>';

  try {
    const response = await fetch(`/api/sentence/${sentenceId}`);
    if (!response.ok) {
      throw new Error("Sentence not found");
    }
    sentenceData = await response.json();
    renderDashboard();
  } catch (error) {
    console.error("Error loading sentence:", error);
    document.getElementById(
      "visualizations"
    ).innerHTML = `<div class="error">Error loading sentence: ${error.message}</div>`;
  }
}

// Render dashboard with sentence data
function renderDashboard() {
  if (!sentenceData) return;

  // Render metrics table
  renderMetricsTable();

  // Render visualizations
  renderVisualizations();
}

// Render metrics table
function renderMetricsTable() {
  const tbody = document.getElementById("metrics-body");
  const rows = [];

  // Gold reference
  rows.push(
    createMetricsRow(
      "Gold (Reference)",
      {
        AER: 0,
        precision: 1,
        recall: 1,
        f1: 1,
      },
      "gold"
    )
  );

  // System metrics
  const categories = ["baseline", "ner", "seg_bert"];
  categories.forEach((category) => {
    Object.keys(sentenceData.systems).forEach((sysKey) => {
      const sys = sentenceData.systems[sysKey];
      if (sys.category === category) {
        rows.push(createMetricsRow(sys.display_name, sys.metrics, sysKey));
      }
    });
  });

  tbody.innerHTML = rows.join("");
}

function createMetricsRow(name, metrics, key) {
  return `
        <tr data-system="${key}">
            <td><strong>${name}</strong></td>
            <td>${metrics.AER.toFixed(4)}</td>
            <td>${metrics.precision.toFixed(4)}</td>
            <td>${metrics.recall.toFixed(4)}</td>
            <td>${metrics.f1.toFixed(4)}</td>
        </tr>
    `;
}

// Render visualizations
function renderVisualizations() {
  const container = document.getElementById("visualizations");

  // Gold visualization (use first available gold)
  const goldHtml = createVisualizationPanel(
    "Gold Alignment",
    sentenceData.gold,
    null,
    sentenceData.gold,
    true
  );

  // System visualizations
  const systemsHtml = Object.keys(sentenceData.systems)
    .map((sysKey) => {
      const sys = sentenceData.systems[sysKey];
      // Each system uses its own gold data
      const goldForSystem = sys.gold || sentenceData.gold;
      return createVisualizationPanel(
        sys.display_name,
        sys,
        sys.predicted_pairs,
        goldForSystem,
        false
      );
    })
    .join("");

  container.innerHTML = goldHtml + systemsHtml;

  // Render alignment lines after DOM is updated and layout is calculated
  setTimeout(() => {
    renderAllAlignments();
  }, 200);
}

// Create visualization panel
function createVisualizationPanel(
  title,
  systemData,
  predictedPairs,
  goldData,
  isGold
) {
  const arTokens = systemData.ar_tokens || [];
  const enTokens = systemData.en_tokens || [];

  // Use gold data for gold panel, system data for others
  const displayGold = isGold ? systemData : systemData.gold || goldData;

  const arWords = arTokens
    .map(
      (token, idx) =>
        `<span class="word-box" data-ar-index="${idx}" data-index="${
          idx + 1
        }" data-word="${token}">${token}</span>`
    )
    .join("");

  const enWords = enTokens
    .map(
      (token, idx) =>
        `<span class="word-box" data-en-index="${idx}" data-index="${
          idx + 1
        }" data-word="${token}">${token}</span>`
    )
    .join("");

  return `
        <div class="visualization-panel" data-system="${
          isGold ? "gold" : systemData.display_name
        }">
            <h3>${title}</h3>
            <div class="alignment-container" id="alignment-${
              isGold ? "gold" : systemData.display_name.replace(/\s+/g, "-")
            }">
                <div class="sentence-row" id="ar-row-${
                  isGold ? "gold" : systemData.display_name.replace(/\s+/g, "-")
                }">
                    <span class="sentence-label">Arabic:</span>
                    <div class="words-container">
                        ${arWords}
                    </div>
                </div>
                <div class="sentence-row" id="en-row-${
                  isGold ? "gold" : systemData.display_name.replace(/\s+/g, "-")
                }">
                    <span class="sentence-label">English:</span>
                    <div class="words-container">
                        ${enWords}
                    </div>
                </div>
                <svg class="alignment-svg" id="svg-${
                  isGold ? "gold" : systemData.display_name.replace(/\s+/g, "-")
                }"></svg>
            </div>
        </div>
    `;
}

// Render alignment lines
function renderAllAlignments() {
  // Render gold alignments - use gold's own alignments
  const goldAlignments = sentenceData.gold.alignments || [];
  renderAlignmentLines(
    "gold",
    sentenceData.gold,
    goldAlignments,
    goldAlignments,
    true
  );

  // Render system alignments
  Object.keys(sentenceData.systems).forEach((sysKey) => {
    const sys = sentenceData.systems[sysKey];
    const panelId = sys.display_name.replace(/\s+/g, "-");
    // Use system's own gold data
    const goldAlignments = (sys.gold && sys.gold.alignments) || [];
    renderAlignmentLines(
      panelId,
      sys,
      sys.predicted_alignments || [],
      goldAlignments,
      false
    );
  });
}

// Render alignment lines for a panel
function renderAlignmentLines(
  panelId,
  systemData,
  predictedAlignments,
  goldAlignments,
  isGold
) {
  const container = document.getElementById(`alignment-${panelId}`);
  if (!container) return;

  const svg = document.getElementById(`svg-${panelId}`);
  if (!svg) return;

  // Get word positions
  const arWords = container.querySelectorAll("[data-ar-index]");
  const enWords = container.querySelectorAll("[data-en-index]");

  // Get container bounds for relative positioning
  const containerRect = container.getBoundingClientRect();

  const arPositions = Array.from(arWords).map((el) => {
    const rect = el.getBoundingClientRect();
    return {
      index: parseInt(el.dataset.arIndex),
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.bottom - containerRect.top - 2, // Slightly above bottom edge
      element: el,
    };
  });

  const enPositions = Array.from(enWords).map((el) => {
    const rect = el.getBoundingClientRect();
    return {
      index: parseInt(el.dataset.enIndex),
      x: rect.left - containerRect.left + rect.width / 2,
      y: rect.top - containerRect.top + 2, // Slightly below top edge
      element: el,
    };
  });

  // Set SVG size to match container
  const containerWidth =
    container.offsetWidth || container.getBoundingClientRect().width;
  const containerHeight =
    container.offsetHeight || container.getBoundingClientRect().height;
  svg.setAttribute("width", containerWidth);
  svg.setAttribute("height", containerHeight);
  svg.style.width = containerWidth + "px";
  svg.style.height = containerHeight + "px";

  // Clear previous lines and classes
  svg.innerHTML = "";
  arWords.forEach((el) => {
    el.className = "word-box";
  });
  enWords.forEach((el) => {
    el.className = "word-box";
  });

  if (isGold) {
    // Render gold alignments (all correct - show all gold alignments)
    const alignmentsToRender = goldAlignments || [];
    if (alignmentsToRender.length > 0) {
      alignmentsToRender.forEach((align) => {
        // Skip punctuation alignments
        const arWord = systemData.ar_tokens[align.ar_index];
        const enWord = systemData.en_tokens[align.en_index];
        if (isPunctuation(arWord) || isPunctuation(enWord)) {
          return;
        }

        const arPos = arPositions.find((p) => p.index === align.ar_index);
        const enPos = enPositions.find((p) => p.index === align.en_index);

        if (arPos && enPos) {
          const line = createLine(
            arPos.x,
            arPos.y,
            enPos.x,
            enPos.y,
            "correct"
          );
          svg.appendChild(line);
          arPos.element.classList.add("aligned");
          enPos.element.classList.add("aligned");
        }
      });
    }
  } else {
    // Create sets for comparison
    const goldSet = new Set();
    if (goldAlignments && goldAlignments.length > 0) {
      goldAlignments.forEach((align) => {
        goldSet.add(`${align.ar_index}-${align.en_index}`);
      });
    }

    const predictedSet = new Set();
    if (predictedAlignments && predictedAlignments.length > 0) {
      predictedAlignments.forEach((align) => {
        // Skip punctuation alignments
        const arWord = systemData.ar_tokens[align.ar_index];
        const enWord = systemData.en_tokens[align.en_index];
        if (isPunctuation(arWord) || isPunctuation(enWord)) {
          return;
        }

        const key = `${align.ar_index}-${align.en_index}`;
        predictedSet.add(key);

        const arPos = arPositions.find((p) => p.index === align.ar_index);
        const enPos = enPositions.find((p) => p.index === align.en_index);

        if (arPos && enPos) {
          const isCorrect = goldSet.has(key);
          const className = isCorrect ? "correct" : "incorrect";

          const line = createLine(
            arPos.x,
            arPos.y,
            enPos.x,
            enPos.y,
            className
          );
          svg.appendChild(line);

          // Highlight words
          if (isCorrect) {
            arPos.element.classList.add("aligned");
            enPos.element.classList.add("aligned");
          } else {
            arPos.element.classList.add("misaligned");
            enPos.element.classList.add("misaligned");
          }
        }
      });
    }

    // Highlight gold-only alignments (missed by prediction)
    if (goldAlignments && goldAlignments.length > 0) {
      goldAlignments.forEach((align) => {
        // Skip punctuation alignments
        const arWord = systemData.ar_tokens[align.ar_index];
        const enWord = systemData.en_tokens[align.en_index];
        if (isPunctuation(arWord) || isPunctuation(enWord)) {
          return;
        }

        const key = `${align.ar_index}-${align.en_index}`;
        if (!predictedSet.has(key)) {
          const arPos = arPositions.find((p) => p.index === align.ar_index);
          const enPos = enPositions.find((p) => p.index === align.en_index);

          if (arPos && enPos) {
            const line = createLine(
              arPos.x,
              arPos.y,
              enPos.x,
              enPos.y,
              "gold-only"
            );
            svg.appendChild(line);

            arPos.element.classList.add("gold-only");
            enPos.element.classList.add("gold-only");
          }
        }
      });
    }
  }
}

// Create SVG line
function createLine(x1, y1, x2, y2, className) {
  const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
  line.setAttribute("x1", x1);
  line.setAttribute("y1", y1);
  line.setAttribute("x2", x2);
  line.setAttribute("y2", y2);
  line.setAttribute("class", `alignment-line ${className}`);
  return line;
}
