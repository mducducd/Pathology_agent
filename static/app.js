(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const folderInput = document.getElementById("folder-input");
  const zipInput = document.getElementById("zip-input");
  const uploadActionTrigger = document.getElementById("upload-action-trigger");
  const uploadActionMenu = document.getElementById("upload-action-menu");
  const uploadActionOptions = Array.from(document.querySelectorAll(".upload-action-option"));
  const uploadActionTip = document.getElementById("upload-action-tip");

  const btnClear = document.getElementById("btn-clear");
  const btnStart = document.getElementById("btn-start");

  const filelist = document.getElementById("filelist");
  const filelistMeta = document.getElementById("filelist-meta");
  const validationEl = document.getElementById("validation");
  const modePill = document.getElementById("mode-pill");

  const promptEl = document.getElementById("prompt");
  const agentSelect = document.getElementById("agent-select");
  const modelSelect = document.getElementById("model-select");
  const extractorSelect = document.getElementById("extractor-select");
  const tileSizeSelect = document.getElementById("tile-size-select");
  const batchSizeSelect = document.getElementById("batch-size-select");
  const tilePrefilterMethodSelect = document.getElementById("tile-prefilter-method-select");
  const tilePrefilterMethodStorageKey = "slide-agent.tile-prefilter-method";

  const statusPill = document.getElementById("status-pill");
  const btnActions = document.getElementById("btn-actions");
  const statusActionsMenu = document.getElementById("status-actions-menu");
  const btnRestart = document.getElementById("btn-restart");
  const btnClearOutputs = document.getElementById("btn-clear-outputs");
  const btnTerminate = document.getElementById("btn-terminate");
  const runLoadingEl = document.getElementById("run-loading");
  const runStateTextEl = document.getElementById("run-state-text");
  const runIdEl = document.getElementById("run-id");
  const runLabelEl = document.getElementById("run-label");
  const slideNameEl = document.getElementById("slide-name");

  const errorBox = document.getElementById("error-box");
  const errorText = document.getElementById("error-text");

  const overviewImg = document.getElementById("overview-img");
  const overviewCanvas = document.getElementById("overview-canvas");
  const overviewEmpty = document.getElementById("overview-empty");
  const darkImg = document.getElementById("dark-img");
  const darkCanvas = document.getElementById("dark-canvas");
  const darkEmpty = document.getElementById("dark-empty");
  const btnDarkToggle = document.getElementById("btn-dark-toggle");
  const finalText = document.getElementById("final-text");
  const reasoningSection = document.getElementById("reasoning-section");
  const reasoningText = document.getElementById("reasoning-text");
  const reportLink = document.getElementById("report-link");
  const stepsEl = document.getElementById("steps");
  const roisEl = document.getElementById("rois");
  const workbenchEl = document.querySelector(".workbench");
  const panelLeftEl = document.querySelector(".panel-left");
  const panelRightEl = document.querySelector(".panel-right");
  const viewerGridEl = document.querySelector(".viewer-grid");
  const viewerMainLeftEl = document.querySelector(".viewer-main-left");
  const resizerLeftMainEl = document.getElementById("resizer-left-main");
  const resizerMainRightEl = document.getElementById("resizer-main-right");
  const resizerViewerSplitEl = document.getElementById("resizer-viewer-split");
  const explorerModal = document.getElementById("explorer-modal");
  const explorerBackdrop = document.getElementById("explorer-backdrop");
  const explorerCloseBtn = document.getElementById("explorer-close");
  const explorerRootSelect = document.getElementById("explorer-root-select");
  const explorerUpBtn = document.getElementById("explorer-up");
  const explorerRefreshBtn = document.getElementById("explorer-refresh");
  const explorerBreadcrumbs = document.getElementById("explorer-breadcrumbs");
  const explorerCurrentPath = document.getElementById("explorer-current-path");
  const explorerError = document.getElementById("explorer-error");
  const explorerEmpty = document.getElementById("explorer-empty");
  const explorerList = document.getElementById("explorer-list");
  const explorerSelectionPreview = document.getElementById("explorer-selection-preview");
  const explorerUseFolderBtn = document.getElementById("explorer-use-folder");
  const explorerUseFileBtn = document.getElementById("explorer-use-file");

  let defaultPrompts = {
    tile: "",
    aml: "",
    wsi: ""
  };
  const uploadActionHints = {
    files: "Standard slides: .svs / .tif / .tiff / .ndpi (single file).",
    folder: "MIRAX: choose the MIRAX folder containing .mrxs/.mrsx and its data directory.",
    zip: "MIRAX zip: upload one .zip containing .mrxs/.mrsx plus the data directory.",
    server: "Browse HPC slide roots directly and run from the server without uploading the WSI.",
  };
  const uploadActionDefaultHint = "Choose a slide source.";

  function selectedAgentType() {
    return (agentSelect && agentSelect.value) ? agentSelect.value : "tile";
  }

  function selectedModelName() {
    return (modelSelect && modelSelect.value) ? modelSelect.value : "GPT-OSS-120B";
  }

  function selectedBatchSize() {
    const parsed = Number.parseInt(
      (batchSizeSelect && batchSizeSelect.value) ? batchSizeSelect.value : "128",
      10
    );
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 128;
  }

  function selectedTilePrefilterMethod() {
    return (tilePrefilterMethodSelect && tilePrefilterMethodSelect.value) ? tilePrefilterMethodSelect.value : "quality";
  }

  function loadStoredValue(key) {
    try {
      return window.localStorage.getItem(key);
    } catch (_e) {
      // Ignore localStorage access issues.
    }
    return null;
  }

  function saveStoredValue(key, value) {
    try {
      window.localStorage.setItem(key, value);
    } catch (_e) {
      // Ignore localStorage access issues.
    }
  }

  function uploadHintForAction(action) {
    if (!action || !uploadActionHints[action]) return uploadActionDefaultHint;
    return uploadActionHints[action];
  }

  function hideUploadActionTip() {
    if (!uploadActionTip) return;
    uploadActionTip.hidden = true;
  }

  function closeUploadActionMenu() {
    if (!uploadActionMenu || !uploadActionTrigger) return;
    uploadActionMenu.hidden = true;
    uploadActionTrigger.setAttribute("aria-expanded", "false");
    hideUploadActionTip();
  }

  function openUploadActionMenu() {
    if (!uploadActionMenu || !uploadActionTrigger) return;
    uploadActionMenu.hidden = false;
    uploadActionTrigger.setAttribute("aria-expanded", "true");
  }

  function toggleUploadActionMenu() {
    if (!uploadActionMenu || !uploadActionTrigger) return;
    if (uploadActionMenu.hidden) {
      openUploadActionMenu();
      return;
    }
    closeUploadActionMenu();
  }

  function positionUploadActionTip(clientX, clientY) {
    if (!uploadActionTip || uploadActionTip.hidden) return;
    const margin = 12;
    const offsetX = 16;
    const offsetY = 14;
    const tipW = uploadActionTip.offsetWidth || 220;
    const tipH = uploadActionTip.offsetHeight || 48;
    const maxX = window.innerWidth - tipW - margin;
    const maxY = window.innerHeight - tipH - margin;
    let x = clientX + offsetX;
    let y = clientY + offsetY;
    if (x > maxX) x = Math.max(margin, clientX - tipW - 16);
    if (y > maxY) y = maxY;
    uploadActionTip.style.left = `${x}px`;
    uploadActionTip.style.top = `${y}px`;
  }

  function showUploadActionTip(text, clientX, clientY) {
    if (!uploadActionTip) return;
    uploadActionTip.textContent = text || uploadActionDefaultHint;
    uploadActionTip.hidden = false;
    positionUploadActionTip(clientX, clientY);
  }

  function triggerUploadAction(action) {
    if (action === "files") {
      fileInput.value = "";
      fileInput.click();
      return;
    }
    if (action === "folder") {
      folderInput.value = "";
      folderInput.click();
      return;
    }
    if (action === "zip") {
      zipInput.value = "";
      zipInput.click();
      return;
    }
    if (action === "server") {
      openExplorer();
    }
  }

  function isKnownDefaultPrompt(value) {
    if (!value) return false;
    return Object.values(defaultPrompts).some((prompt) => prompt && prompt === value);
  }

  async function loadDefaultPrompts() {
    try {
      const res = await fetch("/api/default_prompts", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const payload = await res.json();
      const prompts = payload && payload.prompts ? payload.prompts : {};
      const nextPrompts = {
        tile: typeof prompts.tile === "string" ? prompts.tile : "",
        aml: typeof prompts.aml === "string" ? prompts.aml : "",
        wsi: typeof prompts.wsi === "string" ? prompts.wsi : ""
      };
      const currentValue = promptEl.value;
      const shouldReplace = !currentValue.trim() || isKnownDefaultPrompt(currentValue);
      defaultPrompts = nextPrompts;
      if (shouldReplace) {
        promptEl.value = defaultPrompts[selectedAgentType()] || "";
      }
    } catch (err) {
      console.warn("Failed to load default prompts from backend", err);
    }
  }

  async function loadEmbeddingExtractorOptions() {
    if (!extractorSelect) return;
    try {
      const res = await fetch("/api/embedding_extractors", { cache: "no-store" });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const payload = await res.json();
      const extractors = Array.isArray(payload && payload.extractors) ? payload.extractors : [];
      if (!extractors.length) return;

      const currentValue = extractorSelect.value;
      const defaultExtractor =
        payload && typeof payload.default_extractor === "string" && payload.default_extractor
          ? payload.default_extractor
          : "uni2";

      extractorSelect.innerHTML = "";
      for (const item of extractors) {
        const name = item && typeof item.name === "string" ? item.name : "";
        if (!name) continue;
        const label =
          item && typeof item.label === "string" && item.label.trim()
            ? item.label
            : name;
        const option = document.createElement("option");
        option.value = name;
        option.textContent = label;
        extractorSelect.appendChild(option);
      }

      const nextValue = Array.from(extractorSelect.options).some((opt) => opt.value === currentValue)
        ? currentValue
        : defaultExtractor;
      extractorSelect.value = nextValue;
    } catch (err) {
      console.warn("Failed to load embedding extractor options from backend", err);
    }
  }

  function maybeLoadDefaultPrompt() {
    const type = selectedAgentType();
    const d = defaultPrompts[type] || "";
    if (!promptEl.value.trim()) {
      promptEl.value = d;
      return;
    }
    if (isKnownDefaultPrompt(promptEl.value)) {
      promptEl.value = d;
    }
  }

  if (agentSelect) {
    agentSelect.addEventListener("change", maybeLoadDefaultPrompt);
  }
  loadDefaultPrompts();
  loadEmbeddingExtractorOptions();

  const allowedPrimary = new Set([".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".mrsx"]);
  const allowedZip = ".zip";
  const standard = new Set([".svs", ".tif", ".tiff", ".ndpi"]);
  const mirax = new Set([".mrxs", ".mrsx"]);

  let items = []; // {file, relPath, id}
  let serverSelection = null; // {requestedPath, slidePath, slideFilename, selectionKind, selectionLabel}
  let explorerRoots = [];
  let explorerCurrentPathValue = "";
  let explorerParentPathValue = "";
  let explorerEntries = [];
  let explorerBreadcrumbItems = [];
  let explorerSelectedFilePath = "";
  let explorerSelectedFileName = "";
  let explorerBusy = false;
  let currentRunId = null;
  let currentModelName = null;
  let pollingTimer = null;
  let lastRenderedStep = 0;
  let lastRenderedRoi = 0;
  let darkRegionsLoaded = false;
  let darkBoxes = [];
  let darkMaskMode = false;
  let darkRegionsEnabled = false;
  let baseOverviewImageUrl = "";
  let selectedOverviewRoiId = null;
  let overviewCacheState = null;
  const roiById = new Map();
  let currentReportHref = null;
  let reportFetchToken = 0;
  let activeRunStatus = "";
  let modelSelectionTouched = false;
  let runRequestedModel = null;
  let terminateRequested = false;
  let activeUploadXhr = null;
  let lastCurrentViewState = null;
  let searchTargetBoxPx = null;
  let searchDrawBoxPx = null;
  let searchTransition = null;
  let searchPulse = 0;
  let searchDashOffset = 0;
  let overlayRafId = null;
  let roiListPinnedToBottom = true;
  let imageRevealSeq = 0;
  const ROI_BOX_COLORS = [
    "#0072B2", "#D55E00", "#009E73", "#332288", "#CC79A7", "#117733",
    "#56B4E9", "#AA4499", "#E69F00", "#44AA99", "#88CCEE", "#1F77B4",
  ];
  const ROI_LIST_AUTOSCROLL_THRESHOLD_PX = 36;
  const LAYOUT_STORAGE_KEYS = {
    version: "layout.version",
    leftColPx: "layout.leftColPx",
    rightColPx: "layout.rightColPx",
    viewerLeftColPx: "layout.viewerLeftColPx",
  };
  const LAYOUT_STORAGE_VERSION = "3";
  const DEFAULT_LAYOUT_RATIO = {
    mainLeft: 0.20,
    mainRight: 0.20,
    viewerLeft: 0.70,
  };
  const LAYOUT_BOUNDS = {
    mainLeftMin: 280,
    mainCenterMin: 480,
    mainRightMin: 280,
    viewerLeftMin: 320,
    viewerRightMin: 260,
  };
  const OVERVIEW_EMPTY_TEXT = "No overview yet. Upload a slide and start a run.";
  const DARK_EMPTY_OFF_TEXT = "Click Show to run dark-region detection.";
  const DARK_EMPTY_LOADING_TEXT = "Preparing dark-region view…";
  let layoutResizeRaf = null;

  function isCompactLayout() {
    return window.matchMedia("(max-width: 1160px)").matches;
  }

  function clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }

  function readStoredPx(key) {
    try {
      const raw = localStorage.getItem(key);
      const n = Number(raw);
      return Number.isFinite(n) ? n : null;
    } catch (_e) {
      return null;
    }
  }

  function writeStoredPx(key, value) {
    try {
      localStorage.setItem(key, String(Math.round(value)));
    } catch (_e) {
      // Ignore persistence failures (private mode or blocked storage).
    }
  }

  function readCssVarPx(el, name, fallback) {
    if (!el) return fallback;
    const raw = getComputedStyle(el).getPropertyValue(name).trim();
    if (!raw) return fallback;
    if (raw.includes("%")) return fallback;
    const n = Number.parseFloat(raw);
    return Number.isFinite(n) ? n : fallback;
  }

  function workbenchResizerSizePx() {
    return readCssVarPx(workbenchEl, "--col-resizer-size", 12);
  }

  function viewerResizerSizePx() {
    return readCssVarPx(viewerGridEl, "--subcol-resizer-size", 10);
  }

  function currentWorkbenchWidths() {
    const leftFallback = panelLeftEl ? panelLeftEl.getBoundingClientRect().width : 360;
    const rightFallback = panelRightEl ? panelRightEl.getBoundingClientRect().width : 380;
    return {
      left: readCssVarPx(workbenchEl, "--left-col", leftFallback),
      right: readCssVarPx(workbenchEl, "--right-col", rightFallback),
    };
  }

  function applyWorkbenchWidths(nextLeft, nextRight, persist = true) {
    if (!workbenchEl || isCompactLayout()) return;
    const total = workbenchEl.getBoundingClientRect().width;
    const handle = workbenchResizerSizePx();
    if (!Number.isFinite(total) || total <= 0) return;

    const minLeft = LAYOUT_BOUNDS.mainLeftMin;
    const minMain = LAYOUT_BOUNDS.mainCenterMin;
    const minRight = LAYOUT_BOUNDS.mainRightMin;
    const minNeeded = minLeft + minMain + minRight + handle * 2;
    if (total <= minNeeded) return;

    const current = currentWorkbenchWidths();
    let left = Number.isFinite(nextLeft) ? nextLeft : current.left;
    let right = Number.isFinite(nextRight) ? nextRight : current.right;

    const maxLeft = Math.max(minLeft, total - minRight - minMain - handle * 2);
    left = clamp(left, minLeft, maxLeft);

    const maxRight = Math.max(minRight, total - left - minMain - handle * 2);
    right = clamp(right, minRight, maxRight);

    if (left + right + minMain + handle * 2 > total) {
      right = Math.max(minRight, total - left - minMain - handle * 2);
    }

    workbenchEl.style.setProperty("--left-col", `${Math.round(left)}px`);
    workbenchEl.style.setProperty("--right-col", `${Math.round(right)}px`);

    if (persist) {
      writeStoredPx(LAYOUT_STORAGE_KEYS.leftColPx, left);
      writeStoredPx(LAYOUT_STORAGE_KEYS.rightColPx, right);
    }
  }

  function currentViewerLeftWidth() {
    const fallback = viewerMainLeftEl ? viewerMainLeftEl.getBoundingClientRect().width : 560;
    return readCssVarPx(viewerGridEl, "--viewer-left-col", fallback);
  }

  function defaultWorkbenchWidths() {
    const fallback = currentWorkbenchWidths();
    if (!workbenchEl) return fallback;
    const total = workbenchEl.getBoundingClientRect().width;
    const handle = workbenchResizerSizePx();
    if (!Number.isFinite(total) || total <= 0) return fallback;

    const usable = Math.max(0, total - handle * 2);
    return {
      left: usable * DEFAULT_LAYOUT_RATIO.mainLeft,
      right: usable * DEFAULT_LAYOUT_RATIO.mainRight,
    };
  }

  function defaultViewerLeftWidth() {
    const fallback = currentViewerLeftWidth();
    if (!viewerGridEl) return fallback;
    const total = viewerGridEl.getBoundingClientRect().width;
    const handle = viewerResizerSizePx();
    if (!Number.isFinite(total) || total <= 0) return fallback;
    const usable = Math.max(0, total - handle);
    return usable * DEFAULT_LAYOUT_RATIO.viewerLeft;
  }

  function migrateLayoutDefaultsIfNeeded() {
    try {
      const version = localStorage.getItem(LAYOUT_STORAGE_KEYS.version);
      if (version === LAYOUT_STORAGE_VERSION) return;
      localStorage.removeItem(LAYOUT_STORAGE_KEYS.leftColPx);
      localStorage.removeItem(LAYOUT_STORAGE_KEYS.rightColPx);
      localStorage.removeItem(LAYOUT_STORAGE_KEYS.viewerLeftColPx);
      localStorage.setItem(LAYOUT_STORAGE_KEYS.version, LAYOUT_STORAGE_VERSION);
    } catch (_e) {
      // Ignore storage access failures.
    }
  }

  function applyViewerSplit(nextLeft, persist = true) {
    if (!viewerGridEl || isCompactLayout()) return;
    const total = viewerGridEl.getBoundingClientRect().width;
    const handle = viewerResizerSizePx();
    if (!Number.isFinite(total) || total <= 0) return;

    const minLeft = LAYOUT_BOUNDS.viewerLeftMin;
    const minRight = LAYOUT_BOUNDS.viewerRightMin;
    const minNeeded = minLeft + minRight + handle;
    if (total <= minNeeded) return;

    const currentLeft = currentViewerLeftWidth();
    const desiredLeft = Number.isFinite(nextLeft) ? nextLeft : currentLeft;
    const maxLeft = Math.max(minLeft, total - minRight - handle);
    const left = clamp(desiredLeft, minLeft, maxLeft);

    viewerGridEl.style.setProperty("--viewer-left-col", `${Math.round(left)}px`);
    if (persist) {
      writeStoredPx(LAYOUT_STORAGE_KEYS.viewerLeftColPx, left);
    }
  }

  function bindHorizontalResizer(handleEl, onMove) {
    if (!handleEl) return;
    handleEl.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0 || isCompactLayout()) return;
      ev.preventDefault();
      handleEl.classList.add("dragging");
      document.body.classList.add("is-resizing");
      try {
        handleEl.setPointerCapture(ev.pointerId);
      } catch (_e) {
        // Ignore pointer capture failures.
      }

      const onPointerMove = (moveEv) => onMove(moveEv);
      const onPointerUp = () => {
        handleEl.classList.remove("dragging");
        document.body.classList.remove("is-resizing");
        handleEl.removeEventListener("pointermove", onPointerMove);
        handleEl.removeEventListener("pointerup", onPointerUp);
        handleEl.removeEventListener("pointercancel", onPointerUp);
      };

      handleEl.addEventListener("pointermove", onPointerMove);
      handleEl.addEventListener("pointerup", onPointerUp);
      handleEl.addEventListener("pointercancel", onPointerUp);
    });
  }

  function queueLayoutClamp() {
    if (layoutResizeRaf !== null) return;
    layoutResizeRaf = requestAnimationFrame(() => {
      layoutResizeRaf = null;
      if (isCompactLayout()) {
        document.body.classList.remove("is-resizing");
        return;
      }
      const wb = currentWorkbenchWidths();
      applyWorkbenchWidths(wb.left, wb.right, false);
      applyViewerSplit(currentViewerLeftWidth(), false);
    });
  }

  function initResizableLayout() {
    if (!workbenchEl || !viewerGridEl) return;
    migrateLayoutDefaultsIfNeeded();

    const savedLeft = readStoredPx(LAYOUT_STORAGE_KEYS.leftColPx);
    const savedRight = readStoredPx(LAYOUT_STORAGE_KEYS.rightColPx);
    const savedViewerLeft = readStoredPx(LAYOUT_STORAGE_KEYS.viewerLeftColPx);
    const wbDefault = defaultWorkbenchWidths();
    const viewerDefault = defaultViewerLeftWidth();

    applyWorkbenchWidths(
      Number.isFinite(savedLeft) ? savedLeft : wbDefault.left,
      Number.isFinite(savedRight) ? savedRight : wbDefault.right,
      false,
    );
    applyViewerSplit(
      Number.isFinite(savedViewerLeft) ? savedViewerLeft : viewerDefault,
      false,
    );

    bindHorizontalResizer(resizerLeftMainEl, (ev) => {
      if (!workbenchEl) return;
      const rect = workbenchEl.getBoundingClientRect();
      const handle = workbenchResizerSizePx();
      const desiredLeft = ev.clientX - rect.left - handle / 2;
      const current = currentWorkbenchWidths();
      applyWorkbenchWidths(desiredLeft, current.right, true);
    });

    bindHorizontalResizer(resizerMainRightEl, (ev) => {
      if (!workbenchEl) return;
      const rect = workbenchEl.getBoundingClientRect();
      const handle = workbenchResizerSizePx();
      const desiredRight = rect.right - ev.clientX - handle / 2;
      const current = currentWorkbenchWidths();
      applyWorkbenchWidths(current.left, desiredRight, true);
    });

    bindHorizontalResizer(resizerViewerSplitEl, (ev) => {
      if (!viewerGridEl) return;
      const rect = viewerGridEl.getBoundingClientRect();
      const handle = viewerResizerSizePx();
      const desiredLeft = ev.clientX - rect.left - handle / 2;
      applyViewerSplit(desiredLeft, true);
    });

    window.addEventListener("resize", queueLayoutClamp, { passive: true });
  }

  function bytesToHuman(n) {
    if (!n) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(n) / Math.log(k));
    const v = n / Math.pow(k, i);
    return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${sizes[i]}`;
  }

  function roiColorForId(roiId) {
    const idx = Math.max(0, (Number(roiId) || 1) - 1) % ROI_BOX_COLORS.length;
    return ROI_BOX_COLORS[idx];
  }

  function isListNearBottom(listEl, thresholdPx = ROI_LIST_AUTOSCROLL_THRESHOLD_PX) {
    if (!listEl) return true;
    const remaining = listEl.scrollHeight - listEl.clientHeight - listEl.scrollTop;
    return remaining <= thresholdPx;
  }

  function syncRoiListPinnedState() {
    roiListPinnedToBottom = isListNearBottom(roisEl);
  }

  function maybeScrollRoiListToBottom(force = false) {
    if (!roisEl) return;
    const shouldStick = force || roiListPinnedToBottom || isListNearBottom(roisEl);
    if (!shouldStick) return;
    roisEl.scrollTop = roisEl.scrollHeight;
    roiListPinnedToBottom = true;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function renderInlineMd(text) {
    let out = escapeHtml(text);
    out = out.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');
    out = out.replace(/`([^`]+)`/g, "<code>$1</code>");
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    out = out.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    out = out.replace(/_([^_]+)_/g, "<em>$1</em>");
    return out;
  }

  function markdownToHtml(mdText) {
    const lines = String(mdText || "").replace(/\r\n/g, "\n").split("\n");
    const html = [];
    let inCode = false;
    let codeLang = "";
    let codeLines = [];
    let listType = null;

    function closeList() {
      if (listType) {
        html.push(`</${listType}>`);
        listType = null;
      }
    }

    function closeCode() {
      if (!inCode) return;
      const cls = codeLang ? ` class="lang-${escapeHtml(codeLang)}"` : "";
      html.push(`<pre><code${cls}>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      inCode = false;
      codeLang = "";
      codeLines = [];
    }

    for (const raw of lines) {
      const line = raw || "";

      if (inCode) {
        if (/^```/.test(line.trim())) {
          closeCode();
        } else {
          codeLines.push(line);
        }
        continue;
      }

      const fence = line.trim().match(/^```([a-zA-Z0-9_-]+)?\s*$/);
      if (fence) {
        closeList();
        inCode = true;
        codeLang = fence[1] || "";
        codeLines = [];
        continue;
      }

      if (!line.trim()) {
        closeList();
        continue;
      }

      const h = line.match(/^(#{1,6})\s+(.+)$/);
      if (h) {
        closeList();
        const level = h[1].length;
        html.push(`<h${level}>${renderInlineMd(h[2])}</h${level}>`);
        continue;
      }

      if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line.trim())) {
        closeList();
        html.push("<hr/>");
        continue;
      }

      const ul = line.match(/^\s*[-*+]\s+(.+)$/);
      if (ul) {
        if (listType !== "ul") {
          closeList();
          listType = "ul";
          html.push("<ul>");
        }
        html.push(`<li>${renderInlineMd(ul[1])}</li>`);
        continue;
      }

      const ol = line.match(/^\s*\d+\.\s+(.+)$/);
      if (ol) {
        if (listType !== "ol") {
          closeList();
          listType = "ol";
          html.push("<ol>");
        }
        html.push(`<li>${renderInlineMd(ol[1])}</li>`);
        continue;
      }

      const bq = line.match(/^\s*>\s+(.+)$/);
      if (bq) {
        closeList();
        html.push(`<blockquote><p>${renderInlineMd(bq[1])}</p></blockquote>`);
        continue;
      }

      closeList();
      html.push(`<p>${renderInlineMd(line)}</p>`);
    }

    closeList();
    closeCode();
    return html.join("\n");
  }

  function extractFinalReportOnly(mdText) {
    const text = String(mdText || "");
    if (!text.trim()) return "";

    const lines = text.replace(/\r\n/g, "\n").split("\n");
    let start = -1;
    let headingLevel = 0;

    for (let i = 0; i < lines.length; i++) {
      const m = lines[i].match(/^\s*(#{1,6})\s*final report\b[:\s-]*/i);
      if (m) {
        start = i + 1;
        headingLevel = m[1].length;
        break;
      }
    }

    if (start < 0) {
      for (let i = 0; i < lines.length; i++) {
        if (/^\s*final report\b[:\s-]*/i.test(lines[i])) {
          start = i + 1;
          headingLevel = 7;
          break;
        }
      }
    }

    if (start < 0) {
      // If this looks like a full generated report, do not show everything in Final report box.
      if (
        /(^|\n)\s*##\s*prompt\b/i.test(text) ||
        /(^|\n)\s*##\s*regions?\s+of\s+interest\b/i.test(text) ||
        /(^|\n)\s*##\s*navigation\s+steps\b/i.test(text) ||
        /(^|\n)\s*##\s*model\s+reasoning\b/i.test(text)
      ) {
        return "";
      }
      return text.trim();
    }

    let end = lines.length;
    if (headingLevel <= 6) {
      for (let i = start; i < lines.length; i++) {
        const m = lines[i].match(/^\s*(#{1,6})\s+/);
        if (m && m[1].length <= headingLevel) {
          end = i;
          break;
        }
      }
    } else {
      for (let i = start; i < lines.length; i++) {
        if (/^\s*(#{1,6}\s+|reasoning\b|navigation steps\b|raw output\b)/i.test(lines[i])) {
          end = i;
          break;
        }
      }
    }

    const section = lines.slice(start, end).join("\n").trim();
    return section;
  }

  function renderFinalReportMarkdown(markdownText, waiting = false) {
    finalText.classList.toggle("waiting", !!waiting);
    const source = String(markdownText || "").trim();
    if (!source) {
      finalText.innerHTML = "";
      return;
    }
    finalText.innerHTML = markdownToHtml(source);
  }

  function setReasoningContent(text) {
    if (!reasoningSection || !reasoningText) return;
    const content = String(text || "").trim();
    const hasContent = content.length > 0;
    reasoningSection.hidden = !hasContent;
    reasoningText.textContent = hasContent ? content : "";
    reasoningText.classList.remove("waiting");
  }

  async function fetchAndRenderReport(href, fallbackText) {
    const token = ++reportFetchToken;
    renderFinalReportMarkdown("Loading report markdown…", true);
    try {
      const res = await fetch(href, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const md = await res.text();
      if (token !== reportFetchToken) return;
      const extracted = extractFinalReportOnly(md);
      if (extracted) {
        renderFinalReportMarkdown(extracted, false);
        return;
      }
      const fallbackExtracted = extractFinalReportOnly(fallbackText || "");
      renderFinalReportMarkdown(fallbackExtracted || "No final report available.", false);
    } catch (_e) {
      if (token !== reportFetchToken) return;
      const fallbackExtracted = extractFinalReportOnly(fallbackText || "");
      renderFinalReportMarkdown(fallbackExtracted || "No final report available.", false);
    }
  }

  function extOf(name) {
    const i = name.lastIndexOf(".");
    if (i < 0) return "";
    return name.slice(i).toLowerCase();
  }

  function uniqId() {
    return Math.random().toString(16).slice(2) + Date.now().toString(16);
  }

  function resetUploadInputs() {
    fileInput.value = "";
    folderInput.value = "";
    zipInput.value = "";
  }

  function setServerSelection(selection) {
    items = [];
    serverSelection = selection ? { ...selection } : null;
    resetUploadInputs();
    render();
  }

  function clearSelection() {
    items = [];
    serverSelection = null;
    resetUploadInputs();
    render();
  }

  function setOverviewEmptyState(message) {
    const text = String(message || "").trim();
    overviewEmpty.textContent = text;
    overviewEmpty.hidden = !text;
  }

  function setDarkEmptyState(message) {
    if (!darkEmpty) return;
    const text = String(message || "").trim();
    darkEmpty.textContent = text;
    darkEmpty.hidden = !text;
  }

  function applyOverviewDisplaySource() {
    const nextSrc = baseOverviewImageUrl || "";

    if (!nextSrc) {
      overviewImg.hidden = true;
      setImageSrcWithReveal(overviewImg, "");
      setOverviewEmptyState(OVERVIEW_EMPTY_TEXT);
      if (overviewCanvas) {
        const ctx = overviewCanvas.getContext("2d");
        ctx.clearRect(0, 0, overviewCanvas.width, overviewCanvas.height);
      }
      return;
    }

    overviewImg.hidden = false;
    setImageSrcWithReveal(overviewImg, nextSrc);
    setOverviewEmptyState("");
    renderOverviewRoiOverlay();
  }

  function clearDarkRegions() {
    darkImg.hidden = true;
    darkImg.src = "";
    darkMaskMode = false;
    setDarkEmptyState(darkRegionsEnabled ? DARK_EMPTY_LOADING_TEXT : DARK_EMPTY_OFF_TEXT);
    darkBoxes = [];
    const ctx = darkCanvas.getContext("2d");
    ctx.clearRect(0, 0, darkCanvas.width, darkCanvas.height);
    applyOverviewDisplaySource();
  }

  function setDarkRegionsEnabled(enabled) {
    darkRegionsEnabled = !!enabled;
    if (btnDarkToggle) {
      btnDarkToggle.textContent = darkRegionsEnabled ? "Hide dark regions" : "Show dark regions";
      btnDarkToggle.setAttribute("aria-pressed", darkRegionsEnabled ? "true" : "false");
      btnDarkToggle.classList.toggle("is-active", darkRegionsEnabled);
    }
    if (!darkRegionsEnabled) {
      darkRegionsLoaded = false;
      clearDarkRegions();
      return;
    }
    clearDarkRegions();
    if (currentRunId) {
      fetchDarkRegions(currentRunId);
    }
  }

  function addFiles(files) {
    if (!files || !files.length) return;
    serverSelection = null;
    for (const f of files) {
      const relPath = f.webkitRelativePath || f.name;
      items.push({ file: f, relPath, id: uniqId() });
    }
    resetUploadInputs();
    render();
  }

  function removeItem(id) {
    items = items.filter((x) => x.id !== id);
    // Allow re-selecting the same file/folder immediately after removing it.
    resetUploadInputs();
    render();
  }

  function computeModeAndValidation() {
    if (serverSelection) {
      return {
        ok: true,
        level: "good",
        mode: serverSelection.selectionLabel || "Server selection",
        msg: `Ready to run directly from server: ${serverSelection.slideFilename || serverSelection.requestedPath}.`,
        startLabel: "Start run",
        selectionMode: "server",
      };
    }

    if (items.length === 0) {
      return {
        ok: false,
        level: "idle",
        mode: "No files",
        msg: "Drop, upload, or browse the server to begin.",
        startLabel: "Start run",
        selectionMode: "none",
      };
    }

    const exts = items.map(x => extOf(x.relPath || x.file.name));
    const hasZip = exts.includes(allowedZip);
    const hasStd = exts.some(e => standard.has(e));
    const hasMirax = exts.some(e => mirax.has(e));
    const totalBytes = items.reduce((a, b) => a + (b.file.size || 0), 0);

    if (hasZip) {
      if (items.length !== 1) {
        return {
          ok: false,
          level: "bad",
          mode: "MIRAX zip (invalid mix)",
          msg: "If you upload a .zip, it must be the only file.",
          startLabel: "Upload & Start run",
          selectionMode: "upload",
        };
      }
      return {
        ok: true,
        level: "good",
        mode: "MIRAX zip",
        msg: `Ready to upload 1 zip (${bytesToHuman(totalBytes)}).`,
        startLabel: "Upload & Start run",
        selectionMode: "upload",
      };
    }

    if (hasStd && hasMirax) {
      return {
        ok: false,
        level: "bad",
        mode: "Mixed (invalid)",
        msg: "Do not mix standard slides with MIRAX in one upload.",
        startLabel: "Upload & Start run",
        selectionMode: "upload",
      };
    }

    if (hasStd) {
      if (items.length !== 1) {
        return {
          ok: false,
          level: "bad",
          mode: "Standard slide (invalid)",
          msg: "Standard slides must be uploaded as a single file.",
          startLabel: "Upload & Start run",
          selectionMode: "upload",
        };
      }
      const e = exts[0];
      if (!allowedPrimary.has(e)) {
        return {
          ok: false,
          level: "bad",
          mode: "Unsupported",
          msg: "Unsupported file.",
          startLabel: "Upload & Start run",
          selectionMode: "upload",
        };
      }
      return {
        ok: true,
        level: "good",
        mode: "Standard slide",
        msg: `Ready (${bytesToHuman(totalBytes)}).`,
        startLabel: "Upload & Start run",
        selectionMode: "upload",
      };
    }

    if (hasMirax) {
      if (items.length === 1) {
        return {
          ok: false,
          level: "bad",
          mode: "MIRAX file only (invalid)",
          msg: "MIRAX needs its companion data directory. Upload folder or zip.",
          startLabel: "Upload & Start run",
          selectionMode: "upload",
        };
      }
      // with per-file upload, file count is not scary anymore
      return {
        ok: true,
        level: "good",
        mode: "MIRAX folder/files",
        msg: `Ready (${items.length} files, ${bytesToHuman(totalBytes)}).`,
        startLabel: "Upload & Start run",
        selectionMode: "upload",
      };
    }

    return {
      ok: false,
      level: "bad",
      mode: "Unsupported",
      msg: "No supported slide detected.",
      startLabel: "Upload & Start run",
      selectionMode: "upload",
    };
  }

  function setPill(el, level, text) {
    if (!el) return;
    el.classList.remove("pill-idle", "pill-good", "pill-warn", "pill-bad", "pill-run");
    el.classList.add(
      level === "good" ? "pill-good" :
      level === "warn" ? "pill-warn" :
      level === "bad"  ? "pill-bad" :
      level === "run"  ? "pill-run" : "pill-idle"
    );
    el.textContent = text;
  }

  function isRunBusyStatus(status) {
    return status === "created" || status === "uploading" || status === "pending" || status === "running";
  }

  function syncStatusPillVisibility(status) {
    if (!statusPill) return;
    const normalized = String(status || "idle").toLowerCase();
    // Hide the idle pill to reduce header clutter; show for active/final states.
    statusPill.hidden = (normalized === "idle" || normalized === "");
  }

  function closeStatusActionsMenu() {
    if (!statusActionsMenu || !btnActions) return;
    statusActionsMenu.hidden = true;
    btnActions.setAttribute("aria-expanded", "false");
  }

  function openStatusActionsMenu() {
    if (!statusActionsMenu || !btnActions) return;
    statusActionsMenu.hidden = false;
    btnActions.setAttribute("aria-expanded", "true");
  }

  function toggleStatusActionsMenu() {
    if (!statusActionsMenu || !btnActions) return;
    if (statusActionsMenu.hidden) {
      openStatusActionsMenu();
      return;
    }
    closeStatusActionsMenu();
  }

  function syncTerminateButtonState() {
    if (!btnTerminate && !btnRestart && !btnClearOutputs) return;
    const canTerminate = !!currentRunId && isRunBusyStatus(activeRunStatus);
    if (btnTerminate) {
      btnTerminate.disabled = !canTerminate;
      btnTerminate.classList.toggle("is-active", canTerminate);
    }

    const canRestart =
      !!currentRunId &&
      (activeRunStatus === "done" || activeRunStatus === "error" || activeRunStatus === "terminated") &&
      computeModeAndValidation().ok;
    if (btnRestart) {
      btnRestart.disabled = !canRestart;
      btnRestart.classList.toggle("is-active", canRestart);
    }

    const canClearOutputs = !!currentRunId && activeRunStatus === "done";
    if (btnClearOutputs) {
      btnClearOutputs.disabled = !canClearOutputs;
      btnClearOutputs.classList.toggle("is-active", canClearOutputs);
    }
  }

  function syncStartButtonState(validation) {
    const v = validation || computeModeAndValidation();
    if (btnStart) {
      btnStart.textContent = v.startLabel || "Start run";
    }
    btnStart.disabled = !v.ok || isRunBusyStatus(activeRunStatus);
    syncStatusPillVisibility(activeRunStatus || "idle");
    syncTerminateButtonState();
  }

  function render() {
    filelist.innerHTML = "";
    if (serverSelection) {
      filelistMeta.textContent = "Server path";

      const li = document.createElement("li");
      li.className = "fileitem";

      const left = document.createElement("div");
      left.className = "fileleft";

      const name = document.createElement("div");
      name.className = "filename";
      name.textContent = serverSelection.requestedPath || serverSelection.slideFilename || "Server selection";

      const sub = document.createElement("div");
      sub.className = "filesub";
      sub.textContent = `${serverSelection.selectionLabel || "Server selection"} · ${serverSelection.slideFilename || ""}`;

      left.appendChild(name);
      left.appendChild(sub);

      const right = document.createElement("div");
      right.className = "fileright";

      const tag = document.createElement("span");
      tag.className = "tag";
      tag.textContent = "server";

      const rm = document.createElement("button");
      rm.className = "remove icon-btn icon-remove";
      rm.type = "button";
      rm.setAttribute("aria-label", "Clear server selection");
      rm.title = "Clear server selection";
      rm.innerHTML = `
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path d="M4 7h16M9 7V5h6v2m-8 0 1 12h8l1-12M10 11v6M14 11v6" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      `;
      rm.addEventListener("click", clearSelection);

      right.appendChild(tag);
      right.appendChild(rm);
      li.appendChild(left);
      li.appendChild(right);
      filelist.appendChild(li);
    } else {
      const totalBytes = items.reduce((a, b) => a + (b.file.size || 0), 0);
      filelistMeta.textContent = items.length ? `${items.length} file(s) · ${bytesToHuman(totalBytes)}` : "—";

      for (const it of items) {
        const li = document.createElement("li");
        li.className = "fileitem";

        const left = document.createElement("div");
        left.className = "fileleft";

        const name = document.createElement("div");
        name.className = "filename";
        name.textContent = it.relPath || it.file.name;

        const sub = document.createElement("div");
        sub.className = "filesub";
        sub.textContent = `${bytesToHuman(it.file.size || 0)} · ${it.file.type || "application/octet-stream"}`;

        left.appendChild(name);
        left.appendChild(sub);

        const right = document.createElement("div");
        right.className = "fileright";

        const e = extOf(it.relPath || it.file.name);
        const tag = document.createElement("span");
        tag.className = "tag";
        tag.textContent = e || "file";

        const rm = document.createElement("button");
        rm.className = "remove icon-btn icon-remove";
        rm.type = "button";
        rm.setAttribute("aria-label", "Remove file");
        rm.title = "Remove file";
        rm.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
            <path d="M4 7h16M9 7V5h6v2m-8 0 1 12h8l1-12M10 11v6M14 11v6" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"/>
          </svg>
        `;
        rm.addEventListener("click", () => removeItem(it.id));

        right.appendChild(tag);
        right.appendChild(rm);

        li.appendChild(left);
        li.appendChild(right);
        filelist.appendChild(li);
      }
    }

    const v = computeModeAndValidation();
    setPill(modePill, v.level === "idle" ? "idle" : v.level, v.mode);
    validationEl.className = "validation " + (v.level === "good" ? "ok" : v.level === "warn" ? "warn" : v.level === "idle" ? "" : "bad");
    validationEl.textContent = v.msg;

    syncStartButtonState(v);
  }

  async function traverseEntry(entry, pathPrefix = "") {
    return new Promise((resolve) => {
      if (!entry) return resolve([]);
      if (entry.isFile) {
        entry.file((file) => {
          file._relPath = pathPrefix + file.name;
          resolve([file]);
        }, () => resolve([]));
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const all = [];
        const readBatch = () => {
          reader.readEntries(async (entries) => {
            if (!entries || entries.length === 0) return resolve(all);
            for (const e of entries) {
              const sub = await traverseEntry(e, pathPrefix + entry.name + "/");
              all.push(...sub);
            }
            readBatch();
          }, () => resolve(all));
        };
        readBatch();
      } else {
        resolve([]);
      }
    });
  }

  async function handleDrop(ev) {
    ev.preventDefault();
    dropzone.classList.remove("dragover");

    const dt = ev.dataTransfer;
    if (!dt) return;

    const gotItems = dt.items && dt.items.length > 0;
    if (!gotItems) {
      addFiles(dt.files);
      return;
    }

    const supportsEntry = typeof dt.items[0].webkitGetAsEntry === "function";
    if (!supportsEntry) {
      addFiles(dt.files);
      return;
    }

    const collected = [];
    for (const item of dt.items) {
      const entry = item.webkitGetAsEntry && item.webkitGetAsEntry();
      if (!entry) continue;
      const files = await traverseEntry(entry, "");
      collected.push(...files);
    }

    serverSelection = null;
    for (const f of collected) {
      const rel = f._relPath || f.name;
      items.push({ file: f, relPath: rel, id: uniqId() });
    }
    resetUploadInputs();
    render();
  }

  function setExplorerError(message) {
    if (!explorerError) return;
    const text = String(message || "").trim();
    explorerError.textContent = text;
    explorerError.hidden = !text;
  }

  function setExplorerBusyState(busy) {
    explorerBusy = !!busy;
    if (explorerRootSelect) explorerRootSelect.disabled = explorerBusy;
    if (explorerUpBtn) explorerUpBtn.disabled = explorerBusy || !explorerParentPathValue;
    if (explorerRefreshBtn) explorerRefreshBtn.disabled = explorerBusy || !explorerCurrentPathValue;
    if (explorerUseFolderBtn) explorerUseFolderBtn.disabled = explorerBusy || !explorerCurrentPathValue;
    if (explorerUseFileBtn) explorerUseFileBtn.disabled = explorerBusy || !explorerSelectedFilePath;
  }

  function updateExplorerSelectionPreview() {
    if (!explorerSelectionPreview) return;
    if (explorerSelectedFilePath) {
      explorerSelectionPreview.textContent = `Selected file: ${explorerSelectedFileName || explorerSelectedFilePath}`;
      return;
    }
    if (explorerCurrentPathValue) {
      explorerSelectionPreview.textContent = `Current folder: ${explorerCurrentPathValue}`;
      return;
    }
    explorerSelectionPreview.textContent = "Choose a file or navigate into a MIRAX folder.";
  }

  function setExplorerSelectedFile(entry) {
    explorerSelectedFilePath = entry && entry.kind === "file" ? (entry.path || "") : "";
    explorerSelectedFileName = entry && entry.kind === "file" ? (entry.name || "") : "";
    renderExplorerList();
    updateExplorerSelectionPreview();
    setExplorerBusyState(explorerBusy);
  }

  function renderExplorerBreadcrumbs() {
    if (!explorerBreadcrumbs) return;
    explorerBreadcrumbs.innerHTML = "";
    for (let i = 0; i < explorerBreadcrumbItems.length; i++) {
      const crumb = explorerBreadcrumbItems[i];
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "explorer-crumb" + (i === explorerBreadcrumbItems.length - 1 ? " current" : "");
      btn.textContent = crumb.label || crumb.path || "Path";
      btn.addEventListener("click", () => loadExplorerPath(crumb.path));
      explorerBreadcrumbs.appendChild(btn);
    }
  }

  function renderExplorerList() {
    if (!explorerList) return;
    explorerList.innerHTML = "";
    if (explorerEmpty) {
      explorerEmpty.hidden = explorerEntries.length > 0;
    }

    for (const entry of explorerEntries) {
      const li = document.createElement("li");
      li.className = "explorer-entry";
      if (entry.kind === "file" && entry.path === explorerSelectedFilePath) {
        li.classList.add("is-selected");
      }

      const main = document.createElement("div");
      main.className = "explorer-entry-main";

      const icon = document.createElement("span");
      icon.className = `explorer-entry-icon ${entry.kind}`;
      icon.textContent = entry.kind === "dir" ? "D" : "F";

      const text = document.createElement("div");
      text.className = "explorer-entry-text";

      const name = document.createElement("div");
      name.className = "explorer-entry-name";
      name.textContent = entry.name || entry.path || "Entry";

      const sub = document.createElement("div");
      sub.className = "explorer-entry-sub";
      if (entry.kind === "dir") {
        sub.textContent = entry.hint || "Directory";
      } else {
        const parts = [];
        if (entry.slide_kind === "mirax") {
          parts.push("MIRAX file");
        } else {
          parts.push(entry.ext || "Slide file");
        }
        if (Number.isFinite(entry.size_bytes)) {
          parts.push(bytesToHuman(entry.size_bytes));
        }
        sub.textContent = parts.join(" · ");
      }

      text.appendChild(name);
      text.appendChild(sub);
      main.appendChild(icon);
      main.appendChild(text);

      const actions = document.createElement("div");
      actions.className = "explorer-entry-actions";

      if (entry.kind === "dir") {
        li.addEventListener("click", () => loadExplorerPath(entry.path));

        const openBtn = document.createElement("button");
        openBtn.type = "button";
        openBtn.className = "ghost explorer-entry-btn";
        openBtn.textContent = "Open";
        openBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          loadExplorerPath(entry.path);
        });
        actions.appendChild(openBtn);
      } else {
        li.addEventListener("click", () => setExplorerSelectedFile(entry));
        li.addEventListener("dblclick", () => resolveExplorerSelection(entry.path));

        const selectBtn = document.createElement("button");
        selectBtn.type = "button";
        selectBtn.className = "ghost explorer-entry-btn";
        selectBtn.textContent = entry.path === explorerSelectedFilePath ? "Selected" : "Select";
        selectBtn.addEventListener("click", (ev) => {
          ev.stopPropagation();
          setExplorerSelectedFile(entry);
        });
        actions.appendChild(selectBtn);
      }

      li.appendChild(main);
      li.appendChild(actions);
      explorerList.appendChild(li);
    }
  }

  async function apiServerRoots() {
    const res = await fetch("/api/server_fs/roots", { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function apiServerList(path) {
    const url = `/api/server_fs/list?path=${encodeURIComponent(path)}`;
    const res = await fetch(url, { cache: "no-store" });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function apiResolveServerSelection(path) {
    const res = await fetch("/api/server_fs/resolve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function apiAttachServerSelection(runId, path) {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/select_server_path`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function loadExplorerRoots(preferredPath = "") {
    const data = await apiServerRoots();
    explorerRoots = Array.isArray(data.roots) ? data.roots : [];

    if (explorerRootSelect) {
      explorerRootSelect.innerHTML = "";
      for (const root of explorerRoots) {
        const opt = document.createElement("option");
        opt.value = root.path;
        opt.textContent = root.exists ? root.path : `${root.path} (unavailable)`;
        opt.disabled = !root.exists;
        explorerRootSelect.appendChild(opt);
      }
    }

    if (!explorerRoots.length) {
      throw new Error("No server slide roots are configured on the backend.");
    }

    const existingRoots = explorerRoots.filter((root) => root.exists);
    if (!existingRoots.length) {
      throw new Error("Configured server slide roots are unavailable on this machine.");
    }

    let nextRoot = existingRoots[0].path;
    if (preferredPath) {
      const matched = existingRoots.find((root) => preferredPath === root.path || preferredPath.startsWith(`${root.path}/`));
      if (matched) nextRoot = matched.path;
    }

    if (explorerRootSelect) {
      explorerRootSelect.value = nextRoot;
    }

    return nextRoot;
  }

  async function loadExplorerPath(path) {
    if (!path) return;
    setExplorerBusyState(true);
    setExplorerError("");
    try {
      const data = await apiServerList(path);
      explorerCurrentPathValue = data.current_path || path;
      explorerParentPathValue = data.parent_path || "";
      explorerEntries = Array.isArray(data.entries) ? data.entries : [];
      explorerBreadcrumbItems = Array.isArray(data.breadcrumbs) ? data.breadcrumbs : [];
      if (!explorerSelectedFilePath.startsWith(`${explorerCurrentPathValue}/`)) {
        explorerSelectedFilePath = "";
        explorerSelectedFileName = "";
      }
      if (explorerCurrentPath) {
        explorerCurrentPath.textContent = explorerCurrentPathValue;
      }
      if (explorerRootSelect && data.root_path) {
        explorerRootSelect.value = data.root_path;
      }
      renderExplorerBreadcrumbs();
      renderExplorerList();
      updateExplorerSelectionPreview();
    } catch (e) {
      explorerEntries = [];
      explorerBreadcrumbItems = [];
      renderExplorerBreadcrumbs();
      renderExplorerList();
      setExplorerError(String(e && e.message ? e.message : e));
    } finally {
      setExplorerBusyState(false);
    }
  }

  async function resolveExplorerSelection(path) {
    if (!path) return;
    setExplorerBusyState(true);
    setExplorerError("");
    try {
      const resolved = await apiResolveServerSelection(path);
      setServerSelection({
        requestedPath: resolved.requested_path,
        slidePath: resolved.slide_path,
        slideFilename: resolved.slide_filename,
        selectionKind: resolved.selection_kind,
        selectionLabel: resolved.selection_label,
      });
      closeExplorer();
      runLabelEl.textContent = `Selected ${resolved.slide_filename} from server.`;
    } catch (e) {
      setExplorerError(String(e && e.message ? e.message : e));
    } finally {
      setExplorerBusyState(false);
    }
  }

  async function openExplorer() {
    if (!explorerModal) return;
    explorerModal.hidden = false;
    setExplorerError("");
    explorerEntries = [];
    renderExplorerList();

    try {
      const rawPreferredPath = (serverSelection && serverSelection.requestedPath) || explorerCurrentPathValue || "";
      const lastSlash = rawPreferredPath.lastIndexOf("/");
      const preferredPath = rawPreferredPath && extOf(rawPreferredPath) && lastSlash > 0
        ? rawPreferredPath.slice(0, lastSlash)
        : rawPreferredPath;
      const rootPath = await loadExplorerRoots(preferredPath);
      const initialPath = preferredPath || rootPath;
      await loadExplorerPath(initialPath);
    } catch (e) {
      setExplorerError(String(e && e.message ? e.message : e));
    }
  }

  function closeExplorer() {
    if (!explorerModal) return;
    explorerModal.hidden = true;
    setExplorerError("");
  }

  function upsertLiveStep(stepId, title, subText) {
    if (!stepsEl) return;
    const existing = document.getElementById(stepId);
    if (!title) {
      if (existing) existing.remove();
      return;
    }

    let li = existing;
    if (!li) {
      li = document.createElement("li");
      li.id = stepId;
    }
    // Re-apply class every update so stale DOM from older versions still gets LIVE styling.
    li.className = "logitem live-step-item";
    // Inline fallback so LIVE remains orange even if CSS is cached/stale.
    li.style.borderLeft = "4px solid #ff7a00";
    li.style.background = "var(--bg)";
    li.style.boxShadow = "none";

    li.innerHTML = "";
    const t = document.createElement("div");
    t.className = "logtitle";
    t.textContent = `LIVE. ${title}`;
    t.style.color = "#ff9f3d";
    li.appendChild(t);

    if (subText) {
      const s = document.createElement("div");
      s.className = "logsub";
      s.textContent = subText;
      s.style.color = "#ffbe87";
      li.appendChild(s);
    }

    stepsEl.appendChild(li);
  }

  function upsertLiveStatusStep(title, subText) {
    upsertLiveStep("step-live-status", title, subText);
  }

  function resetRunUI() {
    setPill(statusPill, "idle", "Idle");
    runIdEl.textContent = "—";
    runLabelEl.textContent = "";
    slideNameEl.textContent = "—";
    terminateRequested = false;
    activeUploadXhr = null;
    closeStatusActionsMenu();
    currentReportHref = null;
    reportFetchToken += 1;
    activeRunStatus = "";
    syncStatusPillVisibility("idle");
    lastCurrentViewState = null;
    searchTargetBoxPx = null;
    searchDrawBoxPx = null;
    searchTransition = null;
    searchPulse = 0;
    searchDashOffset = 0;
    if (overlayRafId !== null) {
      cancelAnimationFrame(overlayRafId);
      overlayRafId = null;
    }

    errorBox.hidden = true;
    errorText.textContent = "";

    overviewImg.hidden = true;
    overviewImg.src = "";
    baseOverviewImageUrl = "";
    setOverviewEmptyState(OVERVIEW_EMPTY_TEXT);
    overviewCacheState = null;
    selectedOverviewRoiId = null;
    roiById.clear();
    if (overviewCanvas) {
      const ctx = overviewCanvas.getContext("2d");
      ctx.clearRect(0, 0, overviewCanvas.width, overviewCanvas.height);
    }

    clearDarkRegions();

    renderFinalReportMarkdown("Waiting for model output…", true);
    setReasoningContent("");
    reportLink.textContent = "";
    stepsEl.innerHTML = "";
    roisEl.innerHTML = "";

    lastRenderedStep = 0;
    lastRenderedRoi = 0;
    darkRegionsLoaded = false;
    // Keep dark regions enabled state, just reset loaded state
    if (darkRegionsEnabled) {
      clearDarkRegions();
    }
  }

  function appendLogItem(listEl, title, subText, imgUrl) {
    const li = document.createElement("li");
    li.className = "logitem";

    const t = document.createElement("div");
    t.className = "logtitle";
    t.textContent = title;

    const s = document.createElement("div");
    s.className = "logsub";
    s.textContent = subText || "";

    li.appendChild(t);
    if (subText) li.appendChild(s);

    if (imgUrl) {
      const img = document.createElement("img");
      img.className = "logimg";
      img.alt = title;
      li.appendChild(img);
      setImageSrcWithReveal(img, imgUrl, { revealItem: true });
    }

    listEl.appendChild(li);
    listEl.scrollTop = listEl.scrollHeight;
  }

  function _ensureRoiLoadingBadge(li) {
    let badge = li.querySelector(".roi-loading-badge");
    if (!badge) {
      badge = document.createElement("span");
      badge.className = "roi-loading-badge";
      badge.setAttribute("aria-hidden", "true");
      li.appendChild(badge);
    }
    return badge;
  }

  function _bindRoiImageLoading(li, img, keepListPinned = false) {
    _ensureRoiLoadingBadge(li);
    li.classList.add("roi-loading");

    const done = () => {
      li.classList.remove("roi-loading");
      if (keepListPinned && li.parentElement === roisEl) {
        maybeScrollRoiListToBottom();
      }
    };
    if (img.complete && img.naturalWidth > 0) {
      done();
      return;
    }
    img.addEventListener("load", done, { once: true });
    img.addEventListener("error", done, { once: true });
  }

  function _finishImageReveal(img, token, item) {
    if (!img) return;
    if (token && img.dataset.revealToken !== token) return;
    requestAnimationFrame(() => {
      if (token && img.dataset.revealToken !== token) return;
      img.classList.remove("is-revealing");
      if (item) item.classList.remove("is-revealing");
    });
  }

  function setImageSrcWithReveal(img, src, options = {}) {
    if (!img) return;
    const nextSrc = String(src || "");
    const item = options.revealItem ? img.closest(".logitem") : null;

    if (!nextSrc) {
      img.dataset.revealSrc = "";
      img.dataset.revealToken = "";
      img.classList.remove("is-revealing");
      if (item) item.classList.remove("is-revealing");
      img.removeAttribute("src");
      return;
    }

    const prevSrc = img.dataset.revealSrc || "";
    const currentSrc = img.getAttribute("src") || "";
    const shouldAnimate = !prevSrc && !currentSrc;
    if (prevSrc === nextSrc && currentSrc === nextSrc) {
      if (img.complete && img.naturalWidth > 0) {
        img.classList.remove("is-revealing");
        if (item) item.classList.remove("is-revealing");
      }
      return;
    }

    img.dataset.revealSrc = nextSrc;
    if (!shouldAnimate) {
      img.dataset.revealToken = "";
      img.classList.remove("is-revealing");
      if (item) item.classList.remove("is-revealing");
      img.src = nextSrc;
      return;
    }

    const token = `reveal-${++imageRevealSeq}`;
    img.dataset.revealToken = token;
    img.classList.add("is-revealing");

    if (item) {
      item.classList.remove("is-revealing");
      void item.offsetWidth;
      item.classList.add("is-revealing");
    }

    const done = () => _finishImageReveal(img, token, item);
    img.addEventListener("load", done, { once: true });
    img.addEventListener("error", done, { once: true });
    img.src = nextSrc;

    if (img.complete && img.naturalWidth > 0) {
      done();
    }
  }

  function appendRoiItem(roi, title, subText, imgUrl) {
    const li = document.createElement("li");
    li.className = "logitem roi-item";
    li.dataset.roiId = String(roi.roi_id);
    li.dataset.roiKey = `${roi.roi_id}:${String(roi.debug_path || "")}`;
    const roiColor = roiColorForId(roi.roi_id);
    li.style.boxShadow = `inset 3px 0 0 ${roiColor}`;

    const t = document.createElement("div");
    t.className = "logtitle";
    t.textContent = title;
    t.style.color = roiColor;

    li.appendChild(t);
    if (subText) {
      const s = document.createElement("div");
      s.className = "logsub";
      s.textContent = subText;
      li.appendChild(s);
    }

    if (imgUrl) {
      const img = document.createElement("img");
      img.className = "logimg";
      img.alt = title;
      li.appendChild(img);
      _bindRoiImageLoading(li, img, true);
      setImageSrcWithReveal(img, imgUrl, { revealItem: true });
    }

    roisEl.appendChild(li);
    maybeScrollRoiListToBottom();
  }

  function clearRoiItemsFromList() {
    const nodes = roisEl.querySelectorAll(".roi-item");
    for (const node of nodes) node.remove();
  }

  function setSelectedRoiInList() {
    const nodes = roisEl.querySelectorAll(".roi-item");
    for (const node of nodes) {
      const roiId = Number(node.dataset.roiId || "");
      node.classList.toggle("roi-item-selected", roiId === selectedOverviewRoiId);
    }
  }

  function _overviewBaseDims(cache) {
    if (!cache) return null;
    const bw = Number(cache.base_w0);
    const bh = Number(cache.base_h0);
    if (Number.isFinite(bw) && Number.isFinite(bh) && bw > 0 && bh > 0) {
      return { baseW: bw, baseH: bh };
    }
    const lw = Number(cache.level_w);
    const lh = Number(cache.level_h);
    const ds = Number(cache.level_downsample);
    if (Number.isFinite(lw) && Number.isFinite(lh) && Number.isFinite(ds) && lw > 0 && lh > 0 && ds > 0) {
      return { baseW: lw * ds, baseH: lh * ds };
    }
    return null;
  }

  function _mapLevel0BboxToOverviewPx(bbox) {
    if (!Array.isArray(bbox) || bbox.length < 4) return null;
    if (!overviewImg || overviewImg.hidden || !overviewImg.src) return null;
    const imgW = overviewImg.naturalWidth || overviewImg.width || 0;
    const imgH = overviewImg.naturalHeight || overviewImg.height || 0;
    if (!imgW || !imgH) return null;

    const dims = _overviewBaseDims(overviewCacheState);
    if (!dims) return null;

    const outW = Number(overviewCacheState && overviewCacheState.shown_w) || imgW;
    const outH = Number(overviewCacheState && overviewCacheState.shown_h) || imgH;
    const sx = outW > 0 ? imgW / outW : 1.0;
    const sy = outH > 0 ? imgH / outH : 1.0;

    const x0 = Number(bbox[0]);
    const y0 = Number(bbox[1]);
    const w0 = Number(bbox[2]);
    const h0 = Number(bbox[3]);
    if (![x0, y0, w0, h0].every(Number.isFinite)) return null;

    return {
      x: Math.round((x0 / dims.baseW) * outW * sx),
      y: Math.round((y0 / dims.baseH) * outH * sy),
      w: Math.max(2, Math.round((w0 / dims.baseW) * outW * sx)),
      h: Math.max(2, Math.round((h0 / dims.baseH) * outH * sy)),
    };
  }

  function _lerpBox(a, b, t) {
    return {
      x: a.x + (b.x - a.x) * t,
      y: a.y + (b.y - a.y) * t,
      w: a.w + (b.w - a.w) * t,
      h: a.h + (b.h - a.h) * t,
    };
  }

  function _isSearchActive() {
    return new Set(["created", "uploading", "pending", "running"]).has(activeRunStatus) && !!searchTargetBoxPx;
  }

  function _startOverlayLoop() {
    if (overlayRafId !== null) return;
    overlayRafId = requestAnimationFrame(_overlayFrame);
  }

  function _overlayFrame(ts) {
    overlayRafId = null;
    let keepRunning = false;

    if (searchTransition) {
      const elapsed = ts - searchTransition.start;
      const t = Math.max(0, Math.min(1, elapsed / searchTransition.duration));
      searchDrawBoxPx = _lerpBox(searchTransition.from, searchTransition.to, t);
      if (t < 1) {
        keepRunning = true;
      } else {
        searchDrawBoxPx = { ...searchTransition.to };
        searchTransition = null;
      }
    }

    if (_isSearchActive()) {
      searchPulse = 0.35 + (Math.sin(ts / 180) + 1) * 0.25;
      searchDashOffset = (ts / 25) % 24;
      keepRunning = true;
    }

    renderOverviewRoiOverlay();
    if (keepRunning) _startOverlayLoop();
  }

  function updateSearchingBox(currentView, runStatus) {
    activeRunStatus = runStatus || "";
    lastCurrentViewState = currentView || null;
    const statusActive = new Set(["created", "uploading", "pending", "running"]).has(activeRunStatus);

    if (!statusActive) {
      searchTargetBoxPx = null;
      searchDrawBoxPx = null;
      searchTransition = null;
      renderOverviewRoiOverlay();
      return;
    }

    if (!_isSearchActive() && !currentView) {
      searchTargetBoxPx = null;
      searchDrawBoxPx = null;
      searchTransition = null;
      renderOverviewRoiOverlay();
      return;
    }

    if (!currentView || !Number.isFinite(Number(currentView.x0)) || !Number.isFinite(Number(currentView.y0)) ||
        !Number.isFinite(Number(currentView.w)) || !Number.isFinite(Number(currentView.h))) {
      searchTargetBoxPx = null;
      searchDrawBoxPx = null;
      searchTransition = null;
      renderOverviewRoiOverlay();
      return;
    }

    const mapped = _mapLevel0BboxToOverviewPx([currentView.x0, currentView.y0, currentView.w, currentView.h]);
    if (!mapped) return;

    searchTargetBoxPx = mapped;
    if (!searchDrawBoxPx) {
      searchDrawBoxPx = { ...mapped };
      searchTransition = null;
      renderOverviewRoiOverlay();
      if (_isSearchActive()) _startOverlayLoop();
      return;
    }

    const dx = Math.abs(searchDrawBoxPx.x - mapped.x);
    const dy = Math.abs(searchDrawBoxPx.y - mapped.y);
    const dw = Math.abs(searchDrawBoxPx.w - mapped.w);
    const dh = Math.abs(searchDrawBoxPx.h - mapped.h);
    if (dx + dy + dw + dh < 2) {
      if (_isSearchActive()) _startOverlayLoop();
      return;
    }

    searchTransition = {
      from: { ...searchDrawBoxPx },
      to: { ...mapped },
      start: performance.now(),
      duration: 480,
    };
    _startOverlayLoop();
  }

  function renderOverviewRoiOverlay() {
    if (!overviewCanvas) return;
    const ctx = overviewCanvas.getContext("2d");
    if (!overviewImg || overviewImg.hidden || !overviewImg.src) {
      ctx.clearRect(0, 0, overviewCanvas.width, overviewCanvas.height);
      return;
    }
    const imgW = overviewImg.naturalWidth || overviewImg.width || 0;
    const imgH = overviewImg.naturalHeight || overviewImg.height || 0;
    if (!imgW || !imgH) return;

    if (overviewCanvas.width !== imgW || overviewCanvas.height !== imgH) {
      overviewCanvas.width = imgW;
      overviewCanvas.height = imgH;
    }
    ctx.clearRect(0, 0, overviewCanvas.width, overviewCanvas.height);

    if (darkRegionsEnabled && darkRegionsLoaded) {
      const darkW = (darkImg && (darkImg.naturalWidth || darkImg.width)) || 0;
      const darkH = (darkImg && (darkImg.naturalHeight || darkImg.height)) || 0;
      const canDrawMask = !!(darkMaskMode && darkImg && darkImg.src && darkW && darkH);

      if (canDrawMask) {
        try {
          ctx.save();
          ctx.fillStyle = "rgba(0,0,0,0.45)";
          ctx.fillRect(0, 0, overviewCanvas.width, overviewCanvas.height);
          ctx.globalCompositeOperation = "destination-out";
          ctx.drawImage(darkImg, 0, 0, overviewCanvas.width, overviewCanvas.height);
        } catch (e) {
          // ignore drawing errors
        } finally {
          ctx.restore();
        }

        ctx.save();
        ctx.fillStyle = "rgba(124,240,193,0.98)";
        ctx.font = "bold 11px sans-serif";
        ctx.fillText("Dark-region refined mask", 10, 18);
        ctx.restore();
      } else if (darkImg && darkImg.src && darkBoxes.length) {
        ctx.save();
        ctx.lineWidth = Math.max(1, Math.round(Math.min(imgW, imgH) * 0.003));
        ctx.strokeStyle = "rgba(124,240,193,0.92)";
        ctx.setLineDash([10, 7]);
        for (const b of darkBoxes) {
          if (!b || !Number.isFinite(Number(b.x)) || !Number.isFinite(Number(b.y)) ||
              !Number.isFinite(Number(b.w)) || !Number.isFinite(Number(b.h))) {
            continue;
          }
          ctx.strokeRect(Number(b.x), Number(b.y), Number(b.w), Number(b.h));
        }
        ctx.setLineDash([]);
        ctx.fillStyle = "rgba(124,240,193,0.98)";
        ctx.font = "bold 11px sans-serif";
        ctx.fillText("Dark-region heuristic", 10, 18);
        try {
          ctx.save();
          ctx.fillStyle = "rgba(0,0,0,0.45)";
          ctx.fillRect(0, 0, overviewCanvas.width, overviewCanvas.height);
          ctx.globalCompositeOperation = "destination-out";
          for (const b of darkBoxes) {
            if (!b || !Number.isFinite(Number(b.x)) || !Number.isFinite(Number(b.y)) ||
                !Number.isFinite(Number(b.w)) || !Number.isFinite(Number(b.h))) {
              continue;
            }
            ctx.fillRect(Number(b.x), Number(b.y), Number(b.w), Number(b.h));
          }
        } catch (e) {
          // ignore drawing errors
        } finally {
          ctx.restore();
        }
        ctx.restore();
      }
    }

    const roiEntries = Array.from(roiById.values())
      .filter((roi) => roi && Number.isFinite(Number(roi.roi_id)))
      .sort((a, b) => Number(a.roi_id) - Number(b.roi_id));
    for (const roi of roiEntries) {
      const mapped = _mapLevel0BboxToOverviewPx(roi.view_bbox_level0);
      if (!mapped) continue;
      const color = roiColorForId(roi.roi_id);
      const isSelected = Number(roi.roi_id) === Number(selectedOverviewRoiId);
      ctx.save();
      const strokeWidth = Math.max(
        isSelected ? 2 : 1,
        Math.round(Math.min(imgW, imgH) * (isSelected ? 0.0032 : 0.0024)),
      );
      ctx.strokeStyle = "rgba(0,0,0,0.95)";
      ctx.lineWidth = strokeWidth + 2;
      ctx.strokeRect(mapped.x, mapped.y, mapped.w, mapped.h);
      ctx.strokeStyle = "rgba(255,255,255,0.92)";
      ctx.lineWidth = strokeWidth + 0.8;
      ctx.strokeRect(mapped.x, mapped.y, mapped.w, mapped.h);
      ctx.strokeStyle = color;
      ctx.lineWidth = strokeWidth;
      ctx.strokeRect(mapped.x, mapped.y, mapped.w, mapped.h);
      ctx.font = `bold ${isSelected ? 12 : 11}px sans-serif`;
      const label = `ROI ${roi.roi_id}`;
      const labelPadX = 4;
      const labelPadY = 2;
      const textM = ctx.measureText(label);
      const labelW = Math.ceil(textM.width) + labelPadX * 2;
      const labelH = (isSelected ? 12 : 11) + labelPadY * 2;
      const labelX = mapped.x + 2;
      const labelY = Math.max(0, mapped.y - labelH - 4);
      ctx.fillStyle = "rgba(0,0,0,0.72)";
      ctx.fillRect(labelX, labelY, labelW, labelH);
      ctx.fillStyle = color;
      ctx.fillText(label, labelX + labelPadX, labelY + labelH - labelPadY - 1);
      ctx.restore();
    }

    if (_isSearchActive() && searchDrawBoxPx) {
      ctx.save();
      ctx.setLineDash([8, 6]);
      ctx.lineDashOffset = -searchDashOffset;
      ctx.strokeStyle = `rgba(221, 122, 0, ${Math.max(0.7, searchPulse)})`;
      ctx.lineWidth = Math.max(2, Math.round(Math.min(imgW, imgH) * 0.0036));
      ctx.strokeRect(searchDrawBoxPx.x, searchDrawBoxPx.y, searchDrawBoxPx.w, searchDrawBoxPx.h);
      ctx.setLineDash([]);
      ctx.fillStyle = "rgba(228, 126, 6, 0.98)";
      ctx.font = "bold 11px sans-serif";
      ctx.fillText("Searching ROI…", searchDrawBoxPx.x + 4, Math.max(14, searchDrawBoxPx.y - 6));
      ctx.restore();
    }
  }

  function selectOverviewRoi(roiId) {
    if (!roiById.has(roiId)) return;
    selectedOverviewRoiId = roiId;
    setSelectedRoiInList();
    renderOverviewRoiOverlay();
  }

  function upsertLiveRoiItem(currentView, runStatus) {
    const existing = document.getElementById("roi-live-item");
    const runningStates = new Set(["created", "uploading", "pending", "running"]);
    const hasLive = currentView && currentView.image_url && runningStates.has(runStatus);

    if (!hasLive) {
      if (existing) existing.remove();
      return;
    }

    const nextRoiId = lastRenderedRoi + 1;
    const title = `ROI ${nextRoiId} (searching...)`;
    const parts = [];
    if (currentView.field_width_um && currentView.field_height_um) {
      parts.push(`Field ~${currentView.field_width_um.toFixed(0)}×${currentView.field_height_um.toFixed(0)} µm`);
    }
    if (currentView.tissue_fraction !== undefined && currentView.tissue_fraction !== null) {
      parts.push(`Tissue ${currentView.tissue_fraction.toFixed(2)}`);
    }
    const subText = parts.join(" · ");

    let li = existing;
    if (!li) {
      li = document.createElement("li");
      li.id = "roi-live-item";
      li.className = "logitem live-roi-item";
    }
    li.classList.add("roi-searching");
    // Keep the live preview as the active/latest ROI slot.
    roisEl.appendChild(li);

    let t = li.querySelector(".logtitle");
    if (!t) {
      t = document.createElement("div");
      t.className = "logtitle";
      li.appendChild(t);
    }
    t.textContent = title;

    if (subText) {
      let s = li.querySelector(".logsub");
      if (!s) {
        s = document.createElement("div");
        s.className = "logsub";
        li.appendChild(s);
      }
      s.textContent = subText;
    } else {
      const s = li.querySelector(".logsub");
      if (s) s.remove();
    }

    let img = li.querySelector(".logimg");
    if (!img) {
      img = document.createElement("img");
      img.className = "logimg";
      li.appendChild(img);
    }
    img.alt = title;
    if ((img.dataset.revealSrc || "") !== String(currentView.image_url || "")) {
      _bindRoiImageLoading(li, img);
      setImageSrcWithReveal(img, currentView.image_url, { revealItem: true });
    }
    roisEl.scrollTop = roisEl.scrollHeight;
  }

  // Detect if the LLM's textual final output signals a slide-load failure despite
  // run.status being "done" (e.g. text-only model can't see images and says so).
  const FAILURE_PATTERNS = [
    /unable to (load|open|process|read|view|access)/i,
    /cannot (load|open|process|read|view|access)/i,
    /could not (load|open|process|read|view|access)/i,
    /failed to (load|open|process|read|view|access)/i,
    /no slide (found|available|provided|loaded)/i,
    /slide (not found|could not be loaded|failed to load)/i,
    /file not found/i,
    /image (not|could not be|was not) (found|loaded|processed|provided)/i,
    /error (loading|opening|reading) (the |this )?(slide|image|file)/i,
  ];

  function finalOutputIndicatesFailure(text) {
    if (!text || typeof text !== "string") return false;
    return FAILURE_PATTERNS.some((re) => re.test(text));
  }

  function setModelStatus(status, modelName) {
    if (!runLoadingEl) return;
    let label = (modelName && String(modelName).trim()) || currentModelName || "Model";
    if ((!status || status === "idle") && modelSelect && modelSelect.value) {
      // For pre-run/idle UI, always reflect currently selected model.
      label = modelSelect.value;
    }
    if (runStateTextEl) runStateTextEl.textContent = label;

    runLoadingEl.classList.remove("model-good", "model-warn", "model-bad", "model-idle", "is-active");

    if (!status || status === "idle") {
      // Idle with an available model should read as ready.
      runLoadingEl.classList.add("model-good");
      runLoadingEl.hidden = false;
      return;
    }

    // "done" but LLM output indicates a slide/image load failure → show as red (bad)
    if (status === "done-warn") {
      runLoadingEl.classList.add("model-bad");
      runLoadingEl.hidden = false;
      return;
    }

    if (status === "error") {
      runLoadingEl.classList.add("model-bad");
      runLoadingEl.hidden = false;
      return;
    }
    if (status === "terminated") {
      runLoadingEl.classList.add("model-bad");
      runLoadingEl.hidden = false;
      return;
    }
    if (status === "running") {
      runLoadingEl.classList.add("model-warn", "is-active");
      runLoadingEl.hidden = false;
      return;
    }
    if (status === "created" || status === "uploading" || status === "pending") {
      runLoadingEl.classList.add("model-warn", "is-active");
      runLoadingEl.hidden = false;
      return;
    }
    if (status === "done") {
      runLoadingEl.classList.add("model-good");
      runLoadingEl.hidden = false;
      return;
    }

    runLoadingEl.hidden = true;
  }

  function updateLivePrepProgress(run, st) {
    if (!run) return;
    const status = String(run.status || "");
    const prep = (st && st.roi_candidate_prep && typeof st.roi_candidate_prep === "object")
      ? st.roi_candidate_prep
      : null;

    const busy = isRunBusyStatus(status);
    if (!busy) {
      return;
    }

    if (status === "running" && prep && (prep.active || prep.status === "starting" || prep.status === "running")) {
      const parts = [];
      if (prep.message) parts.push(String(prep.message));
      if (prep.phase) parts.push(`phase=${prep.phase}`);
      if (prep.processed_tiles !== undefined && prep.processed_tiles !== null) {
        parts.push(`tiles=${prep.processed_tiles}`);
      }
      if (prep.processed_batches !== undefined && prep.processed_batches !== null) {
        parts.push(`batches=${prep.processed_batches}`);
      }
      upsertLiveStatusStep("Preparing ROI candidates", parts.join(" · "));
      return;
    }

    if (status === "running" && prep && (prep.status === "done" || prep.status === "failed")) {
      upsertLiveStatusStep(
        prep.status === "done" ? "ROI candidates ready" : "ROI candidate prep failed",
        String(prep.message || "")
      );
      return;
    }
  }

  function syncLiveRunStatusStep(status, details) {
    const state = String(status || "").toLowerCase();
    if (!state) return;
    if (state === "created") {
      upsertLiveStatusStep("Creating run", details || "Preparing upload session.");
      return;
    }
    if (state === "uploading") {
      upsertLiveStatusStep("Uploading slide bundle", details || "");
      return;
    }
    if (state === "pending") {
      upsertLiveStatusStep("Run pending", details || "Waiting for worker.");
      return;
    }
    if (state === "running") {
      upsertLiveStatusStep("Run running", details || "Agent is exploring the slide.");
      return;
    }
    if (state === "done") {
      upsertLiveStatusStep("Run done", details || "All steps completed.");
      return;
    }
    if (state === "done-warn") {
      upsertLiveStatusStep("Run done (failed)", details || "Model finished with failure output.");
      return;
    }
    if (state === "terminated") {
      upsertLiveStatusStep("Run terminated", details || "Terminated by user.");
      return;
    }
    if (state === "error") {
      upsertLiveStatusStep("Run error", details || "Run stopped due to an error.");
    }
  }

  async function fetchServiceModelName() {
    try {
      const res = await fetch("/healthz", { cache: "no-store" });
      if (!res.ok) return;
      const data = await res.json();
      const name = data && typeof data.model_name === "string" ? data.model_name.trim() : "";
      if (!name) return;
      if (modelSelect && modelSelect.options.length) {
        const hasOption = Array.from(modelSelect.options).some((opt) => opt.value === name);
        if (hasOption && !modelSelectionTouched) {
          modelSelect.value = name;
        }
        currentModelName = modelSelect.value || name;
      } else {
        currentModelName = name;
      }
      setModelStatus(activeRunStatus || "idle", currentModelName);
    } catch (_e) {
      // Keep fallback label if health check is unavailable.
    }
  }

  async function pollRun() {
    if (!currentRunId) return;
    try {
      const res = await fetch(`/api/runs/${currentRunId}`);
      if (!res.ok) return;
      const data = await res.json();
      const run = data.run;
      const st = data.wsi_state;
      const tilePrefilterMethod =
        (run && typeof run.tile_prefilter_method === "string" && run.tile_prefilter_method)
          ? String(run.tile_prefilter_method)
          : ((st && typeof st.tile_prefilter_method === "string" && st.tile_prefilter_method) ? String(st.tile_prefilter_method) : null);
      if (tilePrefilterMethodSelect && tilePrefilterMethod) {
        tilePrefilterMethodSelect.value = tilePrefilterMethod;
        saveStoredValue(tilePrefilterMethodStorageKey, tilePrefilterMethod);
      }
      activeRunStatus = run.status || "";
      lastCurrentViewState = (st && st.current_view) ? st.current_view : null;
      overviewCacheState = (st && st.overview_cache) ? st.overview_cache : null;

      slideNameEl.textContent = run.slide_filename || "—";

      if (run.status === "created") setPill(statusPill, "run", "Created");
      if (run.status === "uploading") setPill(statusPill, "run", "Uploading");
      if (run.status === "pending") setPill(statusPill, "run", "Pending");
      if (run.status === "running") setPill(statusPill, "run", "Running");
      // Detect LLM-reported slide/image load failure in final output.
      const hasFinalFailure = run.status === "done" && finalOutputIndicatesFailure(run.final_output || "");
      const effectiveStatus = hasFinalFailure ? "done-warn" : run.status;

      if (run.status === "done" && hasFinalFailure) setPill(statusPill, "bad", "Done (failed)");
      else if (run.status === "done") setPill(statusPill, "good", "Done");
      if (run.status === "error") setPill(statusPill, "bad", "Error");
      if (run.status === "terminated") setPill(statusPill, "bad", "Terminated");
      syncStatusPillVisibility(run.status || "idle");
      const backendModelName = run.model_name ? String(run.model_name).trim() : "";
      if (isRunBusyStatus(run.status) && runRequestedModel) {
        currentModelName = runRequestedModel;
        if (backendModelName && backendModelName !== runRequestedModel) {
          runLabelEl.textContent = `Requested ${runRequestedModel}, backend returned ${backendModelName}.`;
        }
      } else if (backendModelName) {
        currentModelName = backendModelName;
      }
      setModelStatus(effectiveStatus, currentModelName);
      if (run.status === "terminated") {
        terminateRequested = true;
        runLabelEl.textContent = "Run terminated by user.";
      }

      if (run.status === "error") {
        errorBox.hidden = false;
        let msg = "";
        if (run.error_message) msg += "Error: " + run.error_message + "\n";
        if (run.traceback) msg += "\nTraceback:\n" + run.traceback;
        if (!msg) msg = "Unknown error (no message provided).";
        errorText.textContent = msg;
      } else {
        errorBox.hidden = true;
        errorText.textContent = "";
      }

      const newOverviewUrl = (st && st.overview_image_url) ? st.overview_image_url : "";
      const overviewUrlChanged = baseOverviewImageUrl !== newOverviewUrl;
      baseOverviewImageUrl = newOverviewUrl;
      applyOverviewDisplaySource();
      // Re-fetch dark regions when overview image changes (if enabled)
      if (overviewUrlChanged && darkRegionsEnabled && currentRunId) {
        darkRegionsLoaded = false;  // Reset to allow re-fetch
        fetchDarkRegions(currentRunId);
      }
      updateSearchingBox(lastCurrentViewState, run.status);

      const hasReportPath = !!run.report_path;
      if (!hasReportPath) {
        if (run.final_output) {
          renderFinalReportMarkdown(extractFinalReportOnly(run.final_output), false);
        } else if (run.status === "running" || run.status === "pending" || run.status === "created" || run.status === "uploading") {
          renderFinalReportMarkdown("Running…", true);
        } else if (run.status === "terminated") {
          renderFinalReportMarkdown("Run was terminated before completion.", false);
        } else {
          renderFinalReportMarkdown("", false);
        }
      }

      if (run.reasoning_content) setReasoningContent(run.reasoning_content);
      else setReasoningContent("");

      if (run.report_path) {
        const relPath = run.report_path.replace(/.*outputs[\\/]/, "");
        const href = `/reports/${relPath}`;
        reportLink.innerHTML = `Report: <a href="${href}" target="_blank" rel="noreferrer">open Markdown report</a>`;
        if (href !== currentReportHref) {
          currentReportHref = href;
          fetchAndRenderReport(href, run.final_output || "");
        }
      } else {
        reportLink.textContent = "";
        currentReportHref = null;
      }

      let incomingSteps = [];
      if (st && Array.isArray(st.step_log)) {
        incomingSteps = st.step_log
          .filter((step) => step && Number.isFinite(Number(step.step_index)))
          .sort((a, b) => Number(a.step_index) - Number(b.step_index));
        const maxIncomingStep = incomingSteps.length
          ? Number(incomingSteps[incomingSteps.length - 1].step_index)
          : 0;
        if (maxIncomingStep < lastRenderedStep) {
          stepsEl.innerHTML = "";
          lastRenderedStep = 0;
        }
        for (const step of incomingSteps) {
          if (step.step_index > lastRenderedStep) {
            const parts = [];
            if (step.nav_reason) parts.push(step.nav_reason);
            if (step.field_width_um && step.field_height_um) parts.push(`Field ~${step.field_width_um.toFixed(0)}×${step.field_height_um.toFixed(0)} µm`);
            if (step.tissue_fraction !== undefined && step.tissue_fraction !== null) parts.push(`Tissue ${step.tissue_fraction.toFixed(2)}`);
            if (step.roi_candidate_stage) parts.push(`ROI stage: ${step.roi_candidate_stage}`);
            if (step.roi_candidate_source) parts.push(`Source: ${step.roi_candidate_source}`);
            if (step.roi_candidate_count !== undefined && step.roi_candidate_count !== null) {
              parts.push(`Top-K: ${step.roi_candidate_count}`);
            }
            if (step.roi_candidate_pipeline) parts.push(step.roi_candidate_pipeline);
            if (step.roi_candidate_warning) parts.push(`Warning: ${step.roi_candidate_warning}`);
            if (step.roi_candidate_index_meta && typeof step.roi_candidate_index_meta === "object") {
              const meta = step.roi_candidate_index_meta;
              const extractor = meta.extractor_id ? String(meta.extractor_id) : null;
              const tiles = Number.isFinite(Number(meta.num_tiles)) ? Number(meta.num_tiles) : null;
              const dim = Number.isFinite(Number(meta.feature_dim)) ? Number(meta.feature_dim) : null;
              const bits = [];
              if (extractor) bits.push(`extractor=${extractor}`);
              if (tiles !== null) bits.push(`tiles=${tiles}`);
              if (dim !== null) bits.push(`dim=${dim}`);
              if (bits.length) parts.push(bits.join(", "));
            }
            appendLogItem(stepsEl, `S${step.step_index}. ${String(step.tool || "")}`, parts.join(" · "), step.image_url || null);
            lastRenderedStep = step.step_index;
          }
        }
      }
      syncLiveRunStatusStep(effectiveStatus);
      updateLivePrepProgress(run, st);
      upsertLiveStep("step-live-prep", "", "");
      const incomingRois = (st && Array.isArray(st.roi_marks))
        ? st.roi_marks.filter((roi) => roi && Number.isFinite(Number(roi.roi_id)))
        : [];
      incomingRois.sort((a, b) => Number(a.roi_id) - Number(b.roi_id));

      roiById.clear();
      for (const roi of incomingRois) {
        roiById.set(Number(roi.roi_id), roi);
      }

      const renderedItems = Array.from(roisEl.querySelectorAll(".roi-item"));
      const renderedKeys = renderedItems.map((node) => String(node.dataset.roiKey || ""));
      const incomingKeys = incomingRois.map((roi) => `${roi.roi_id}:${String(roi.debug_path || "")}`);
      const needsRebuild = (
        renderedKeys.length !== incomingKeys.length ||
        renderedKeys.some((k, i) => k !== incomingKeys[i])
      );
      if (needsRebuild) {
        clearRoiItemsFromList();
        lastRenderedRoi = 0;
      }

      for (const roi of incomingRois) {
        if (roi.roi_id > lastRenderedRoi) {
          const parts = [];
          if (roi.importance !== undefined && roi.importance !== null) parts.push(`Importance ${roi.importance}`);
          if (roi.field_width_um && roi.field_height_um) parts.push(`Field ~${roi.field_width_um.toFixed(0)}×${roi.field_height_um.toFixed(0)} µm`);
          if (roi.tissue_fraction !== undefined && roi.tissue_fraction !== null) parts.push(`Tissue ${roi.tissue_fraction.toFixed(2)}`);
          if (roi.note) parts.push(`Note: ${roi.note}`);
          appendRoiItem(roi, `ROI ${roi.roi_id}: ${roi.label || ""}`, parts.join(" · "), roi.image_url || null);
          lastRenderedRoi = roi.roi_id;
        }
      }

      if (!incomingRois.length) {
        selectedOverviewRoiId = null;
      } else if (!selectedOverviewRoiId || !roiById.has(selectedOverviewRoiId)) {
        selectedOverviewRoiId = Number(incomingRois[incomingRois.length - 1].roi_id);
      }

      upsertLiveRoiItem(st && st.current_view ? st.current_view : null, run.status);
      setSelectedRoiInList();
      renderOverviewRoiOverlay();

      if (run.status === "done" || run.status === "error" || run.status === "terminated") {
        terminateRequested = run.status === "terminated";
        runRequestedModel = null;
        if (pollingTimer) {
          clearInterval(pollingTimer);
          pollingTimer = null;
        }
      }
      syncStartButtonState();
    } catch (e) {
      // ignore transient errors
    }
  }

  function renderDarkOverlay() {
    if (!darkImg.src) {
      const ctx = darkCanvas.getContext("2d");
      ctx.clearRect(0, 0, darkCanvas.width, darkCanvas.height);
      return;
    }
    const w = darkImg.naturalWidth || darkImg.width;
    const h = darkImg.naturalHeight || darkImg.height;
    if (!w || !h) return;
    darkCanvas.width = w;
    darkCanvas.height = h;
    const ctx = darkCanvas.getContext("2d");
    ctx.clearRect(0, 0, w, h);
    if (darkMaskMode) {
      try {
        ctx.fillStyle = "rgba(0,0,0,0.45)";
        ctx.fillRect(0, 0, w, h);
        ctx.globalCompositeOperation = "destination-out";
        ctx.drawImage(darkImg, 0, 0, w, h);
      } catch (e) {
        // ignore drawing errors
      } finally {
        ctx.globalCompositeOperation = "source-over";
      }
      return;
    }
    if (!darkBoxes.length) {
      return;
    }
    ctx.lineWidth = Math.max(1, Math.round(Math.min(w, h) * 0.003));
    ctx.strokeStyle = "rgba(124,240,193,0.9)";
    for (const b of darkBoxes) {
      ctx.strokeRect(b.x, b.y, b.w, b.h);
    }
  }

  async function fetchDarkRegions(runId) {
    if (!runId || darkRegionsLoaded || !darkRegionsEnabled) return;
    try {
      const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/dark_regions`, { cache: "no-store" });
      if (!res.ok) throw new Error(await res.text());
      const data = await res.json();
      const overlayUrl = data.mask_url || data.image_url || "";
      if (!overlayUrl) throw new Error("No dark-region overlay returned.");

      darkRegionsLoaded = true;
      darkBoxes = Array.isArray(data.boxes) ? data.boxes : [];
      darkMaskMode = Boolean(data.mask_url);
      darkImg.hidden = true;
      darkImg.onload = () => {
        renderDarkOverlay();
        renderOverviewRoiOverlay();
      };
      darkImg.src = overlayUrl;
      setDarkEmptyState(darkMaskMode ? "Showing refined mask in Overview." : "Showing coarse boxes in Overview.");
      applyOverviewDisplaySource();
    } catch (e) {
      darkRegionsLoaded = false;
      darkImg.hidden = true;
      darkImg.src = "";
      darkMaskMode = false;
      setDarkEmptyState(String(e && e.message ? e.message : e));
      applyOverviewDisplaySource();
    }
  }

  function xhrUploadSingle(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      activeUploadXhr = xhr;
      xhr.open("POST", url, true);

      xhr.upload.onprogress = (ev) => {
        if (!ev.lengthComputable) return;
        onProgress(ev.loaded, ev.total);
      };

      xhr.onerror = () => {
        activeUploadXhr = null;
        reject(new Error("Network error"));
      };
      xhr.onabort = () => {
        activeUploadXhr = null;
        reject(new Error("Upload terminated by user."));
      };
      xhr.onload = () => {
        activeUploadXhr = null;
        if (xhr.status < 200 || xhr.status >= 300) {
          return reject(new Error(xhr.responseText || `HTTP ${xhr.status}`));
        }
        try {
          resolve(JSON.parse(xhr.responseText));
        } catch {
          resolve({ ok: true });
        }
      };

      xhr.send(formData);
    });
  }

  async function apiCreateRun() {
    const fd = new FormData();
    fd.append("agent_type", selectedAgentType());
    fd.append("model_name", selectedModelName());
    fd.append("prompt", promptEl.value || "");
    fd.append("extractor_name", extractorSelect ? extractorSelect.value : "uni2");
    fd.append("tile_size_px", tileSizeSelect ? tileSizeSelect.value : "224");
    fd.append("batch_size", String(selectedBatchSize()));
    fd.append("tile_prefilter_method", selectedTilePrefilterMethod());
    const res = await fetch("/api/runs/create", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function apiFinalize(runId) {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/finalize`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function apiTerminate(runId) {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/terminate`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function apiClearOutputs(runId) {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/clear_outputs`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function uploadAllFilesPerRequest(runId) {
    const totalBytes = items.reduce((a, b) => a + (b.file.size || 0), 0);
    let uploadedBytes = 0;

    for (let i = 0; i < items.length; i++) {
      if (terminateRequested) throw new Error("Run terminated by user.");
      const it = items[i];

      // Upload one file
      const fd = new FormData();
      fd.append("relpath", it.relPath || it.file.name);
      fd.append("file", it.file, it.file.name);

      const fileSize = it.file.size || 0;

      setPill(statusPill, "run", `Uploading ${i + 1}/${items.length}…`);
      upsertLiveStatusStep(
        "Uploading slide bundle",
        `${i + 1}/${items.length} · ${it.relPath} (${bytesToHuman(fileSize)})`
      );

      let lastLoaded = 0;
      await xhrUploadSingle(`/api/runs/${encodeURIComponent(runId)}/upload`, fd, (loaded, total) => {
        // overall progress: uploadedBytes + loaded for current file
        // but subtract previous loaded to avoid double counting
        const delta = loaded - lastLoaded;
        lastLoaded = loaded;
        uploadedBytes += delta;

        const overallPct = totalBytes > 0 ? Math.round((uploadedBytes / totalBytes) * 100) : 0;
        upsertLiveStatusStep(
          "Uploading slide bundle",
          `${i + 1}/${items.length} · ${it.relPath} · ${Math.max(0, Math.min(100, overallPct))}%`
        );
      });

      // Ensure we count full file even if progress events were weird
      if (lastLoaded < fileSize) {
        uploadedBytes += (fileSize - lastLoaded);
        const overallPct = totalBytes > 0 ? Math.round((uploadedBytes / totalBytes) * 100) : 0;
        upsertLiveStatusStep(
          "Uploading slide bundle",
          `${i + 1}/${items.length} · ${it.relPath} · ${Math.max(0, Math.min(100, overallPct))}%`
        );
      }
      if (terminateRequested) throw new Error("Run terminated by user.");
    }
  }

  async function startFlow() {
    const v = computeModeAndValidation();
    if (!v.ok) return;
    const usingServerSelection = !!serverSelection;

    resetRunUI();
    closeStatusActionsMenu();
    terminateRequested = false;
    runRequestedModel = selectedModelName();
    currentModelName = runRequestedModel;
    activeRunStatus = "created";
    syncStatusPillVisibility("created");
    upsertLiveStatusStep("Creating run", "Initializing run metadata…");
    syncStartButtonState(v);
    errorBox.hidden = true;
    errorText.textContent = "";

    try {
      setPill(statusPill, "run", "Creating run…");
      const created = await apiCreateRun();
      currentRunId = created.run_id;
      const backendModelName = created.model_name ? String(created.model_name).trim() : "";
      if (backendModelName && runRequestedModel && backendModelName !== runRequestedModel) {
        runLabelEl.textContent = `Requested ${runRequestedModel}, backend returned ${backendModelName}.`;
      } else {
        runLabelEl.textContent = `Run created (${selectedAgentType().toUpperCase()}).`;
      }
      runIdEl.textContent = currentRunId;
      setModelStatus("created", currentModelName);
      syncTerminateButtonState();

      if (usingServerSelection) {
        setPill(statusPill, "run", "Preparing server slide…");
        upsertLiveStatusStep(
          "Attaching server slide",
          serverSelection ? serverSelection.requestedPath : "Resolving server selection…"
        );
        await apiAttachServerSelection(currentRunId, serverSelection.requestedPath);
        if (terminateRequested) throw new Error("Run terminated by user.");
      } else {
        // Upload files one by one
        setPill(statusPill, "run", "Uploading…");
        syncLiveRunStatusStep("uploading");

        await uploadAllFilesPerRequest(currentRunId);
        if (terminateRequested) throw new Error("Run terminated by user.");
      }

      // Finalize + start
      setPill(statusPill, "run", usingServerSelection ? "Starting…" : "Finalizing…");
      upsertLiveStatusStep(
        usingServerSelection ? "Starting run" : "Finalizing run",
        usingServerSelection ? "Validating server slide and starting agent…" : "Validating bundle and starting agent…"
      );
      setModelStatus("pending", currentModelName);

      await apiFinalize(currentRunId);
      if (terminateRequested) throw new Error("Run terminated by user.");

      setPill(statusPill, "run", "Pending");
      runLabelEl.textContent = `Run started.`;
      activeRunStatus = "pending";
      syncLiveRunStatusStep("pending", "Run started.");
      syncStartButtonState(v);

      if (pollingTimer) clearInterval(pollingTimer);
      pollingTimer = setInterval(pollRun, 500);
      pollRun();

    } catch (e) {
      if (terminateRequested) {
        setPill(statusPill, "bad", "Terminated");
        setModelStatus("terminated", currentModelName);
        activeRunStatus = "terminated";
        syncStatusPillVisibility("terminated");
        runLabelEl.textContent = "Run terminated by user.";
        syncLiveRunStatusStep("terminated", "Run terminated by user.");
      } else {
        setPill(statusPill, "bad", "Error");
        setModelStatus("error", currentModelName);
        activeRunStatus = "error";
        syncStatusPillVisibility("error");
        const errMsg = String(e && e.message ? e.message : e);
        syncLiveRunStatusStep("error", errMsg);
      }
      runRequestedModel = null;
      syncStartButtonState(v);
      if (terminateRequested) {
        errorBox.hidden = true;
        errorText.textContent = "";
      } else {
        errorBox.hidden = false;
        errorText.textContent = String(e && e.message ? e.message : e);
      }
      return;
    }
  }

  async function terminateCurrentRun() {
    if (!currentRunId || !isRunBusyStatus(activeRunStatus)) return;
    terminateRequested = true;
    closeStatusActionsMenu();
    if (activeUploadXhr) {
      try {
        activeUploadXhr.abort();
      } catch (_e) {
        // Ignore upload abort failures.
      }
    }
    runLabelEl.textContent = "Terminating run…";
    upsertLiveStatusStep("Terminating run", "Sending terminate request…");
    syncTerminateButtonState();
    try {
      await apiTerminate(currentRunId);
      activeRunStatus = "terminated";
      runRequestedModel = null;
      setPill(statusPill, "bad", "Terminated");
      setModelStatus("terminated", currentModelName);
      syncStatusPillVisibility("terminated");
      runLabelEl.textContent = "Run terminated by user.";
      syncLiveRunStatusStep("terminated", "Run terminated by user.");
      if (pollingTimer) {
        clearInterval(pollingTimer);
        pollingTimer = null;
      }
      await pollRun();
    } catch (e) {
      runLabelEl.textContent = `Terminate failed: ${String(e && e.message ? e.message : e)}`;
      syncLiveRunStatusStep("error", runLabelEl.textContent);
    } finally {
      syncStartButtonState();
      syncTerminateButtonState();
    }
  }

  async function restartCurrentRun() {
    if (!currentRunId) return;
    if (!(activeRunStatus === "done" || activeRunStatus === "error" || activeRunStatus === "terminated")) return;
    closeStatusActionsMenu();
    const v = computeModeAndValidation();
    if (!v.ok) {
      runLabelEl.textContent = "Restart requires a valid file selection.";
      syncStartButtonState(v);
      return;
    }
    runLabelEl.textContent = "Restarting run…";
    await startFlow();
  }

  async function clearCurrentRunOutputs() {
    if (!currentRunId || activeRunStatus !== "done") return;
    closeStatusActionsMenu();
    runLabelEl.textContent = "Clearing outputs…";
    try {
      await apiClearOutputs(currentRunId);
      await pollRun();
      runLabelEl.textContent = "Outputs cleared.";
      syncStartButtonState();
    } catch (e) {
      runLabelEl.textContent = `Clear outputs failed: ${String(e && e.message ? e.message : e)}`;
    }
  }

  // Buttons
  btnClear.addEventListener("click", clearSelection);
  btnStart.addEventListener("click", startFlow);
  if (tilePrefilterMethodSelect) {
    const storedTilePrefilterMethod = loadStoredValue(tilePrefilterMethodStorageKey);
    if (storedTilePrefilterMethod) {
      tilePrefilterMethodSelect.value = storedTilePrefilterMethod;
    }
    tilePrefilterMethodSelect.addEventListener("change", () => {
      saveStoredValue(tilePrefilterMethodStorageKey, selectedTilePrefilterMethod());
    });
  }
  if (btnActions && statusActionsMenu) {
    btnActions.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleStatusActionsMenu();
    });
    statusActionsMenu.addEventListener("click", (e) => {
      e.stopPropagation();
    });
    document.addEventListener("click", (e) => {
      const target = e.target;
      if (!(target instanceof Element)) {
        closeStatusActionsMenu();
        return;
      }
      if (!statusActionsMenu.contains(target) && !btnActions.contains(target)) {
        closeStatusActionsMenu();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        closeStatusActionsMenu();
      }
    });
  }
  if (btnTerminate) {
    btnTerminate.addEventListener("click", terminateCurrentRun);
  }
  if (btnRestart) {
    btnRestart.addEventListener("click", restartCurrentRun);
  }
  if (btnClearOutputs) {
    btnClearOutputs.addEventListener("click", clearCurrentRunOutputs);
  }
  if (btnDarkToggle) {
    btnDarkToggle.addEventListener("click", () => setDarkRegionsEnabled(!darkRegionsEnabled));
    btnDarkToggle.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        setDarkRegionsEnabled(!darkRegionsEnabled);
      }
    });
  }
  if (overviewImg) {
    overviewImg.addEventListener("load", renderOverviewRoiOverlay);
  }

  if (uploadActionTrigger && uploadActionMenu) {
    uploadActionTrigger.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleUploadActionMenu();
    });
    uploadActionTrigger.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown" || e.key === "Enter" || e.key === " ") {
        e.preventDefault();
        openUploadActionMenu();
        if (uploadActionOptions.length) {
          uploadActionOptions[0].focus();
        }
      }
    });
  }

  if (uploadActionOptions.length) {
    for (const optionBtn of uploadActionOptions) {
      const action = optionBtn.dataset.uploadAction || "";
      optionBtn.addEventListener("mouseenter", (e) => {
        const ev = e;
        showUploadActionTip(uploadHintForAction(action), ev.clientX, ev.clientY);
      });
      optionBtn.addEventListener("mousemove", (e) => {
        const ev = e;
        showUploadActionTip(uploadHintForAction(action), ev.clientX, ev.clientY);
      });
      optionBtn.addEventListener("mouseleave", hideUploadActionTip);
      optionBtn.addEventListener("focus", () => {
        const rect = optionBtn.getBoundingClientRect();
        showUploadActionTip(uploadHintForAction(action), rect.right + 6, rect.top + 6);
      });
      optionBtn.addEventListener("blur", hideUploadActionTip);
      optionBtn.addEventListener("click", () => {
        triggerUploadAction(action);
        if (uploadActionTrigger) {
          uploadActionTrigger.textContent = optionBtn.textContent || "Select slide source";
        }
        closeUploadActionMenu();
      });
    }
  }

  document.addEventListener("click", (e) => {
    if (!uploadActionMenu || !uploadActionTrigger) return;
    const target = e.target;
    if (!(target instanceof Element)) {
      closeUploadActionMenu();
      return;
    }
    if (!uploadActionMenu.contains(target) && !uploadActionTrigger.contains(target)) {
      closeUploadActionMenu();
    }
  });
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") {
      closeExplorer();
      closeUploadActionMenu();
    }
  });

  if (explorerCloseBtn) {
    explorerCloseBtn.addEventListener("click", closeExplorer);
  }
  if (explorerBackdrop) {
    explorerBackdrop.addEventListener("click", closeExplorer);
  }
  if (explorerRootSelect) {
    explorerRootSelect.addEventListener("change", () => {
      explorerSelectedFilePath = "";
      explorerSelectedFileName = "";
      loadExplorerPath(explorerRootSelect.value);
    });
  }
  if (explorerUpBtn) {
    explorerUpBtn.addEventListener("click", () => {
      if (explorerParentPathValue) {
        explorerSelectedFilePath = "";
        explorerSelectedFileName = "";
        loadExplorerPath(explorerParentPathValue);
      }
    });
  }
  if (explorerRefreshBtn) {
    explorerRefreshBtn.addEventListener("click", () => {
      if (explorerCurrentPathValue) {
        loadExplorerPath(explorerCurrentPathValue);
      }
    });
  }
  if (explorerUseFolderBtn) {
    explorerUseFolderBtn.addEventListener("click", () => {
      if (explorerCurrentPathValue) {
        resolveExplorerSelection(explorerCurrentPathValue);
      }
    });
  }
  if (explorerUseFileBtn) {
    explorerUseFileBtn.addEventListener("click", () => {
      if (explorerSelectedFilePath) {
        resolveExplorerSelection(explorerSelectedFilePath);
      }
    });
  }

  if (modelSelect) {
    modelSelect.addEventListener("change", () => {
      modelSelectionTouched = true;
      if (!isRunBusyStatus(activeRunStatus)) {
        currentModelName = selectedModelName();
      }
      setModelStatus(activeRunStatus || "idle", currentModelName);
    });
  }

  // Inputs
  fileInput.addEventListener("change", () => {
    addFiles(fileInput.files);
    fileInput.value = "";
  });
  folderInput.addEventListener("change", () => {
    addFiles(folderInput.files);
    folderInput.value = "";
  });
  zipInput.addEventListener("change", () => {
    clearSelection();
    addFiles(zipInput.files);
    zipInput.value = "";
  });

  // Dropzone DnD
  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", handleDrop);

  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      fileInput.value = "";
      fileInput.click();
    }
  });

  if (overviewImg) {
    overviewImg.addEventListener("load", () => {
      updateSearchingBox(lastCurrentViewState, activeRunStatus);
      renderOverviewRoiOverlay();
    });
  }

  if (roisEl) {
    roisEl.addEventListener("click", (e) => {
      const target = e.target;
      if (!(target instanceof Element)) return;
      const item = target.closest(".roi-item");
      if (!item) return;
      const roiId = Number(item.dataset.roiId || "");
      if (!Number.isFinite(roiId)) return;
      selectOverviewRoi(roiId);
    });
  }

  // Initial render
  initResizableLayout();
  currentModelName = selectedModelName();
  setModelStatus("idle");
  syncStatusPillVisibility("idle");
  setDarkRegionsEnabled(true);
  setExplorerBusyState(false);
  updateExplorerSelectionPreview();
  render();
  fetchServiceModelName();
})();
