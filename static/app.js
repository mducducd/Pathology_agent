(() => {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const folderInput = document.getElementById("folder-input");
  const zipInput = document.getElementById("zip-input");

  const btnPickFiles = document.getElementById("btn-pick-files");
  const btnPickFolder = document.getElementById("btn-pick-folder");
  const btnPickZip = document.getElementById("btn-pick-zip");
  const btnClear = document.getElementById("btn-clear");
  const btnStart = document.getElementById("btn-start");

  const filelist = document.getElementById("filelist");
  const filelistMeta = document.getElementById("filelist-meta");
  const validationEl = document.getElementById("validation");
  const modePill = document.getElementById("mode-pill");

  const promptEl = document.getElementById("prompt");
  const agentInputs = Array.from(document.querySelectorAll('input[name="agent_type"]'));

  const progressWrap = document.getElementById("progress-wrap");
  const progressFill = document.getElementById("progress-fill");
  const progressPct = document.getElementById("progress-pct");
  const progressDetail = document.getElementById("progress-detail");

  const statusPill = document.getElementById("status-pill");
  const runIdEl = document.getElementById("run-id");
  const runLabelEl = document.getElementById("run-label");
  const slideNameEl = document.getElementById("slide-name");

  const errorBox = document.getElementById("error-box");
  const errorText = document.getElementById("error-text");

  const overviewImg = document.getElementById("overview-img");
  const overviewEmpty = document.getElementById("overview-empty");
  const currentImg = document.getElementById("current-img");
  const currentEmpty = document.getElementById("current-empty");
  const currentMeta = document.getElementById("current-meta");
  const finalText = document.getElementById("final-text");
  const reasoningText = document.getElementById("reasoning-text");
  const reportLink = document.getElementById("report-link");
  const stepsEl = document.getElementById("steps");
  const roisEl = document.getElementById("rois");

  const defaultPrompts = {
    msi: `Your only task is to assess this H&E whole-slide image for morphologic features suggestive of microsatellite instability (MSI-high).
- First, identify the likely organ and tumor type.
- Then, systematically explore multiple distinct tumor regions at high power, using the field width in micrometers and tissue_fraction to ensure you are truly at high magnification with tissue in view when assessing tumor-infiltrating lymphocytes and cytologic detail.
- Use the WSI navigation tools as instructed and provide a nav_reason for every tool call.
- Mark ROIs that best illustrate MSI-relevant features (or representative MSS-like morphology if MSI features are absent).
- After each ROI, inspect the CURRENT VIEW ROI image; discard it if it is mostly background or not diagnostic.
- At the end, give a qualitative assessment of MSI likelihood and explicitly recommend confirmatory immunohistochemistry or molecular testing.`,
    wsi: `Inspect the whole-slide image and describe the likely tissue of origin and any key findings (including tumors, inflammatory infiltrates, necrosis, etc.).
Use the WSI tools to get an overview and then pan/zoom as needed, similar to a human pathologist using a digital slide viewer.
Use the approximate field width in micrometers and tissue_fraction to ensure you reach true high-power views on tissue when you need cellular detail.
Provide nav_reason for each tool call.
Mark important regions of interest with wsi_mark_roi_norm so they can be highlighted in the final report.
After each ROI, review the CURRENT VIEW ROI image and call wsi_discard_last_roi if the ROI is mostly background or not diagnostic.
If you cannot find a suspicious lesion after exploring representative areas at adequate magnification, state that no obvious lesion was identified.`
  };

  function selectedAgentType() {
    const el = document.querySelector('input[name="agent_type"]:checked');
    return el ? el.value : "msi";
  }

  function maybeLoadDefaultPrompt() {
    const type = selectedAgentType();
    const d = defaultPrompts[type] || "";
    if (!promptEl.value.trim()) {
      promptEl.value = d;
      return;
    }
    if (promptEl.value === defaultPrompts.msi || promptEl.value === defaultPrompts.wsi) {
      promptEl.value = d;
    }
  }

  agentInputs.forEach((inp) => inp.addEventListener("change", maybeLoadDefaultPrompt));
  promptEl.value = defaultPrompts[selectedAgentType()];

  const allowedPrimary = new Set([".svs", ".tif", ".tiff", ".ndpi", ".mrxs", ".mrsx"]);
  const allowedZip = ".zip";
  const standard = new Set([".svs", ".tif", ".tiff", ".ndpi"]);
  const mirax = new Set([".mrxs", ".mrsx"]);

  let items = []; // {file, relPath, id}
  let currentRunId = null;
  let pollingTimer = null;
  let lastRenderedStep = 0;
  let lastRenderedRoi = 0;

  function bytesToHuman(n) {
    if (!n) return "0 B";
    const k = 1024;
    const sizes = ["B", "KB", "MB", "GB", "TB"];
    const i = Math.floor(Math.log(n) / Math.log(k));
    const v = n / Math.pow(k, i);
    return `${v.toFixed(v >= 10 || i === 0 ? 0 : 1)} ${sizes[i]}`;
  }

  function extOf(name) {
    const i = name.lastIndexOf(".");
    if (i < 0) return "";
    return name.slice(i).toLowerCase();
  }

  function uniqId() {
    return Math.random().toString(16).slice(2) + Date.now().toString(16);
  }

  function clearSelection() {
    items = [];
    fileInput.value = "";
    folderInput.value = "";
    zipInput.value = "";
    render();
  }

  function addFiles(files) {
    for (const f of files) {
      const relPath = f.webkitRelativePath || f.name;
      items.push({ file: f, relPath, id: uniqId() });
    }
    render();
  }

  function removeItem(id) {
    items = items.filter((x) => x.id !== id);
    render();
  }

  function computeModeAndValidation() {
    if (items.length === 0) {
      return { ok: false, level: "idle", mode: "No files", msg: "Drop or choose files to begin." };
    }

    const exts = items.map(x => extOf(x.relPath || x.file.name));
    const hasZip = exts.includes(allowedZip);
    const hasStd = exts.some(e => standard.has(e));
    const hasMirax = exts.some(e => mirax.has(e));
    const totalBytes = items.reduce((a, b) => a + (b.file.size || 0), 0);

    if (hasZip) {
      if (items.length !== 1) {
        return { ok: false, level: "bad", mode: "MIRAX zip (invalid mix)", msg: "If you upload a .zip, it must be the only file." };
      }
      return { ok: true, level: "good", mode: "MIRAX zip", msg: `Ready to upload 1 zip (${bytesToHuman(totalBytes)}).` };
    }

    if (hasStd && hasMirax) {
      return { ok: false, level: "bad", mode: "Mixed (invalid)", msg: "Do not mix standard slides with MIRAX in one upload." };
    }

    if (hasStd) {
      if (items.length !== 1) {
        return { ok: false, level: "bad", mode: "Standard slide (invalid)", msg: "Standard slides must be uploaded as a single file." };
      }
      const e = exts[0];
      if (!allowedPrimary.has(e)) {
        return { ok: false, level: "bad", mode: "Unsupported", msg: "Unsupported file." };
      }
      return { ok: true, level: "good", mode: "Standard slide", msg: `Ready (${bytesToHuman(totalBytes)}).` };
    }

    if (hasMirax) {
      if (items.length === 1) {
        return { ok: false, level: "bad", mode: "MIRAX file only (invalid)", msg: "MIRAX needs its companion data directory. Upload folder or zip." };
      }
      // with per-file upload, file count is not scary anymore
      return { ok: true, level: "good", mode: "MIRAX folder/files", msg: `Ready (${items.length} files, ${bytesToHuman(totalBytes)}).` };
    }

    return { ok: false, level: "bad", mode: "Unsupported", msg: "No supported slide detected." };
  }

  function setPill(el, level, text) {
    el.classList.remove("pill-idle", "pill-good", "pill-warn", "pill-bad", "pill-run");
    el.classList.add(
      level === "good" ? "pill-good" :
      level === "warn" ? "pill-warn" :
      level === "bad"  ? "pill-bad" :
      level === "run"  ? "pill-run" : "pill-idle"
    );
    el.textContent = text;
  }

  function render() {
    filelist.innerHTML = "";
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
      rm.className = "remove";
      rm.type = "button";
      rm.textContent = "Remove";
      rm.addEventListener("click", () => removeItem(it.id));

      right.appendChild(tag);
      right.appendChild(rm);

      li.appendChild(left);
      li.appendChild(right);
      filelist.appendChild(li);
    }

    const v = computeModeAndValidation();
    setPill(modePill, v.level === "idle" ? "idle" : v.level, v.mode);
    validationEl.className = "validation " + (v.level === "good" ? "ok" : v.level === "warn" ? "warn" : v.level === "idle" ? "" : "bad");
    validationEl.textContent = v.msg;

    btnStart.disabled = !v.ok;
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

    for (const f of collected) {
      const rel = f._relPath || f.name;
      items.push({ file: f, relPath: rel, id: uniqId() });
    }
    render();
  }

  function showProgress(show) {
    progressWrap.style.display = show ? "block" : "none";
    progressWrap.setAttribute("aria-hidden", show ? "false" : "true");
    if (!show) {
      progressFill.style.width = "0%";
      progressPct.textContent = "0%";
      progressDetail.textContent = "";
    }
  }

  function resetRunUI() {
    setPill(statusPill, "idle", "Idle");
    runIdEl.textContent = "—";
    runLabelEl.textContent = "";
    slideNameEl.textContent = "—";

    errorBox.hidden = true;
    errorText.textContent = "";

    overviewImg.hidden = true;
    overviewImg.src = "";
    overviewEmpty.textContent = "—";

    currentImg.hidden = true;
    currentImg.src = "";
    currentEmpty.textContent = "—";
    currentMeta.textContent = "—";

    finalText.textContent = "";
    reasoningText.textContent = "";
    reportLink.textContent = "";
    stepsEl.innerHTML = "";
    roisEl.innerHTML = "";

    lastRenderedStep = 0;
    lastRenderedRoi = 0;
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
      img.src = imgUrl;
      img.alt = title;
      li.appendChild(img);
    }

    listEl.appendChild(li);
  }

  async function pollRun() {
    if (!currentRunId) return;
    try {
      const res = await fetch(`/api/runs/${currentRunId}`);
      if (!res.ok) return;
      const data = await res.json();
      const run = data.run;
      const st = data.wsi_state;

      slideNameEl.textContent = run.slide_filename || "—";

      if (run.status === "created") setPill(statusPill, "run", "Created");
      if (run.status === "uploading") setPill(statusPill, "run", "Uploading");
      if (run.status === "pending") setPill(statusPill, "run", "Pending");
      if (run.status === "running") setPill(statusPill, "run", "Running");
      if (run.status === "done") setPill(statusPill, "good", "Done");
      if (run.status === "error") setPill(statusPill, "bad", "Error");

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

      if (st && st.overview_image_url) {
        overviewImg.hidden = false;
        overviewImg.src = st.overview_image_url;
        overviewEmpty.textContent = "";
      } else {
        overviewImg.hidden = true;
        overviewImg.src = "";
        overviewEmpty.textContent = "—";
      }

      if (st && st.current_view && st.current_view.image_url) {
        currentImg.hidden = false;
        currentImg.src = st.current_view.image_url;
        currentEmpty.textContent = "";
        const v = st.current_view;
        const metaParts = [];
        if (v.field_width_um && v.field_height_um) {
          metaParts.push(`Field ~${v.field_width_um.toFixed(0)} × ${v.field_height_um.toFixed(0)} µm`);
        }
        if (v.tissue_fraction !== undefined && v.tissue_fraction !== null) {
          metaParts.push(`Tissue fraction ${v.tissue_fraction.toFixed(2)}`);
        }
        currentMeta.textContent = metaParts.length ? metaParts.join(" · ") : "—";
      } else {
        currentImg.hidden = true;
        currentImg.src = "";
        currentEmpty.textContent = "—";
        currentMeta.textContent = "—";
      }

      finalText.textContent = run.final_output || "";
      reasoningText.textContent = run.reasoning_content || "";

      if (run.report_path) {
        const relPath = run.report_path.replace(/.*wsi_reports[\\/]/, "");
        const href = `/reports/${relPath}`;
        reportLink.innerHTML = `Report: <a href="${href}" target="_blank" rel="noreferrer">open Markdown report</a>`;
      } else {
        reportLink.textContent = "";
      }

      if (st && Array.isArray(st.step_log)) {
        for (const step of st.step_log) {
          if (step.step_index > lastRenderedStep) {
            const parts = [];
            if (step.nav_reason) parts.push(`Reason: ${step.nav_reason}`);
            if (step.field_width_um && step.field_height_um) parts.push(`Field ~${step.field_width_um.toFixed(0)}×${step.field_height_um.toFixed(0)} µm`);
            if (step.tissue_fraction !== undefined && step.tissue_fraction !== null) parts.push(`Tissue ${step.tissue_fraction.toFixed(2)}`);
            appendLogItem(stepsEl, `Step ${step.step_index} · ${String(step.tool || "")}`, parts.join(" · "), step.image_url || null);
            lastRenderedStep = step.step_index;
          }
        }
      }

      if (st && Array.isArray(st.roi_marks)) {
        for (const roi of st.roi_marks) {
          if (roi.roi_id > lastRenderedRoi) {
            const parts = [];
            if (roi.importance !== undefined && roi.importance !== null) parts.push(`Importance ${roi.importance}`);
            if (roi.field_width_um && roi.field_height_um) parts.push(`Field ~${roi.field_width_um.toFixed(0)}×${roi.field_height_um.toFixed(0)} µm`);
            if (roi.tissue_fraction !== undefined && roi.tissue_fraction !== null) parts.push(`Tissue ${roi.tissue_fraction.toFixed(2)}`);
            if (roi.note) parts.push(`Note: ${roi.note}`);
            appendLogItem(roisEl, `ROI ${roi.roi_id}: ${roi.label || ""}`, parts.join(" · "), roi.image_url || null);
            lastRenderedRoi = roi.roi_id;
          }
        }
      }

      if (run.status === "done" || run.status === "error") {
        if (pollingTimer) {
          clearInterval(pollingTimer);
          pollingTimer = null;
        }
      }
    } catch (e) {
      // ignore transient errors
    }
  }

  function xhrUploadSingle(url, formData, onProgress) {
    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", url, true);

      xhr.upload.onprogress = (ev) => {
        if (!ev.lengthComputable) return;
        onProgress(ev.loaded, ev.total);
      };

      xhr.onerror = () => reject(new Error("Network error"));
      xhr.onload = () => {
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
    fd.append("prompt", promptEl.value || "");
    const res = await fetch("/api/runs/create", { method: "POST", body: fd });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function apiFinalize(runId) {
    const res = await fetch(`/api/runs/${encodeURIComponent(runId)}/finalize`, { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    return await res.json();
  }

  async function uploadAllFilesPerRequest(runId) {
    const totalBytes = items.reduce((a, b) => a + (b.file.size || 0), 0);
    let uploadedBytes = 0;

    for (let i = 0; i < items.length; i++) {
      const it = items[i];

      // Upload one file
      const fd = new FormData();
      fd.append("relpath", it.relPath || it.file.name);
      fd.append("file", it.file, it.file.name);

      const fileSize = it.file.size || 0;

      setPill(statusPill, "run", `Uploading ${i + 1}/${items.length}…`);
      progressDetail.textContent = `${it.relPath} (${bytesToHuman(fileSize)})`;

      let lastLoaded = 0;
      await xhrUploadSingle(`/api/runs/${encodeURIComponent(runId)}/upload`, fd, (loaded, total) => {
        // overall progress: uploadedBytes + loaded for current file
        // but subtract previous loaded to avoid double counting
        const delta = loaded - lastLoaded;
        lastLoaded = loaded;
        uploadedBytes += delta;

        const pct = totalBytes > 0 ? Math.round((uploadedBytes / totalBytes) * 100) : 0;
        progressFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
        progressPct.textContent = `${Math.max(0, Math.min(100, pct))}%`;
      });

      // Ensure we count full file even if progress events were weird
      if (lastLoaded < fileSize) {
        uploadedBytes += (fileSize - lastLoaded);
        const pct = totalBytes > 0 ? Math.round((uploadedBytes / totalBytes) * 100) : 0;
        progressFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
        progressPct.textContent = `${Math.max(0, Math.min(100, pct))}%`;
      }
    }
  }

  async function startFlow() {
    const v = computeModeAndValidation();
    if (!v.ok) return;

    resetRunUI();
    showProgress(true);
    btnStart.disabled = true;
    errorBox.hidden = true;
    errorText.textContent = "";

    try {
      setPill(statusPill, "run", "Creating run…");
      const created = await apiCreateRun();
      currentRunId = created.run_id;
      runIdEl.textContent = currentRunId;
      runLabelEl.textContent = `Run created (${selectedAgentType().toUpperCase()}).`;

      // Upload files one by one
      setPill(statusPill, "run", "Uploading…");
      progressFill.style.width = "0%";
      progressPct.textContent = "0%";
      progressDetail.textContent = "";

      await uploadAllFilesPerRequest(currentRunId);

      // Finalize + start
      setPill(statusPill, "run", "Finalizing…");
      progressDetail.textContent = "Validating bundle and starting agent…";
      progressFill.style.width = "100%";
      progressPct.textContent = "100%";

      await apiFinalize(currentRunId);

      showProgress(false);
      setPill(statusPill, "run", "Pending");
      runLabelEl.textContent = `Run started.`;

      if (pollingTimer) clearInterval(pollingTimer);
      pollingTimer = setInterval(pollRun, 1000);
      pollRun();

    } catch (e) {
      showProgress(false);
      setPill(statusPill, "bad", "Error");
      btnStart.disabled = false;
      errorBox.hidden = false;
      errorText.textContent = String(e && e.message ? e.message : e);
      return;
    }
  }

  // Buttons
  btnPickFiles.addEventListener("click", () => fileInput.click());
  btnPickFolder.addEventListener("click", () => folderInput.click());
  btnPickZip.addEventListener("click", () => zipInput.click());
  btnClear.addEventListener("click", clearSelection);
  btnStart.addEventListener("click", startFlow);

  // Inputs
  fileInput.addEventListener("change", () => addFiles(fileInput.files));
  folderInput.addEventListener("change", () => addFiles(folderInput.files));
  zipInput.addEventListener("change", () => {
    clearSelection();
    addFiles(zipInput.files);
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
      fileInput.click();
    }
  });

  function showProgress(show) {
    progressWrap.style.display = show ? "block" : "none";
    progressWrap.setAttribute("aria-hidden", show ? "false" : "true");
    if (!show) {
      progressFill.style.width = "0%";
      progressPct.textContent = "0%";
      progressDetail.textContent = "";
    }
  }

  // Initial render
  render();
})();
