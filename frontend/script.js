document.addEventListener("DOMContentLoaded", () => {

    // ===== SIDEBAR =====
    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebarToggle");
    sidebarToggle.addEventListener("click", () => sidebar.classList.toggle("collapsed"));

    // ===== TABS =====
    const tabLinks = document.querySelectorAll(".tab-link");
    const tabPanes = document.querySelectorAll(".tab-pane");
    tabLinks.forEach(link => {
        link.addEventListener("click", () => {
            tabLinks.forEach(l => l.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            link.classList.add("active");
            const target = document.getElementById(link.dataset.target);
            if (target) target.classList.add("active");
            if (link.dataset.target === "tab-history") renderHistoryList();
        });
    });

    // ===== CONFIG MODAL =====
    const configOverlay = document.getElementById("configOverlay");
    const configToggleBtn = document.getElementById("configToggleBtn");
    const closeConfigBtn = document.getElementById("closeConfigBtn");
    const saveApiBtn = document.getElementById("saveApiBtn");
    const apiUrlInput = document.getElementById("apiUrlInput");
    const apiStatusDisplay = document.getElementById("apiStatusDisplay");
    const serverStatusBadge = document.getElementById("serverStatusBadge");
    const statusDot = serverStatusBadge.querySelector(".status-dot");
    const statusText = document.getElementById("statusText");

    let apiUrl = localStorage.getItem("waferApiUrl") || "http://localhost:8000";
    apiUrlInput.value = apiUrl;

    configToggleBtn.addEventListener("click", () => configOverlay.classList.add("active"));
    closeConfigBtn.addEventListener("click", () => configOverlay.classList.remove("active"));
    configOverlay.addEventListener("click", (e) => { if (e.target === configOverlay) configOverlay.classList.remove("active"); });
    saveApiBtn.addEventListener("click", () => {
        apiUrl = apiUrlInput.value.trim().replace(/\/$/, "");
        localStorage.setItem("waferApiUrl", apiUrl);
        configOverlay.classList.remove("active");
        checkServerHealth();
    });

    async function checkServerHealth() {
        statusDot.className = "status-dot";
        statusText.innerText = "Checking...";
        try {
            const res = await fetch(`${apiUrl}/health`, { method: "GET", headers: { "ngrok-skip-browser-warning": "true" } });
            if (res.ok) {
                statusDot.classList.add("online");
                statusText.innerText = "Engine Online";
                apiStatusDisplay.innerText = "Connected: " + apiUrl;
                apiStatusDisplay.style.color = "var(--success)";
            } else throw new Error("Bad response");
        } catch {
            statusDot.classList.add("offline");
            statusText.innerText = "Engine Offline";
            apiStatusDisplay.innerText = "Unreachable — check your Ngrok tunnel";
            apiStatusDisplay.style.color = "var(--danger)";
        }
    }
    checkServerHealth();

    // ===== EXPLANATION MODAL =====
    const explanationOverlay = document.getElementById("explanationOverlay");
    const closeExplanationBtn = document.getElementById("closeExplanationBtn");
    closeExplanationBtn.addEventListener("click", () => explanationOverlay.classList.remove("active"));
    explanationOverlay.addEventListener("click", (e) => { if (e.target === explanationOverlay) explanationOverlay.classList.remove("active"); });

    function buildExplanation(className, confidence, riskScore, action) {
        const confidencePct = (confidence * 100).toFixed(1);
        document.getElementById("explanationSubtitle").innerText = `${className} defect · ${confidencePct}% confidence`;

        let riskColor = "var(--success)";
        if (action === "STOP LOT" || action === "STOP") riskColor = "var(--danger)";
        else if (action === "INVESTIGATE") riskColor = "var(--warning)";

        const defectExplanations = {
            "Center": "The defect is concentrated at the wafer center, typically caused by systematic gas distribution or plasma non-uniformity during CVD or etch processes.",
            "Donut": "A ring-like defect surrounding the center, often caused by temperature gradients or irregular photoresist spinning at the wafer hub.",
            "Edge-Loc": "Defects localized along specific edge arcs, typically caused by edge bead removal issues, clamp ring pressure, or edge gas flow anomalies.",
            "Edge-Ring": "A full circumferential defect ring indicating chuck or wafer-holder edge sealing issues, or plasma edge sheath inconsistencies.",
            "Loc": "Localized cluster defects suggesting contamination from a specific chamber component, particulate generation, or handling robot contact.",
            "Near-Full": "Nearly the entire wafer surface is affected, indicating a severe, systemic process failure such as a blocked nozzle or major equipment malfunction.",
            "Random": "Scattered random defects without spatial pattern, typically from airborne contamination, random particle events, or minor handling issues.",
            "Scratch": "Linear scratches indicating mechanical contact — likely from robot blade alignment, cassette slot friction, or probe card contact during testing.",
            "None": "No defect pattern detected. The wafer appears clean and within expected yield parameters."
        };

        const actionSteps = {
            "STOP LOT": [
                { title: "Halt the Lot", detail: "Immediately stop further processing of this wafer lot to prevent cascading yield loss across downstream steps." },
                { title: "Isolate the Chamber", detail: "Lock out the process chamber for inspection. Do not run additional wafers until root cause is confirmed." },
                { title: "Engineer Escalation", detail: "Alert the process engineer and fab manager. Initiate a formal Equipment Alarm Report (EAR)." }
            ],
            "STOP": [
                { title: "Halt the Lot", detail: "Immediately stop further processing to prevent additional yield loss." },
                { title: "Isolate the Chamber", detail: "Lock out the process chamber for inspection and maintenance review." },
                { title: "Engineer Escalation", detail: "Alert the process engineer immediately and file an Equipment Alarm Report." }
            ],
            "INVESTIGATE": [
                { title: "Flag the Lot", detail: "Mark this lot for enhanced monitoring. Do not yet stop production, but increase sampling frequency." },
                { title: "Review Sensor Logs", detail: "Pull the last 24h process data for the relevant tool — check pressure, temperature, and gas flow traces for anomalies." },
                { title: "Watch for Clustering", detail: "If the next 2-3 wafers from this chamber show similar patterns, escalate to a STOP LOT action immediately." }
            ],
            "MONITOR": [
                { title: "Continue Production", detail: "No immediate process intervention is required. This detection is within acceptable yield bounds." },
                { title: "Standard Sampling", detail: "Maintain regular SPC monitoring. Note this wafer in the defect database for trend tracking." },
                { title: "Periodic Review", detail: "Review defect trend charts at end-of-shift. Escalate only if frequency increases above the control limit." }
            ]
        };

        const steps = actionSteps[action] || actionSteps["MONITOR"];
        const defectDesc = defectExplanations[className] || "Pattern identified by the AI classifier based on spatial feature maps extracted from EfficientNet-B0.";

        const stepsHTML = steps.map((s, i) => `
            <div class="explanation-step">
                <span class="step-number">0${i + 1}</span>
                <div class="step-text"><strong>${s.title}</strong><br>${s.detail}</div>
            </div>
        `).join("");

        document.getElementById("explanationBody").innerHTML = `
            <div style="display: flex; align-items: baseline; gap: 0.5rem; margin-bottom: 0.3rem;">
                <span class="explanation-risk-badge" style="color: ${riskColor};">${riskScore}<sup>/100</sup></span>
                <span style="background: ${riskColor}22; color: ${riskColor}; border: 1px solid ${riskColor}55; border-radius: 6px; padding: 4px 12px; font-weight: 700; font-size: 0.8rem; letter-spacing: 1px;">${action}</span>
            </div>
            <p style="color: var(--text-sec); font-size: 0.88rem; line-height: 1.6; margin-bottom: 0.4rem;">${defectDesc}</p>
            <div class="explanation-steps">${stepsHTML}</div>
        `;
    }

    window.showExplanation = function (className, confidence, riskScore, action) {
        buildExplanation(className, confidence, riskScore, action);
        explanationOverlay.classList.add("active");
    };

    // Event delegation for risk-btn (data-* approach avoids quote issues in onclick)
    document.addEventListener("click", (e) => {
        const btn = e.target.closest(".risk-btn");
        if (!btn) return;
        const cls    = btn.dataset.cls;
        const conf   = parseFloat(btn.dataset.conf);
        const risk   = parseInt(btn.dataset.risk, 10);
        const action = btn.dataset.action;
        if (cls) window.showExplanation(cls, conf, risk, action);
    });

    // ===== SAMPLE IMAGES =====
    const sampleGrid = document.getElementById("sampleGrid");
    const sampleClasses = ["center", "donut", "edge_loc", "edge_ring", "loc", "near_full", "random", "scratch", "none"];
    sampleClasses.forEach(c => {
        const img = document.createElement("img");
        img.src = `samples/${c}.png`;
        img.className = "sample-img";
        img.title = c.replace("_", " ");
        img.onerror = () => img.style.display = "none";
        img.addEventListener("click", () => {
            document.querySelector('[data-target="tab-dashboard"]').click();
            fetch(img.src).then(r => r.blob()).then(blob => {
                handleFiles([new File([blob], `${c}.png`, { type: "image/png" })]);
            }).catch(e => console.error("Sample load failed", e));
        });
        if (sampleGrid) sampleGrid.appendChild(img);
    });

    // ===== UPLOAD ZONE =====
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const selectFileBtn = document.getElementById("selectFileBtn");
    const uploadEmpty = document.getElementById("uploadEmpty");
    const uploadPreview = document.getElementById("uploadPreview");
    const previewTitle = document.getElementById("previewTitle");
    const previewGrid = document.getElementById("previewGrid");
    const resetBtn = document.getElementById("resetBtn");
    const runInferenceBtn = document.getElementById("runInferenceBtn");

    const analysisModeToggle = document.getElementById("analysisModeToggle");
    const labelSingle = document.getElementById("labelSingle");
    const labelMulti = document.getElementById("labelMulti");

    analysisModeToggle.addEventListener("change", () => {
        if (analysisModeToggle.checked) {
            labelSingle.classList.remove("active");
            labelMulti.classList.add("active");
        } else {
            labelSingle.classList.add("active");
            labelMulti.classList.remove("active");
        }
    });

    const resultsZone = document.getElementById("resultsZone");
    const progressLabel = document.getElementById("progressLabel");
    const progressCount = document.getElementById("progressCount");
    const progressFill = document.getElementById("progressFill");
    const batchResultsContainer = document.getElementById("batchResultsContainer");
    const resultsControls = document.getElementById("resultsControls");

    let currentFiles = [];
    // Stores the raw result data objects for the current session
    let currentSessionResults = [];

    selectFileBtn.addEventListener("click", (e) => { e.preventDefault(); fileInput.click(); });
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
    dropZone.addEventListener("drop", e => { e.preventDefault(); dropZone.classList.remove("dragover"); if (e.dataTransfer.files.length) handleFiles(Array.from(e.dataTransfer.files)); });
    fileInput.addEventListener("change", (e) => { if (e.target.files.length) handleFiles(Array.from(e.target.files)); });

    function handleFiles(files) {
        const imageFiles = files.filter(f => f.type.match("image.*")).slice(0, 20);
        if (imageFiles.length === 0) return;
        currentFiles = imageFiles;

        previewGrid.innerHTML = "";
        imageFiles.forEach(file => {
            const reader = new FileReader();
            reader.onload = e => {
                const wrap = document.createElement("div");
                wrap.className = "preview-thumb";
                const img = document.createElement("img");
                img.src = e.target.result;
                img.alt = file.name;
                const lbl = document.createElement("div");
                lbl.className = "preview-thumb-label";
                lbl.title = file.name;
                lbl.textContent = file.name;
                wrap.appendChild(img);
                wrap.appendChild(lbl);
                previewGrid.appendChild(wrap);
            };
            reader.readAsDataURL(file);
        });

        const extra = files.length > 20 ? ` (${files.length - 20} ignored, max 20)` : "";
        previewTitle.textContent = `${imageFiles.length} image${imageFiles.length > 1 ? "s" : ""} selected${extra}`;
        uploadEmpty.classList.add("hidden");
        uploadPreview.classList.remove("hidden");
        resultsZone.classList.add("hidden");
        resultsControls.style.display = "none";
        batchResultsContainer.innerHTML = "";
        currentSessionResults = [];
    }

    resetBtn.addEventListener("click", () => {
        currentFiles = [];
        fileInput.value = "";
        previewGrid.innerHTML = "";
        batchResultsContainer.innerHTML = "";
        uploadPreview.classList.add("hidden");
        uploadEmpty.classList.remove("hidden");
        resultsZone.classList.add("hidden");
        resultsControls.style.display = "none";
        currentSessionResults = [];
    });

    // ===== BATCH INFERENCE =====
    runInferenceBtn.addEventListener("click", async () => {
        if (currentFiles.length === 0) return;

        runInferenceBtn.disabled = true;
        runInferenceBtn.textContent = "⏳ Running...";
        resultsControls.style.display = "none";
        currentSessionResults = [];

        resultsZone.classList.remove("hidden");
        batchResultsContainer.innerHTML = "";

        const total = currentFiles.length;
        let completed = 0;

        function updateProgress() {
            const pct = total > 0 ? (completed / total) * 100 : 0;
            progressCount.textContent = `${completed} / ${total}`;
            progressFill.style.width = pct + "%";
            if (completed < total) {
                progressLabel.textContent = `Analyzing image ${completed + 1} of ${total}...`;
            } else {
                progressLabel.textContent = `✅ Analysis complete — ${total} wafer${total > 1 ? "s" : ""} processed`;
            }
        }
        updateProgress();

        for (let i = 0; i < total; i++) {
            const file = currentFiles[i];

            const origUrl = await new Promise(res => {
                const r = new FileReader();
                r.onload = e => res(e.target.result);
                r.readAsDataURL(file);
            });

            try {
                const isMultiMode = analysisModeToggle.checked;
                const endpoint = isMultiMode ? "/predict_multi" : "/predict";

                const fd = new FormData();
                fd.append("image", file);
                const resp = await fetch(`${apiUrl}${endpoint}`, {
                    method: "POST",
                    body: fd,
                    headers: { "ngrok-skip-browser-warning": "true" }
                });
                if (!resp.ok) {
                    let errMsg = `HTTP ${resp.status}`;
                    try {
                        const errBody = await resp.json();
                        if (errBody.detail) errMsg = errBody.detail;
                    } catch (_) {}
                    throw new Error(errMsg);
                }
                const data = await resp.json();
                if (data.error) throw new Error(data.error);

                // Store result for history/PDF
                currentSessionResults.push({ filename: file.name, origUrl, data, isMulti: isMultiMode, error: null });

                if (isMultiMode) {
                    appendMultiResultRow(file.name, origUrl, data);
                } else {
                    appendResultRow(file.name, origUrl, data);
                }
            } catch (err) {
                currentSessionResults.push({ filename: file.name, origUrl, data: null, error: err.message });
                appendErrorRow(file.name, origUrl, err.message);
            }

            completed++;
            updateProgress();
        }

        runInferenceBtn.disabled = false;
        runInferenceBtn.textContent = "✨ Analyze All";
        // Show action buttons after run
        if (currentSessionResults.length > 0) {
            resultsControls.style.display = "flex";
        }
    });

    // ===== RESULT ROW BUILDERS =====
    function appendMultiResultRow(filename, origUrl, data) {
        const defects = data.detected_defects || [];
        const confMap = data.confidence_per_class || {};
        const camUrl = data.gradcam_png_base64 ? `data:image/png;base64,${data.gradcam_png_base64}` : null;

        const isClean = data.is_clean;
        const risk = data.risk_score !== undefined ? data.risk_score : 0;
        const action = data.action || "MONITOR";
        const isStop = action === "STOP LINE" || action === "STOP LOT" || action === "STOP";
        const isWarn = action === "INVESTIGATE";
        const rowClass = isClean ? "row-ok" : (isStop ? "row-danger" : (isWarn ? "row-warning" : "row-ok"));
        const badgeClass = isClean ? "ok" : (isStop ? "stop" : (isWarn ? "warn" : "ok"));
        const riskColorClass = risk < 30 ? "risk-low" : risk < 70 ? "risk-mid" : "risk-high";

        const camHTML = camUrl
            ? `<div class="batch-img-wrap"><img src="${camUrl}" alt="Grad-CAM" /><div class="img-label">CAM (Max Conf)</div></div>`
            : `<div class="batch-img-wrap" style="display:flex;align-items:center;justify-content:center;opacity:0.4;font-size:0.7rem;color:#888;">No CAM</div>`;

        let defTags = isClean
            ? `<span class="action-badge ok" style="font-size:0.7rem;">None</span>`
            : defects.map(d => `<span class="action-badge stop" style="font-size:0.7rem; margin-right:4px; margin-bottom:4px;">${d}</span>`).join("");

        let maxConf = 0;
        if (!isClean && defects.length > 0) {
            maxConf = Math.max(...defects.map(d => confMap[d] || 0));
        }
        const confPct = (maxConf * 100).toFixed(1);

        const row = document.createElement("div");
        row.className = `batch-row ${rowClass}`;
        row.innerHTML = `
            <div class="batch-images">
                <div class="batch-img-wrap">
                    <img src="${origUrl}" alt="Original" />
                    <div class="img-label">Original</div>
                </div>
                ${camHTML}
            </div>
            <div class="batch-info">
                <div class="batch-filename" title="${filename}">📄 ${filename}</div>
                <div style="margin-top: 6px; display: flex; flex-wrap: wrap;">${defTags}</div>
            </div>
            <div class="batch-confidence">
                <div class="conf-label">${isClean ? "Clean" : "Max Confidence"}</div>
                <div class="conf-pct">${isClean ? "—" : confPct + "%"}</div>
                ${!isClean ? `<div class="conf-bar-bg"><div class="conf-bar-fill" style="width: ${maxConf * 100}%"></div></div>` : ""}
            </div>
            <div class="batch-risk">
                ${isClean ? `<div style="color: var(--text-sec); font-size: 0.8rem; text-align: center;">Clean<br>Risk: 0</div>` : `
                <button class="risk-btn"
                    data-cls="Mixed (${defects.join(', ')})"
                    data-conf="${maxConf}"
                    data-risk="${risk}"
                    data-action="${action}">
                    <span class="risk-value ${riskColorClass}">${risk}</span>
                    <span class="risk-denom">/ 100</span>
                    <span class="risk-hint">▸ CLICK FOR DETAILS</span>
                </button>
                `}
            </div>
            <div class="batch-action">
                <span class="action-badge ${badgeClass}">${action}</span>
            </div>
        `;
        batchResultsContainer.appendChild(row);
    }

    function appendResultRow(filename, origUrl, data) {
        const cls = data.predicted_class || "Unknown";
        const conf = data.confidence || 0;
        const risk = data.risk_score !== undefined ? data.risk_score : 0;
        const action = data.action || "MONITOR";
        const camUrl = data.gradcam_png_base64 ? `data:image/png;base64,${data.gradcam_png_base64}` : null;
        const confPct = (conf * 100).toFixed(1);

        const isOk = cls.toLowerCase() === "none";
        const isStop = action === "STOP LINE" || action === "STOP LOT" || action === "STOP";
        const isWarn = action === "INVESTIGATE";
        const rowClass = isStop ? "row-danger" : (isWarn ? "row-warning" : "row-ok");
        const clsClass = isOk ? "cls-ok" : "cls-defect";
        const badgeClass = isStop ? "stop" : (isWarn ? "warn" : "ok");
        const riskColorClass = risk < 30 ? "risk-low" : risk < 70 ? "risk-mid" : "risk-high";

        const camHTML = camUrl
            ? `<div class="batch-img-wrap"><img src="${camUrl}" alt="Grad-CAM" /><div class="img-label">Grad-CAM</div></div>`
            : `<div class="batch-img-wrap" style="display:flex;align-items:center;justify-content:center;opacity:0.4;font-size:0.7rem;color:#888;">No CAM</div>`;

        const row = document.createElement("div");
        row.className = `batch-row ${rowClass}`;
        row.innerHTML = `
            <div class="batch-images">
                <div class="batch-img-wrap">
                    <img src="${origUrl}" alt="Original" />
                    <div class="img-label">Original</div>
                </div>
                ${camHTML}
            </div>
            <div class="batch-info">
                <div class="batch-filename" title="${filename}">📄 ${filename}</div>
                <div class="batch-classname ${clsClass}">${cls}</div>
            </div>
            <div class="batch-confidence">
                <div class="conf-label">Confidence</div>
                <div class="conf-pct">${confPct}%</div>
                <div class="conf-bar-bg"><div class="conf-bar-fill" style="width: ${conf * 100}%"></div></div>
            </div>
            <div class="batch-risk">
                <button class="risk-btn"
                    data-cls="${cls}"
                    data-conf="${conf}"
                    data-risk="${risk}"
                    data-action="${action}">
                    <span class="risk-value ${riskColorClass}">${risk}</span>
                    <span class="risk-denom">/ 100</span>
                    <span class="risk-hint">▸ CLICK FOR DETAILS</span>
                </button>
            </div>
            <div class="batch-action">
                <span class="action-badge ${badgeClass}">${action}</span>
            </div>
        `;
        batchResultsContainer.appendChild(row);
    }

    function appendErrorRow(filename, origUrl, errMsg) {
        const row = document.createElement("div");
        row.className = "batch-row row-danger";
        row.style.opacity = "0.6";
        row.innerHTML = `
            <div class="batch-images">
                <div class="batch-img-wrap"><img src="${origUrl}" alt="Input" /><div class="img-label">Input</div></div>
            </div>
            <div class="batch-info">
                <div class="batch-filename" title="${filename}">📄 ${filename}</div>
                <div class="batch-classname cls-defect" style="font-size:1rem;">⚠️ Inference Failed</div>
                <div style="font-size: 0.78rem; color: var(--danger); margin-top: 4px;">${errMsg}</div>
            </div>
            <div class="batch-confidence">—</div>
            <div class="batch-risk">—</div>
            <div class="batch-action"><span class="action-badge stop">ERROR</span></div>
        `;
        batchResultsContainer.appendChild(row);
    }

    // ===================================================
    // ===== HISTORY FEATURE =====
    // ===================================================
    const HISTORY_KEY = "wafermap_history";
    const MAX_HISTORY = 50; // max sessions stored

    function loadHistory() {
        try {
            return JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
        } catch { return []; }
    }

    function saveHistoryToStorage(history) {
        try {
            localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
        } catch (e) {
            console.warn("History save failed (storage quota?)", e);
        }
    }

    function updateHistoryBadge() {
        const history = loadHistory();
        const badge = document.getElementById("historySidebarBadge");
        if (history.length > 0) {
            badge.textContent = history.length;
            badge.style.display = "inline-block";
        } else {
            badge.style.display = "none";
        }
        const countLabel = document.getElementById("historyCountLabel");
        if (countLabel) countLabel.textContent = `${history.length} session${history.length !== 1 ? "s" : ""} saved`;
    }

    // Save current session button
    const saveHistoryBtn = document.getElementById("saveHistoryBtn");
    saveHistoryBtn.addEventListener("click", () => {
        if (currentSessionResults.length === 0) return;

        const history = loadHistory();
        const timestamp = new Date().toISOString();
        const mode = analysisModeToggle.checked ? "Multi-Defect" : "Single-Defect";
        const totalWafers = currentSessionResults.length;
        const defectCount = currentSessionResults.filter(r => {
            if (r.error) return false;
            if (r.isMulti) return !r.data.is_clean;
            return r.data && r.data.predicted_class && r.data.predicted_class.toLowerCase() !== "none";
        }).length;

        // Serialize results (strip large base64 gradcam to keep storage lean — store first 4 chars as placeholder flag)
        const serializableResults = currentSessionResults.map(r => {
            if (!r.data) return r;
            const d = { ...r.data };
            if (d.gradcam_png_base64) {
                d.gradcam_png_base64 = d.gradcam_png_base64.substring(0, 8) + "…TRUNCATED";
            }
            return { ...r, data: d };
        });

        const session = {
            id: Date.now(),
            timestamp,
            mode,
            totalWafers,
            defectCount,
            results: serializableResults
        };

        history.unshift(session);
        if (history.length > MAX_HISTORY) history.splice(MAX_HISTORY);
        saveHistoryToStorage(history);
        updateHistoryBadge();

        // Visual feedback
        saveHistoryBtn.textContent = "✅ Saved!";
        saveHistoryBtn.disabled = true;
        setTimeout(() => {
            saveHistoryBtn.textContent = "🕓 Save to History";
            saveHistoryBtn.disabled = false;
        }, 2000);
    });

    function renderHistoryList() {
        const history = loadHistory();
        const historyList = document.getElementById("historyList");
        const historyEmpty = document.getElementById("historyEmpty");
        updateHistoryBadge();

        if (history.length === 0) {
            historyList.innerHTML = "";
            historyEmpty.style.display = "block";
            return;
        }
        historyEmpty.style.display = "none";
        historyList.innerHTML = "";

        history.forEach(session => {
            const date = new Date(session.timestamp);
            const dateStr = date.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
            const timeStr = date.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });

            const cleanCount = session.totalWafers - session.defectCount;
            const defectRatio = session.totalWafers > 0 ? (session.defectCount / session.totalWafers * 100).toFixed(0) : 0;

            const card = document.createElement("div");
            card.className = "history-card glass-panel";
            card.innerHTML = `
                <div class="history-card-header">
                    <div class="history-card-meta">
                        <span class="history-mode-badge ${session.mode === "Multi-Defect" ? "multi" : "single"}">${session.mode}</span>
                        <span class="history-date">${dateStr} · ${timeStr}</span>
                    </div>
                    <div class="history-card-actions">
                        <button class="btn btn-secondary history-view-btn" data-id="${session.id}" style="padding:0.4rem 0.9rem; font-size:0.82rem;">🔍 View</button>
                        <button class="btn btn-secondary history-export-btn" data-id="${session.id}" style="padding:0.4rem 0.9rem; font-size:0.82rem;">📄 PDF</button>
                        <button class="history-delete-btn" data-id="${session.id}" title="Delete session">🗑</button>
                    </div>
                </div>
                <div class="history-card-stats">
                    <div class="history-stat">
                        <span class="history-stat-value">${session.totalWafers}</span>
                        <span class="history-stat-label">Wafers</span>
                    </div>
                    <div class="history-stat">
                        <span class="history-stat-value" style="color:var(--danger);">${session.defectCount}</span>
                        <span class="history-stat-label">Defects</span>
                    </div>
                    <div class="history-stat">
                        <span class="history-stat-value" style="color:var(--success);">${cleanCount}</span>
                        <span class="history-stat-label">Clean</span>
                    </div>
                    <div class="history-stat">
                        <div class="history-yield-bar">
                            <div class="history-yield-fill" style="width: ${100 - defectRatio}%"></div>
                        </div>
                        <span class="history-stat-label">${100 - defectRatio}% Yield</span>
                    </div>
                </div>
            `;
            historyList.appendChild(card);
        });

        // Bind view buttons
        historyList.querySelectorAll(".history-view-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const session = history.find(s => s.id == btn.dataset.id);
                if (session) openHistoryDetail(session);
            });
        });

        // Bind export buttons
        historyList.querySelectorAll(".history-export-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const session = history.find(s => s.id == btn.dataset.id);
                if (session) exportSessionToPdf(session);
            });
        });

        // Bind delete buttons
        historyList.querySelectorAll(".history-delete-btn").forEach(btn => {
            btn.addEventListener("click", () => {
                const h = loadHistory().filter(s => s.id != btn.dataset.id);
                saveHistoryToStorage(h);
                renderHistoryList();
            });
        });
    }

    // History Detail Modal
    const historyDetailOverlay = document.getElementById("historyDetailOverlay");
    const closeHistoryDetailBtn = document.getElementById("closeHistoryDetailBtn");
    closeHistoryDetailBtn.addEventListener("click", () => historyDetailOverlay.classList.remove("active"));
    historyDetailOverlay.addEventListener("click", (e) => { if (e.target === historyDetailOverlay) historyDetailOverlay.classList.remove("active"); });

    function openHistoryDetail(session) {
        const date = new Date(session.timestamp);
        document.getElementById("historyDetailSubtitle").textContent =
            `${session.mode} · ${date.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" })} ${date.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" })} · ${session.totalWafers} wafers`;

        const body = document.getElementById("historyDetailBody");
        body.innerHTML = "";

        session.results.forEach(r => {
            const item = document.createElement("div");
            item.className = "history-detail-item";

            if (r.error) {
                item.innerHTML = `
                    <div class="history-detail-thumb" style="background:#111; display:flex; align-items:center; justify-content:center; color:var(--danger); font-size:1.3rem;">⚠️</div>
                    <div class="history-detail-info">
                        <div class="history-detail-filename">${r.filename}</div>
                        <div style="color:var(--danger); font-size:0.82rem; margin-top:3px;">${r.error}</div>
                    </div>
                    <span class="action-badge stop" style="flex-shrink:0;">ERROR</span>
                `;
            } else if (r.isMulti) {
                const defects = r.data.detected_defects || [];
                const isClean = r.data.is_clean;
                item.innerHTML = `
                    <div class="history-detail-thumb" style="background:#111; display:flex; align-items:center; justify-content:center; font-size:1.2rem;">${isClean ? "✅" : "🔴"}</div>
                    <div class="history-detail-info">
                        <div class="history-detail-filename">${r.filename}</div>
                        <div style="margin-top:4px;">${isClean ? '<span class="action-badge ok" style="font-size:0.7rem;">Clean</span>' : defects.map(d => `<span class="action-badge stop" style="font-size:0.7rem; margin-right:3px;">${d}</span>`).join("")}</div>
                    </div>
                    <span class="action-badge ${isClean ? "ok" : "stop"}" style="flex-shrink:0;">${isClean ? "CLEAN" : "DEFECT"}</span>
                `;
            } else {
                const cls = r.data.predicted_class || "Unknown";
                const conf = ((r.data.confidence || 0) * 100).toFixed(1);
                const action = r.data.action || "MONITOR";
                const isStop = action === "STOP LOT" || action === "STOP";
                const isWarn = action === "INVESTIGATE";
                const badgeClass = isStop ? "stop" : (isWarn ? "warn" : "ok");
                item.innerHTML = `
                    <div class="history-detail-thumb" style="background:#111; display:flex; align-items:center; justify-content:center; font-size:1.2rem;">${cls.toLowerCase() === "none" ? "✅" : "🔴"}</div>
                    <div class="history-detail-info">
                        <div class="history-detail-filename">${r.filename}</div>
                        <div style="font-weight:700; margin-top:2px; color:${cls.toLowerCase() === "none" ? "var(--success)" : "var(--danger)"};">${cls}</div>
                        <div style="font-size:0.8rem; color:var(--text-sec);">Confidence: ${conf}% · Risk: ${r.data.risk_score ?? "—"}/100</div>
                    </div>
                    <span class="action-badge ${badgeClass}" style="flex-shrink:0;">${action}</span>
                `;
            }
            body.appendChild(item);
        });

        historyDetailOverlay.classList.add("active");
    }

    // Clear history button
    document.getElementById("clearHistoryBtn").addEventListener("click", () => {
        if (confirm("Clear all history? This cannot be undone.")) {
            saveHistoryToStorage([]);
            renderHistoryList();
        }
    });

    // Export ALL history button
    document.getElementById("exportAllHistoryBtn").addEventListener("click", () => {
        const history = loadHistory();
        if (history.length === 0) { alert("No history to export."); return; }
        exportAllHistoryToPdf(history);
    });

    // ===================================================
    // ===== PDF EXPORT =====
    // ===================================================
    const exportPdfBtn = document.getElementById("exportPdfBtn");
    exportPdfBtn.addEventListener("click", () => {
        if (currentSessionResults.length === 0) return;
        const session = {
            timestamp: new Date().toISOString(),
            mode: analysisModeToggle.checked ? "Multi-Defect" : "Single-Defect",
            results: currentSessionResults,
            totalWafers: currentSessionResults.length,
            defectCount: currentSessionResults.filter(r => {
                if (r.error) return false;
                if (r.isMulti) return !r.data.is_clean;
                return r.data && r.data.predicted_class && r.data.predicted_class.toLowerCase() !== "none";
            }).length
        };
        exportSessionToPdf(session);
    });

    function exportSessionToPdf(session) {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });

        const pageW = 210, pageH = 297, margin = 15;
        const contentW = pageW - margin * 2;
        const date = new Date(session.timestamp);
        const dateStr = date.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
        const timeStr = date.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });

        let y = margin;

        // ---- HEADER ----
        pdf.setFillColor(20, 10, 40);
        pdf.rect(0, 0, pageW, 40, "F");
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(20);
        pdf.setTextColor(200, 170, 255);
        pdf.text("⚡ WaferMap AI", margin, 17);
        pdf.setFontSize(10);
        pdf.setTextColor(160, 140, 200);
        pdf.text("Semiconductor Defect Intelligence Report", margin, 25);
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(9);
        pdf.setTextColor(120, 110, 160);
        pdf.text(`Generated: ${dateStr} at ${timeStr}  ·  Mode: ${session.mode}`, margin, 33);

        y = 50;

        // ---- SUMMARY BOX ----
        const cleanCount = session.totalWafers - session.defectCount;
        const yieldPct = session.totalWafers > 0 ? ((cleanCount / session.totalWafers) * 100).toFixed(1) : "0.0";

        pdf.setFillColor(30, 20, 55);
        pdf.roundedRect(margin, y, contentW, 30, 4, 4, "F");
        pdf.setDrawColor(80, 60, 140);
        pdf.roundedRect(margin, y, contentW, 30, 4, 4, "S");

        const col = contentW / 3;
        const statY = y + 12;
        [
            { label: "Total Wafers", value: session.totalWafers, color: [200, 170, 255] },
            { label: "Defects Found", value: session.defectCount, color: [231, 76, 60] },
            { label: "Yield Estimate", value: yieldPct + "%", color: [46, 204, 113] }
        ].forEach((stat, i) => {
            const cx = margin + col * i + col / 2;
            pdf.setFont("helvetica", "bold");
            pdf.setFontSize(16);
            pdf.setTextColor(...stat.color);
            pdf.text(String(stat.value), cx, statY, { align: "center" });
            pdf.setFont("helvetica", "normal");
            pdf.setFontSize(8);
            pdf.setTextColor(160, 140, 200);
            pdf.text(stat.label, cx, statY + 7, { align: "center" });
        });

        y += 40;

        // ---- RESULTS TABLE ----
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(11);
        pdf.setTextColor(200, 170, 255);
        pdf.text("Analysis Results", margin, y);
        y += 5;

        // Table header
        const cols = { file: margin, cls: margin + 65, conf: margin + 120, risk: margin + 150, action: margin + 172 };
        const rowH = 8;
        pdf.setFillColor(40, 25, 70);
        pdf.rect(margin, y, contentW, rowH, "F");
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(7.5);
        pdf.setTextColor(160, 140, 200);
        pdf.text("File", cols.file + 2, y + 5.5);
        pdf.text("Classification", cols.cls, y + 5.5);
        pdf.text("Confidence", cols.conf, y + 5.5);
        pdf.text("Risk", cols.risk, y + 5.5);
        pdf.text("Action", cols.action, y + 5.5);
        y += rowH;

        session.results.forEach((r, idx) => {
            if (y + rowH > pageH - margin) {
                pdf.addPage();
                y = margin;
            }

            // Alternating background
            if (idx % 2 === 0) {
                pdf.setFillColor(22, 14, 42);
                pdf.rect(margin, y, contentW, rowH, "F");
            }

            let clsText = "—", confText = "—", riskText = "—", actionText = "—";
            let actionColor = [46, 204, 113];

            if (r.error) {
                clsText = "ERROR"; actionText = "FAILED";
                actionColor = [231, 76, 60];
            } else if (r.isMulti) {
                const defects = (r.data.detected_defects || []);
                clsText = r.data.is_clean ? "Clean" : defects.slice(0, 3).join(", ") + (defects.length > 3 ? "…" : "");
                actionText = r.data.is_clean ? "MONITOR" : "STOP";
                actionColor = r.data.is_clean ? [46, 204, 113] : [231, 76, 60];
            } else if (r.data) {
                clsText = r.data.predicted_class || "Unknown";
                confText = ((r.data.confidence || 0) * 100).toFixed(1) + "%";
                riskText = String(r.data.risk_score ?? "—");
                actionText = r.data.action || "MONITOR";
                const isStop = actionText === "STOP LOT" || actionText === "STOP";
                const isWarn = actionText === "INVESTIGATE";
                actionColor = isStop ? [231, 76, 60] : isWarn ? [245, 176, 65] : [46, 204, 113];
            }

            pdf.setFont("helvetica", "normal");
            pdf.setFontSize(7);
            pdf.setTextColor(220, 220, 230);

            // File name (truncate)
            const maxFilenameWidth = 60;
            const truncFile = pdf.getStringUnitWidth(r.filename) * 7 / pdf.internal.scaleFactor > maxFilenameWidth
                ? r.filename.substring(0, 28) + "…"
                : r.filename;

            pdf.text(truncFile, cols.file + 2, y + 5.5);
            pdf.text(clsText, cols.cls, y + 5.5);
            pdf.text(confText, cols.conf, y + 5.5);
            pdf.text(riskText, cols.risk, y + 5.5);

            pdf.setFont("helvetica", "bold");
            pdf.setTextColor(...actionColor);
            pdf.text(actionText, cols.action, y + 5.5);

            // Row separator
            pdf.setDrawColor(40, 30, 65);
            pdf.line(margin, y + rowH, margin + contentW, y + rowH);

            y += rowH;
        });

        y += 8;

        // ---- FOOTER ----
        if (y + 20 > pageH - margin) { pdf.addPage(); y = margin; }
        pdf.setDrawColor(80, 60, 140);
        pdf.line(margin, y, margin + contentW, y);
        y += 5;
        pdf.setFont("helvetica", "italic");
        pdf.setFontSize(8);
        pdf.setTextColor(100, 90, 140);
        pdf.text("WaferMap AI · EfficientNet-B0 powered · SanDisk Hackathon 2025", pageW / 2, y + 4, { align: "center" });
        pdf.text("This report is AI-generated and should be reviewed by a qualified process engineer.", pageW / 2, y + 9, { align: "center" });

        const filename = `wafermap_report_${date.toISOString().slice(0, 10)}.pdf`;
        pdf.save(filename);
    }

    function exportAllHistoryToPdf(history) {
        const { jsPDF } = window.jspdf;
        const pdf = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
        const pageW = 210, pageH = 297, margin = 15;
        const contentW = pageW - margin * 2;

        // Cover page
        pdf.setFillColor(10, 5, 25);
        pdf.rect(0, 0, pageW, pageH, "F");
        pdf.setFont("helvetica", "bold");
        pdf.setFontSize(28);
        pdf.setTextColor(180, 140, 255);
        pdf.text("WaferMap AI", pageW / 2, 90, { align: "center" });
        pdf.setFontSize(14);
        pdf.setTextColor(130, 100, 210);
        pdf.text("Full History Export", pageW / 2, 104, { align: "center" });
        pdf.setFont("helvetica", "normal");
        pdf.setFontSize(10);
        pdf.setTextColor(100, 80, 160);
        pdf.text(`${history.length} sessions · Exported ${new Date().toLocaleDateString()}`, pageW / 2, 115, { align: "center" });

        history.forEach((session, idx) => {
            pdf.addPage();
            let y = margin;

            const date = new Date(session.timestamp);
            const dateStr = date.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
            const timeStr = date.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit" });

            pdf.setFillColor(20, 10, 40);
            pdf.rect(0, 0, pageW, 30, "F");
            pdf.setFont("helvetica", "bold");
            pdf.setFontSize(13);
            pdf.setTextColor(200, 170, 255);
            pdf.text(`Session ${idx + 1} — ${dateStr} at ${timeStr}`, margin, 14);
            pdf.setFont("helvetica", "normal");
            pdf.setFontSize(9);
            pdf.setTextColor(140, 120, 190);
            pdf.text(`Mode: ${session.mode}  ·  ${session.totalWafers} wafers  ·  ${session.defectCount} defects`, margin, 22);

            y = 40;

            const cleanCount = session.totalWafers - session.defectCount;
            const yieldPct = session.totalWafers > 0 ? ((cleanCount / session.totalWafers) * 100).toFixed(1) : "0.0";

            // Table header
            const rowH = 8;
            pdf.setFillColor(40, 25, 70);
            pdf.rect(margin, y, contentW, rowH, "F");
            pdf.setFont("helvetica", "bold");
            pdf.setFontSize(7.5);
            pdf.setTextColor(160, 140, 200);
            pdf.text("File", margin + 2, y + 5.5);
            pdf.text("Classification", margin + 65, y + 5.5);
            pdf.text("Confidence", margin + 120, y + 5.5);
            pdf.text("Risk", margin + 150, y + 5.5);
            pdf.text("Action", margin + 172, y + 5.5);
            y += rowH;

            session.results.forEach((r, i) => {
                if (y + rowH > pageH - margin) { pdf.addPage(); y = margin; }
                if (i % 2 === 0) { pdf.setFillColor(22, 14, 42); pdf.rect(margin, y, contentW, rowH, "F"); }

                let cls = "—", conf = "—", risk = "—", action = "—";
                let ac = [46, 204, 113];

                if (r.error) { cls = "ERROR"; action = "FAILED"; ac = [231, 76, 60]; }
                else if (r.isMulti) {
                    const def = (r.data.detected_defects || []);
                    cls = r.data.is_clean ? "Clean" : def.slice(0, 3).join(", ") + (def.length > 3 ? "…" : "");
                    action = r.data.is_clean ? "MONITOR" : "STOP";
                    ac = r.data.is_clean ? [46, 204, 113] : [231, 76, 60];
                } else if (r.data) {
                    cls = r.data.predicted_class || "?";
                    conf = ((r.data.confidence || 0) * 100).toFixed(1) + "%";
                    risk = String(r.data.risk_score ?? "—");
                    action = r.data.action || "MONITOR";
                    const isStop = action === "STOP LOT" || action === "STOP";
                    ac = isStop ? [231, 76, 60] : action === "INVESTIGATE" ? [245, 176, 65] : [46, 204, 113];
                }

                pdf.setFont("helvetica", "normal"); pdf.setFontSize(7); pdf.setTextColor(220, 220, 230);
                const fn = r.filename.length > 30 ? r.filename.substring(0, 28) + "…" : r.filename;
                pdf.text(fn, margin + 2, y + 5.5);
                pdf.text(cls, margin + 65, y + 5.5);
                pdf.text(conf, margin + 120, y + 5.5);
                pdf.text(risk, margin + 150, y + 5.5);
                pdf.setFont("helvetica", "bold"); pdf.setTextColor(...ac);
                pdf.text(action, margin + 172, y + 5.5);
                pdf.setDrawColor(40, 30, 65); pdf.line(margin, y + rowH, margin + contentW, y + rowH);
                y += rowH;
            });

            y += 5;
            pdf.setFont("helvetica", "normal"); pdf.setFontSize(9); pdf.setTextColor(46, 204, 113);
            pdf.text(`Yield: ${yieldPct}% (${cleanCount} clean / ${session.totalWafers} total)`, margin, y + 5);
        });

        pdf.save(`wafermap_history_${new Date().toISOString().slice(0, 10)}.pdf`);
    }

    // ===================================================
    // ===== PERFORMANCE METRICS TAB =====
    // ===================================================

    // --- Raw metrics data (from reports/*.json) ---
    const SINGLE_METRICS = {
        test_accuracy: 88.57,
        macro_f1: 0.8826,
        best_val_accuracy: 86.71,
        training_epochs: 10,
        train_acc: [0.241, 0.346, 0.420, 0.591, 0.710, 0.804, 0.846, 0.862, 0.872, 0.877],
        val_acc:   [0.304, 0.386, 0.478, 0.645, 0.783, 0.838, 0.862, 0.865, 0.867, 0.865],
        per_class: {
            "Center":    { precision: 0.869,  recall: 0.8795, f1: 0.8743, support: 83 },
            "Donut":     { precision: 0.9346, recall: 0.9346, f1: 0.9346, support: 107 },
            "Edge-Loc":  { precision: 0.8447, recall: 0.8614, f1: 0.8529, support: 101 },
            "Edge-Ring": { precision: 0.9703, recall: 0.9333, f1: 0.9515, support: 105 },
            "Loc":       { precision: 0.7808, recall: 0.6951, f1: 0.7355, support: 82 },
            "Near-Full": { precision: 0.8333, recall: 1.0,    f1: 0.9091, support: 30 },
            "Random":    { precision: 0.8879, recall: 0.9035, f1: 0.8957, support: 114 },
            "Scratch":   { precision: 0.8667, recall: 0.91,   f1: 0.8878, support: 100 },
            "None":      { precision: 0.9151, recall: 0.8899, f1: 0.9023, support: 109 }
        }
    };

    const MIXED_METRICS = {
        macro_f1: 0.8806,
        hamming_loss: 0.0793,
        best_val_f1: 0.8514,
        epochs_run: 10,
        train_f1:  [0.360, 0.524, 0.608, 0.658, 0.734, 0.787, 0.822, 0.842, 0.860, 0.857],
        val_f1:    [0.464, 0.624, 0.642, 0.693, 0.745, 0.783, 0.802, 0.819, 0.851, 0.843],
        val_hamming: [0.352, 0.271, 0.255, 0.200, 0.164, 0.120, 0.103, 0.096, 0.086, 0.086],
        per_class: {
            "Center":    { precision: 1.0,    recall: 0.9915, f1: 0.9957, support: 117 },
            "Donut":     { precision: 1.0,    recall: 1.0,    f1: 1.0,    support: 120 },
            "Edge-Loc":  { precision: 0.6149, recall: 0.8492, f1: 0.7133, support: 126 },
            "Edge-Ring": { precision: 0.8024, recall: 1.0,    f1: 0.8904, support: 134 },
            "Loc":       { precision: 0.9286, recall: 0.8387, f1: 0.8814, support: 186 },
            "Near-Full": { precision: 0.6,    recall: 1.0,    f1: 0.75,   support: 6 },
            "Random":    { precision: 0.7162, recall: 0.9425, f1: 0.8139, support: 174 },
            "Scratch":   { precision: 1.0,    recall: 1.0,    f1: 1.0,    support: 12 }
        }
    };

    const chartDefaults = {
        grid: 'rgba(255,255,255,0.06)',
        tick: '#888',
        font: "'Inter', sans-serif"
    };

    function makeBarChart(canvasId, labels, values, label, colorFn) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        const colors = values.map(colorFn);
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels,
                datasets: [{
                    label,
                    data: values,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c.replace('0.75', '1')),
                    borderWidth: 1,
                    borderRadius: 5,
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: ctx => ` ${(ctx.parsed.y * 100).toFixed(1)}%`
                        }
                    }
                },
                scales: {
                    x: { grid: { color: chartDefaults.grid }, ticks: { color: chartDefaults.tick, font: { family: chartDefaults.font, size: 11 } } },
                    y: {
                        min: 0.5, max: 1,
                        grid: { color: chartDefaults.grid },
                        ticks: { color: chartDefaults.tick, callback: v => (v * 100).toFixed(0) + '%', font: { family: chartDefaults.font } }
                    }
                }
            }
        });
    }

    function makeLineChart(canvasId, epochs, datasets) {
        const ctx = document.getElementById(canvasId);
        if (!ctx) return;
        const labels = Array.from({ length: epochs }, (_, i) => `E${i + 1}`);
        new Chart(ctx, {
            type: 'line',
            data: { labels, datasets },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { labels: { color: '#ccc', font: { family: chartDefaults.font, size: 11 } } } },
                scales: {
                    x: { grid: { color: chartDefaults.grid }, ticks: { color: chartDefaults.tick, font: { family: chartDefaults.font } } },
                    y: {
                        min: 0, max: 1,
                        grid: { color: chartDefaults.grid },
                        ticks: { color: chartDefaults.tick, callback: v => (v * 100).toFixed(0) + '%', font: { family: chartDefaults.font } }
                    }
                }
            }
        });
    }

    function f1Color(f1) {
        if (f1 >= 0.92) return 'rgba(46,204,113,0.75)';
        if (f1 >= 0.85) return 'rgba(130,100,255,0.75)';
        if (f1 >= 0.75) return 'rgba(245,176,65,0.75)';
        return 'rgba(231,76,60,0.75)';
    }

    function fillMetricsTable(tableId, perClass) {
        const tbody = document.querySelector(`#${tableId} tbody`);
        if (!tbody) return;
        tbody.innerHTML = '';
        Object.entries(perClass).forEach(([cls, m]) => {
            const f1Pct = (m.f1 * 100).toFixed(1);
            const col = f1Color(m.f1);
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td class="cls-name">${cls}</td>
                <td>${(m.precision * 100).toFixed(1)}%</td>
                <td>${(m.recall * 100).toFixed(1)}%</td>
                <td style="font-weight:700; color:${col.replace('0.75','1')}">${f1Pct}%</td>
                <td style="color:var(--text-sec)">${m.support}</td>
                <td class="f1-bar-wrap">
                    <div class="f1-mini-bar">
                        <div class="f1-mini-fill" style="width:${f1Pct}%; background:${col.replace('0.75','1')}"></div>
                    </div>
                </td>
            `;
            tbody.appendChild(tr);
        });
    }

    let singleBuilt = false;
    let mixedBuilt  = false;


    function buildSingleCharts() {
        if (singleBuilt) return;
        singleBuilt = true;

        const singleClasses = Object.keys(SINGLE_METRICS.per_class);
        const singleF1s = singleClasses.map(c => SINGLE_METRICS.per_class[c].f1);
        makeBarChart('singleF1Chart', singleClasses, singleF1s, 'F1 Score', f1Color);

        makeLineChart('singleAccChart', 10, [
            { label: 'Train Accuracy', data: SINGLE_METRICS.train_acc, borderColor: 'hsl(250,90%,65%)', backgroundColor: 'hsla(250,90%,65%,0.1)', tension: 0.4, fill: true, pointRadius: 3 },
            { label: 'Val Accuracy',   data: SINGLE_METRICS.val_acc,   borderColor: '#2ecc71',           backgroundColor: 'rgba(46,204,113,0.1)',   tension: 0.4, fill: true, pointRadius: 3 }
        ]);

        fillMetricsTable('singleMetricsTable', SINGLE_METRICS.per_class);
    }

    function buildMixedCharts() {
        if (mixedBuilt) return;
        mixedBuilt = true;

        const mixedClasses = Object.keys(MIXED_METRICS.per_class);
        const mixedF1s = mixedClasses.map(c => MIXED_METRICS.per_class[c].f1);
        makeBarChart('mixedF1Chart', mixedClasses, mixedF1s, 'F1 Score', f1Color);

        makeLineChart('mixedValChart', 10, [
            { label: 'Val F1 (Multi-Label)',  data: MIXED_METRICS.val_f1,     borderColor: '#2ecc71',  backgroundColor: 'rgba(46,204,113,0.1)', tension: 0.4, fill: true,  pointRadius: 3 },
            { label: 'Hamming Loss (↓ better)', data: MIXED_METRICS.val_hamming, borderColor: '#e74c3c', backgroundColor: 'rgba(231,76,60,0.1)',  tension: 0.4, fill: false, pointRadius: 3, borderDash: [4, 3] }
        ]);

        fillMetricsTable('mixedMetricsTable', MIXED_METRICS.per_class);
    }

    // Engine switcher — also triggers lazy chart build for whichever panel becomes visible
    document.querySelectorAll('.engine-tab').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.engine-tab').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            const eng = btn.dataset.engine;
            const singlePanel = document.getElementById('metrics-single');
            const mixedPanel  = document.getElementById('metrics-mixed');
            singlePanel.style.display = eng === 'single' ? '' : 'none';
            mixedPanel.style.display  = eng === 'mixed'  ? '' : 'none';
            // Build charts only now — canvas is visible so Chart.js can measure dimensions
            if (eng === 'single') buildSingleCharts();
            if (eng === 'mixed')  buildMixedCharts();
        });
    });

    // Build single charts when Performance tab is first opened
    document.querySelectorAll('.tab-link').forEach(link => {
        link.addEventListener('click', () => {
            if (link.dataset.target === 'tab-metrics') buildSingleCharts();
        });
    });

    // ===== INIT =====
    updateHistoryBadge();
});
