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

        // Determine risk colour
        let riskColor = "var(--success)";
        if (action === "STOP LOT" || action === "STOP") riskColor = "var(--danger)";
        else if (action === "INVESTIGATE") riskColor = "var(--warning)";

        // Build explanation steps specific to the defect type + action
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

    // Analysis Mode Toggle
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

    // Results
    const resultsZone = document.getElementById("resultsZone");
    const progressLabel = document.getElementById("progressLabel");
    const progressCount = document.getElementById("progressCount");
    const progressFill = document.getElementById("progressFill");
    const batchResultsContainer = document.getElementById("batchResultsContainer");

    let currentFiles = [];

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
        batchResultsContainer.innerHTML = "";
    }

    resetBtn.addEventListener("click", () => {
        currentFiles = [];
        fileInput.value = "";
        previewGrid.innerHTML = "";
        batchResultsContainer.innerHTML = "";
        uploadPreview.classList.add("hidden");
        uploadEmpty.classList.remove("hidden");
        resultsZone.classList.add("hidden");
    });

    // ===== BATCH INFERENCE =====
    runInferenceBtn.addEventListener("click", async () => {
        if (currentFiles.length === 0) return;

        // Disable button during run
        runInferenceBtn.disabled = true;
        runInferenceBtn.textContent = "⏳ Running...";

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

            // Read file as data URL for original display
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
                    // Try to get the friendly detail message from a FastAPI HTTPException
                    let errMsg = `HTTP ${resp.status}`;
                    try {
                        const errBody = await resp.json();
                        if (errBody.detail) errMsg = errBody.detail;
                    } catch (_) {}
                    throw new Error(errMsg);
                }
                const data = await resp.json();
                if (data.error) throw new Error(data.error);
                
                if (isMultiMode) {
                    appendMultiResultRow(file.name, origUrl, data);
                } else {
                    appendResultRow(file.name, origUrl, data);
                }
            } catch (err) {
                appendErrorRow(file.name, origUrl, err.message);
            }

            completed++;
            updateProgress();
        }

        runInferenceBtn.disabled = false;
        runInferenceBtn.textContent = "✨ Analyze All";
    });

    function appendMultiResultRow(filename, origUrl, data) {
        const defects = data.detected_defects || [];
        const confMap = data.confidence_per_class || {};
        const camUrl = data.gradcam_png_base64 ? `data:image/png;base64,${data.gradcam_png_base64}` : null;
        
        // Styling based on cleanliness
        const isClean = data.is_clean;
        const rowClass = isClean ? "row-ok" : "row-danger";
        const badgeClass = isClean ? "ok" : "stop";
        const actionText = isClean ? "MONITOR" : "STOP (MULTI)";

        const camHTML = camUrl
            ? `<div class="batch-img-wrap"><img src="${camUrl}" alt="Grad-CAM" /><div class="img-label">CAM (Max Conf)</div></div>`
            : `<div class="batch-img-wrap" style="display:flex;align-items:center;justify-content:center;opacity:0.4;font-size:0.7rem;color:#888;">No CAM</div>`;

        // Format defect tags
        let defTags = isClean 
            ? `<span class="action-badge ok" style="font-size:0.7rem;">None</span>`
            : defects.map(d => `<span class="action-badge stop" style="font-size:0.7rem; margin-right:4px; margin-bottom:4px;">${d}</span>`).join("");
            
        // Max confidence
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
                <div style="color: var(--text-sec); font-size: 0.8rem; text-align: center;">Multi-Label Mode<br>Risk Scorer N/A</div>
            </div>
            <div class="batch-action">
                <span class="action-badge ${badgeClass}">${actionText}</span>
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

        // Row styling
        const isOk = cls.toLowerCase() === "none";
        const isStop = action === "STOP LOT" || action === "STOP";
        const isWarn = action === "INVESTIGATE";
        const rowClass = isStop ? "row-danger" : (isWarn ? "row-warning" : "row-ok");
        const clsClass = isOk ? "cls-ok" : "cls-defect";
        const badgeClass = isStop ? "stop" : (isWarn ? "warn" : "ok");

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
                <button class="risk-btn" onclick="showExplanation('${cls}', ${conf}, ${risk}, '${action}')">
                    <span class="risk-value">${risk}</span>
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
});
