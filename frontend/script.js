document.addEventListener("DOMContentLoaded", () => {
    // --- Layout & Sidebar logic ---
    const sidebar = document.getElementById("sidebar");
    const sidebarToggle = document.getElementById("sidebarToggle");
    const tabLinks = document.querySelectorAll(".tab-link");
    const tabPanes = document.querySelectorAll(".tab-pane");

    // Toggle Sidebar
    sidebarToggle.addEventListener("click", () => {
        sidebar.classList.toggle("collapsed");
    });

    // Tab Switching
    tabLinks.forEach(link => {
        link.addEventListener("click", () => {
            // Remove active class from all
            tabLinks.forEach(l => l.classList.remove("active"));
            tabPanes.forEach(p => p.classList.remove("active"));
            // Add active to current
            link.classList.add("active");
            const target = document.getElementById(link.dataset.target);
            if (target) target.classList.add("active");
        });
    });

    // --- Modal Config Logic ---
    const configToggleBtn = document.getElementById("configToggleBtn");
    const configOverlay = document.getElementById("configOverlay");
    const closeConfigBtn = document.getElementById("closeConfigBtn");
    
    // Explanation Modal Elements
    const explanationOverlay = document.getElementById("explanationOverlay");
    const closeExplanationBtn = document.getElementById("closeExplanationBtn");
    const explanationText = document.getElementById("explanationText");

    const saveApiBtn = document.getElementById("saveApiBtn");
    const apiUrlInput = document.getElementById("apiUrlInput");
    const apiStatusDisplay = document.getElementById("apiStatusDisplay");
    
    // Status Badge
    const serverStatusBadge = document.getElementById("serverStatusBadge");
    const statusDot = serverStatusBadge.querySelector(".status-dot");
    const statusText = document.getElementById("statusText");

    let apiUrl = localStorage.getItem("waferApiUrl") || "http://localhost:8000";
    apiUrlInput.value = apiUrl;
    
    // Open/Close modal
    configToggleBtn.addEventListener("click", () => {
        configOverlay.classList.add("active");
    });
    closeConfigBtn.addEventListener("click", () => {
        configOverlay.classList.remove("active");
    });
    // Optional: click outside to close
    configOverlay.addEventListener("click", (e) => {
        if(e.target === configOverlay) configOverlay.classList.remove("active");
    });

    saveApiBtn.addEventListener("click", () => {
        apiUrl = apiUrlInput.value.trim().replace(/\/$/, ""); 
        localStorage.setItem("waferApiUrl", apiUrl);
        configOverlay.classList.remove("active");
        checkServerRealtime();
    });

    // Explanation Modal Close
    closeExplanationBtn.addEventListener("click", () => {
        explanationOverlay.classList.remove("active");
    });
    explanationOverlay.addEventListener("click", (e) => {
        if(e.target === explanationOverlay) explanationOverlay.classList.remove("active");
    });

    // Ping the backend /health to reassure the user
    async function checkServerRealtime() {
        statusDot.className = "status-dot"; // reset
        statusText.innerText = "Checking...";
        try {
            // we use the health endpoint instead of predict
            const res = await fetch(`${apiUrl}/health`, { 
                method: "GET",
                headers: { "ngrok-skip-browser-warning": "true" }
            });
            if (res.ok) {
                statusDot.classList.add("online");
                statusText.innerText = "Target Online";
                apiStatusDisplay.innerText = "Targeting: " + apiUrl;
                apiStatusDisplay.style.color = "var(--success)";
            } else {
                throw new Error("Bad response");
            }
        } catch (err) {
            statusDot.classList.add("offline");
            statusText.innerText = "Target Offline";
            apiStatusDisplay.innerText = "Server Unreachable";
            apiStatusDisplay.style.color = "var(--danger)";
        }
    }
    // Check on startup
    checkServerRealtime();


    // --- Samples Population ---
    const sampleGrid = document.getElementById("sampleGrid");
    const sampleClasses = ["center", "donut", "edge_loc", "edge_ring", "loc", "near_full", "random", "scratch", "none"];
    sampleClasses.forEach(c => {
        const img = document.createElement("img");
        img.src = `samples/${c}.png`;
        img.className = "sample-img";
        img.title = c.replace("_", " ");
        img.onerror = () => img.style.display = "none";
        img.addEventListener("click", () => {
            // Auto switch to Dashboard tab and load image
            document.querySelector('[data-target="tab-dashboard"]').click();
            loadSampleImage(img.src, c);
        });
        if(sampleGrid) sampleGrid.appendChild(img);
    });

    async function loadSampleImage(src, className) {
        try {
            const response = await fetch(src);
            const blob = await response.blob();
            const file = new File([blob], `${className}.png`, { type: "image/png" });
            handleFile(file);
        } catch (e) {
            console.error("Failed to load sample blob.");
        }
    }

    // --- Core Predictor UI ---
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const selectFileBtn = document.getElementById("selectFileBtn");
    const uiUpload = dropZone.querySelector(".upload-content");
    const uiPreview = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const resetBtn = document.getElementById("resetBtn");
    const runInferenceBtn = document.getElementById("runInferenceBtn");
    
    // Results
    const resultsZone = document.getElementById("resultsZone");
    
    // Batch UI Elements
    const batchProgressText = document.getElementById("batchProgressText");
    const batchProgressFill = document.getElementById("batchProgressFill");
    const batchResultsContainer = document.getElementById("batchResultsContainer");

    let currentFiles = [];

    selectFileBtn.addEventListener("click", (e) => {
        e.preventDefault();
        fileInput.click();
    });

    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", e => { e.preventDefault(); dropZone.classList.remove("dragover"); });
    dropZone.addEventListener("drop", e => {
        e.preventDefault(); dropZone.classList.remove("dragover");
        if(e.dataTransfer.files.length) handleFiles(e.dataTransfer.files);
    });
    fileInput.addEventListener("change", (e) => {
        if(e.target.files.length) handleFiles(e.target.files);
    });

    function handleFiles(files) {
        let maxFiles = Math.min(files.length, 20);
        currentFiles = [];
        const imagePreviewContainer = document.getElementById("imagePreviewContainer");
        imagePreviewContainer.innerHTML = "";
        
        for(let i = 0; i < maxFiles; i++) {
            if(!files[i].type.match("image.*")) continue;
            currentFiles.push(files[i]);
            
            const reader = new FileReader();
            reader.onload = e => {
                const img = document.createElement("img");
                img.src = e.target.result;
                img.setAttribute("data-filename", files[i].name);
                imagePreviewContainer.appendChild(img);
            };
            reader.readAsDataURL(files[i]);
        }
        
        if (currentFiles.length > 0) {
            uiUpload.classList.add("hidden");
            uiPreview.classList.remove("hidden");
            resultsZone.classList.add("hidden");
        }
    }

    resetBtn.addEventListener("click", () => {
        currentFiles = []; fileInput.value = "";
        const imagePreviewContainer = document.getElementById("imagePreviewContainer");
        imagePreviewContainer.innerHTML = "";
        uiPreview.classList.add("hidden");
        uiUpload.classList.remove("hidden");
        resultsZone.classList.add("hidden");
        batchResultsContainer.innerHTML = "";
    });

    // Run Backend Inference Sequential Batch
    runInferenceBtn.addEventListener("click", async () => {
        if(currentFiles.length === 0) return;
        
        resultsZone.classList.remove("hidden");
        batchResultsContainer.innerHTML = "";
        
        const total = currentFiles.length;
        let completed = 0;
        
        // Update Progress
        const updateProgress = () => {
            batchProgressText.innerText = `Processing: ${completed} / ${total}`;
            batchProgressFill.style.width = `${(completed / total) * 100}%`;
        };
        updateProgress();

        // Process sequentially
        for (let i = 0; i < total; i++) {
            const file = currentFiles[i];
            
            // Generate Original Image Data URL
            const origImageUrl = await new Promise(res => {
                const r = new FileReader();
                r.onload = e => res(e.target.result);
                r.readAsDataURL(file);
            });

            try {
                const formData = new FormData();
                formData.append("image", file);
                const response = await fetch(`${apiUrl}/predict`, {
                    method: "POST",
                    body: formData,
                    headers: { "ngrok-skip-browser-warning": "true" }
                });
                
                if(!response.ok) throw new Error(`Status ${response.status}`);
                const data = await response.json();
                if(data.error) throw new Error(data.error);
                
                // Append row
                addBatchResultRow(file.name, origImageUrl, data);

            } catch (err) {
                console.error("Batch error on file " + file.name, err);
            }
            
            completed++;
            updateProgress();
        }
        
    });

    function generateRiskExplanation(className, confidence, riskScore, action) {
        let text = `The model identified a <strong>${className}</strong> defect with <strong>${(confidence*100).toFixed(1)}% confidence</strong>. `;
        
        if (action === "STOP LOT" || action === "STOP") {
            text += `A high risk score of <strong>${riskScore}</strong> was triggered because this defect pattern (e.g. Center, Edge-Ring, Donut) typically arises from systematic equipment failures (like miscalibrated etching tools or gas distribution issues). Halting the lot is required to prevent widespread wafer scrapping.`;
        } else if (action === "INVESTIGATE") {
            text += `A moderate risk score of <strong>${riskScore}</strong> was calculated. This indicates a potential rising issue in the tool line. Although a hard stop isn't mandatory yet, reviewing the sensor logs and monitoring subsequent wafers for clustering is highly recommended.`;
        } else {
            text += `A very low risk score of <strong>${riskScore}</strong> was given. Random or None patterns generally represent isolated anomalies rather than systemic chamber faults, so no immediate process intervention is required. Contextual monitoring is sufficient.`;
        }
        return text;
    }

    // Opens explanation modal
    window.showRiskExplanation = function(className, confidence, riskScore, action) {
        explanationText.innerHTML = generateRiskExplanation(className, confidence, riskScore, action);
        explanationOverlay.classList.add("active");
    };

    function addBatchResultRow(filename, origUrl, data) {
        const confidencePct = (data.confidence * 100).toFixed(1);
        const className = data.class;
        const riskVal = data.risk_score !== undefined ? data.risk_score : 0;
        const action = data.action || "MONITOR";
        const camDataUrl = data.gradcam_png_base64 ? `data:image/png;base64,${data.gradcam_png_base64}` : "";
        
        // CSS coloring logic
        let colorClass = "color-good";
        let scoreClass = "status-card good";
        if(action === "STOP LOT" || action === "STOP") { colorClass = "color-bad"; scoreClass = "status-card bad"; }
        else if(action === "INVESTIGATE") { colorClass = "color-warn"; scoreClass = "status-card warn"; }

        const rowHTML = `
            <div class="batch-row fade-in">
                <div class="batch-imgs">
                    <img src="${origUrl}" class="batch-thumb" alt="Original Input" title="Input: ${filename}" />
                    ${camDataUrl ? `<img src="${camDataUrl}" class="batch-thumb" alt="Grad-CAM" title="Grad-CAM localization" />` : ''}
                </div>
                
                <div class="batch-info">
                    <h3 class="batch-class ${className!=='none'?'color-bad':''}">${className}</h3>
                    <div class="confidence-bar-bg" style="max-width:200px; margin: 4px 0; height:4px;">
                        <div class="confidence-bar-fill" style="width: ${data.confidence * 100}%"></div>
                    </div>
                    <span class="batch-meta">File: ${filename} &nbsp;|&nbsp; Confidence: ${confidencePct}%</span>
                </div>

                <div class="batch-score-section">
                    <div class="batch-score" onclick="showRiskExplanation('${className}', ${data.confidence}, ${riskVal}, '${action}')">
                        ${riskVal}<span style="font-size: 1rem; color: #888">/100</span>
                    </div>
                    <span class="batch-action ${colorClass}">${action}</span>
                </div>
            </div>
        `;
        
        // Insert naturally to the container
        batchResultsContainer.insertAdjacentHTML('beforeend', rowHTML);
    }
});
