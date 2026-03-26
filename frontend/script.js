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
    const loadingState = document.getElementById("loadingState");
    const predictedClassLabel = document.getElementById("predictedClass");
    const confidenceFill = document.getElementById("confidenceFill");
    const confidenceText = document.getElementById("confidenceText");
    const statusCard = document.getElementById("statusCard");
    const actionCard = document.getElementById("actionCard");
    const riskScore = document.getElementById("riskScore");
    const actionText = document.getElementById("actionText");
    const resultOriginalImage = document.getElementById("resultOriginalImage");
    const resultCamImage = document.getElementById("resultCamImage");

    let currentFile = null;

    selectFileBtn.addEventListener("click", (e) => {
        e.preventDefault();
        fileInput.click();
    });

    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
    dropZone.addEventListener("dragleave", e => { e.preventDefault(); dropZone.classList.remove("dragover"); });
    dropZone.addEventListener("drop", e => {
        e.preventDefault(); dropZone.classList.remove("dragover");
        if(e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", (e) => {
        if(e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
        if(!file.type.match("image.*")) return;
        currentFile = file;
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            resultOriginalImage.src = e.target.result; 
            uiUpload.classList.add("hidden");
            uiPreview.classList.remove("hidden");
            resultsZone.classList.add("hidden");
            statusCard.className = "card glass-panel status-card"; 
            actionCard.className = "card glass-panel status-card";
            resultCamImage.src = "";
        }
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener("click", () => {
        currentFile = null; fileInput.value = "";
        uiPreview.classList.add("hidden");
        uiUpload.classList.remove("hidden");
        resultsZone.classList.add("hidden");
    });

    // Run Backend Inference
    runInferenceBtn.addEventListener("click", async () => {
        if(!currentFile) return;
        resultsZone.classList.remove("hidden");
        loadingState.classList.remove("hidden");
        
        setTimeout(async () => {
            try {
                const formData = new FormData();
                formData.append("image", currentFile);
                const response = await fetch(`${apiUrl}/predict`, {
                    method: "POST",
                    body: formData,
                    headers: { "ngrok-skip-browser-warning": "true" }
                });
                if(!response.ok) throw new Error(`Status ${response.status}`);
                const data = await response.json();
                if(data.error) throw new Error(data.error);
                updateResultsUI(data);
            } catch (err) {
                alert(`API Error! Are you sure your Ngrok tunnel ${apiUrl} is running?\n\nError: ` + err.message);
                console.error(err);
                checkServerRealtime();
            } finally {
                loadingState.classList.add("hidden");
            }
        }, 300);
    });

    function updateResultsUI(data) {
        const confidencePct = (data.confidence * 100).toFixed(1);
        predictedClassLabel.innerText = data.class;
        confidenceText.innerText = confidencePct;
        confidenceFill.style.width = data.confidence * 100 + "%";

        if(data.class !== "none" && data.class !== "Normal") {
             statusCard.className = "card glass-panel status-card bad";
        } else {
             statusCard.className = "card glass-panel status-card good";
        }

        let riskVal = data.risk_score !== undefined ? data.risk_score : 0;
        let action = data.action || "MONITOR";
        let color = "var(--success)";

        if(action === "STOP LOT" || action === "STOP") {
            color = "var(--danger)";
            actionCard.className = "card glass-panel status-card bad";
        } else if(action === "INVESTIGATE") {
            color = "var(--warning)";
            actionCard.className = "card glass-panel status-card warn";
        } else {
            actionCard.className = "card glass-panel status-card good";
        }

        riskScore.innerText = riskVal + "/100";
        actionText.innerText = action;
        actionText.style.color = color;

        if(data.gradcam_png_base64) {
            resultCamImage.src = `data:image/png;base64,${data.gradcam_png_base64}`;
        }
    }
});
