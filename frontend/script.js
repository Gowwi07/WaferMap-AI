document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
    const selectFileBtn = document.getElementById("selectFileBtn"); // new explicit btn
    
    const configToggle = document.getElementById("configToggle");
    const configPanel = document.getElementById("configPanel");
    const saveApiBtn = document.getElementById("saveApiBtn");
    const apiUrlInput = document.getElementById("apiUrlInput");
    const apiStatusDisplay = document.getElementById("apiStatusDisplay");
    
    const uiUpload = dropZone.querySelector(".upload-content");
    const uiPreview = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const resetBtn = document.getElementById("resetBtn");
    const runInferenceBtn = document.getElementById("runInferenceBtn");
    
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
    const sampleGrid = document.getElementById("sampleGrid");

    let currentFile = null;

    // --- Configuration Logic ---
    let apiUrl = localStorage.getItem("waferApiUrl") || "http://localhost:8000";
    apiUrlInput.value = apiUrl;
    updateStatusLabel();

    // Toggle Config panel
    configToggle.addEventListener("click", () => {
        configPanel.classList.toggle("active");
    });

    saveApiBtn.addEventListener("click", () => {
        apiUrl = apiUrlInput.value.trim().replace(/\/$/, ""); 
        localStorage.setItem("waferApiUrl", apiUrl);
        updateStatusLabel();
        configPanel.classList.remove("active");
    });

    function updateStatusLabel() {
        if(apiUrl.includes("ngrok") || apiUrl.includes("localhost") || apiUrl.includes("cloudflare")) {
            apiStatusDisplay.innerText = "Targeting: " + Math.min(apiUrl.length, 30) > 30 ? apiUrl.substring(0,30) + '...' : apiUrl;
            apiStatusDisplay.style.color = "var(--success)";
        } else {
            apiStatusDisplay.innerText = "Please provide valid URL";
        }
    }

    // --- Populating Sample Images ---
    const sampleClasses = ["center", "donut", "edge_loc", "edge_ring", "loc", "near_full", "random", "scratch", "none"];
    sampleClasses.forEach(c => {
        const img = document.createElement("img");
        img.src = `samples/${c}.png`;
        img.className = "sample-img";
        img.title = c.replace("_", " ");
        img.onerror = () => img.style.display = "none"; // Hide if extraction didn't find one
        img.addEventListener("click", () => loadSampleImage(img.src, c));
        sampleGrid.appendChild(img);
    });

    async function loadSampleImage(src, className) {
        try {
            const response = await fetch(src);
            const blob = await response.blob();
            const file = new File([blob], `${className}.png`, { type: "image/png" });
            handleFile(file);
        } catch (e) {
            console.error("Failed to fetch sample image.");
        }
    }

    // --- File Input Triggers ---
    selectFileBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        fileInput.click();
    });

    dropZone.addEventListener("click", (e) => {
        // Optional: Let clicking anywhere in upload-content trigger it
        if (!uiUpload.classList.contains("hidden") && e.target !== selectFileBtn && e.target.nodeName !== "INPUT") {
             fileInput.click();
        }
    });

    // --- File Upload Logic ---
    dropZone.addEventListener("dragover", e => {
        e.preventDefault();
        dropZone.classList.add("dragover");
    });

    dropZone.addEventListener("dragleave", e => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
    });

    dropZone.addEventListener("drop", e => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if(e.dataTransfer.files.length) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if(e.target.files.length) {
            handleFile(e.target.files[0]);
        }
    });

    function handleFile(file) {
        if(!file.type.match("image.*")) {
            alert("Please upload image file (jpg, png).");
            return;
        }

        currentFile = file;
        
        // Show local preview
        const reader = new FileReader();
        reader.onload = e => {
            imagePreview.src = e.target.result;
            resultOriginalImage.src = e.target.result; 
            
            uiUpload.classList.add("hidden");
            uiPreview.classList.remove("hidden");
            
            // reset UI state
            resultsZone.classList.add("hidden");
            statusCard.className = "card glass-panel status-card"; 
            actionCard.className = "card glass-panel status-card";
            resultCamImage.src = "";
        }
        reader.readAsDataURL(file);
    }

    resetBtn.addEventListener("click", () => {
        currentFile = null;
        fileInput.value = "";
        uiPreview.classList.add("hidden");
        uiUpload.classList.remove("hidden");
        resultsZone.classList.add("hidden");
    });

    // --- API Request Logic ---
    runInferenceBtn.addEventListener("click", async () => {
        if(!currentFile) return;

        resultsZone.classList.remove("hidden");
        loadingState.classList.remove("hidden");
        
        // Let UI flush
        setTimeout(async () => {
            try {
                const formData = new FormData();
                formData.append("image", currentFile);

                const response = await fetch(`${apiUrl}/predict`, {
                    method: "POST",
                    body: formData,
                    headers: { "ngrok-skip-browser-warning": "true" }
                });

                if(!response.ok) {
                    throw new Error(`API Error: ${response.status}`);
                }

                const data = await response.json();
                if(data.error) throw new Error(data.error);

                updateResultsUI(data);
            } catch (err) {
                alert("Failed to reach API! Ensure Ngrok tunnel is running and URL doesn't have a trailing slash. Log: " + err.message);
                console.error(err);
            } finally {
                loadingState.classList.add("hidden");
            }
        }, 100);
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
