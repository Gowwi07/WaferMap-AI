document.addEventListener("DOMContentLoaded", () => {
    // UI Elements
    const dropZone = document.getElementById("dropZone");
    const fileInput = document.getElementById("fileInput");
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
    const riskScore = document.getElementById("riskScore");
    const actionText = document.getElementById("actionText");
    const resultOriginalImage = document.getElementById("resultOriginalImage");
    const resultCamImage = document.getElementById("resultCamImage");

    let currentFile = null;

    // --- Configuration Logic ---
    let apiUrl = localStorage.getItem("waferApiUrl") || "http://localhost:8000";
    apiUrlInput.value = apiUrl;
    updateStatusLabel();

    configToggle.addEventListener("click", () => {
        configPanel.classList.toggle("active");
    });

    saveApiBtn.addEventListener("click", () => {
        apiUrl = apiUrlInput.value.trim().replace(/\/$/, ""); // Remove trailing slash
        localStorage.setItem("waferApiUrl", apiUrl);
        updateStatusLabel();
        configPanel.classList.remove("active");
    });

    function updateStatusLabel() {
        if(apiUrl.includes("ngrok") || apiUrl.includes("localhost") || apiUrl.includes("cloudflare")) {
            apiStatusDisplay.innerText = "Targeting: " + apiUrl;
            apiStatusDisplay.style.color = "var(--success)";
        } else {
            apiStatusDisplay.innerText = "Please provide valid URL";
        }
    }

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
            resultOriginalImage.src = e.target.result; // Update original image in results
            
            uiUpload.classList.add("hidden");
            uiPreview.classList.remove("hidden");
            
            // reset UI state
            resultsZone.classList.add("hidden");
            statusCard.className = "card glass-panel status-card"; 
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
        
        // Let UI update
        setTimeout(async () => {
            try {
                const formData = new FormData();
                formData.append("image", currentFile);

                const response = await fetch(`${apiUrl}/predict`, {
                    method: "POST",
                    body: formData,
                    // If hitting an ngrok endpoint, you often need this header to bypass browser block screen for free tier
                    headers: {
                        "ngrok-skip-browser-warning": "69420" 
                    }
                });

                if(!response.ok) {
                    throw new Error(`API Error: ${response.status}`);
                }

                const data = await response.json();
                if(data.error) {
                    throw new Error(data.error);
                }

                updateResultsUI(data);

            } catch (err) {
                alert("Failed to reach API! Did you setup the tunnel? Error details: " + err.message);
                console.error(err);
            } finally {
                loadingState.classList.add("hidden");
            }
        }, 300);
    });

    function updateResultsUI(data) {
        // Class & Confidence
        const confidencePct = (data.confidence * 100).toFixed(1);
        predictedClassLabel.innerText = data.class;
        confidenceText.innerText = confidencePct;
        confidenceFill.style.width = data.confidence * 100 + "%";

        if(data.class !== "none" && data.class !== "Normal") {
             statusCard.classList.add("bad");
        } else {
             statusCard.classList.add("good");
        }

        // Mock yield risk logic (since the API doesn't expose the risk score, we generate a mock or you can add risk score to the backend)
        // Wait, calculate_risk_score is in src/risk_score.py. Since the frontend receives class & confidence, let's do a simple mapping here
        // If the API had returned yield risk, we would use it!
        let riskVal = 10;
        let action = "MONITOR";
        let color = "var(--success)";

        if(data.class === "Donut" || data.class === "Center") {
            riskVal = 85; 
            action = "STOP LOT";
            color = "var(--danger)";
        } else if(data.class !== "none" && data.class !== "Normal") {
            riskVal = 55;
            action = "INVESTIGATE";
            color = "var(--warning)";
        }

        riskScore.innerText = riskVal + "/100";
        actionText.innerText = action;
        actionText.style.color = color;

        // Image Update
        if(data.gradcam_png_base64) {
            resultCamImage.src = `data:image/png;base64,${data.gradcam_png_base64}`;
        }
    }
});
