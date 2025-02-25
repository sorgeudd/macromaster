// Initialize variables first
let ws = null;
let isRecordingMacro = false;
let isRecordingSound = false;
let isLearning = false;
let monitoring = false;

// Initialize WebSocket connection
function initializeWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connection established");
            updateConnectionStatus(true);
            refreshMacroList();
            refreshSoundList();
        };

        ws.onclose = () => {
            console.log("WebSocket connection closed");
            updateConnectionStatus(false);
            // Try to reconnect after 5 seconds
            setTimeout(initializeWebSocket, 5000);
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            updateConnectionStatus(false);
            addLog("WebSocket connection error", "error");
        };

        ws.onmessage = handleWebSocketMessage;
    } catch (error) {
        console.error("Error initializing WebSocket:", error);
        updateConnectionStatus(false);
        addLog("Failed to initialize WebSocket connection", "error");
    }
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById("system-status");
    const indicator = document.getElementById("system-status-indicator");
    if (statusElement && indicator) {
        statusElement.textContent = connected ? "Connected" : "Disconnected";
        indicator.className = `status-indicator ${connected ? 'active' : 'error'}`;
    }
}

function handleWebSocketMessage(event) {
    try {
        const data = JSON.parse(event.data);
        console.log("Received message:", data);

        switch (data.type) {
            case 'status_update':
                updateStatus('window', data.window_status, data.window_status === 'Detected');
                updateStatus('bot', data.bot_status, data.bot_status === 'Running');
                updateStatus('learning', data.learning_status, data.learning_status === 'Active');
                updateStatus('macro', data.macro_status, data.macro_status === 'Recording');
                updateStatus('sound', data.sound_status, data.sound_status === 'Recording');
                updateStatus('monitoring', data.monitoring_status, data.monitoring_status === 'Active');
                break;
            case 'log':
                addLog(data.message, data.level);
                break;
            case 'error':
                addLog(data.error, 'error');
                break;
            case 'macros_updated':
                updateMacroList(data.macros);
                break;
            case 'sounds_updated':
                updateSoundList(data.sounds);
                break;
        }
    } catch (error) {
        console.error("Error handling WebSocket message:", error);
        addLog("Error processing server message", "error");
    }
}

function sendWebSocketMessage(type, data = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addLog("WebSocket connection not available", "error");
        return false;
    }

    try {
        const message = { type, ...data };
        ws.send(JSON.stringify(message));
        return true;
    } catch (error) {
        console.error("Error sending WebSocket message:", error);
        addLog("Failed to send message to server", "error");
        return false;
    }
}

function addLog(message, level = 'info') {
    const logs = document.getElementById('logs');
    if (!logs) return;

    const entry = document.createElement('div');
    entry.className = `log-entry log-${level}`;
    entry.textContent = `${new Date().toLocaleTimeString()}: ${message}`;
    logs.insertBefore(entry, logs.firstChild);

    // Keep only the last 100 log entries
    while (logs.children.length > 100) {
        logs.removeChild(logs.lastChild);
    }
}

// Macro Recording Functions
function startMacroRecording() {
    if (!isRecordingMacro) {
        isRecordingMacro = true;
        sendWebSocketMessage("start_macro");
        updateMacroStatus(true);
    }
}

function stopMacroRecording() {
    if (isRecordingMacro) {
        isRecordingMacro = false;
        sendWebSocketMessage("stop_macro");
        updateMacroStatus(false);
    }
}

function updateMacroStatus(recording) {
    const recordButton = document.getElementById("record-macro-btn");
    const stopButton = document.getElementById("stop-macro-btn");
    if (recordButton && stopButton) {
        recordButton.disabled = recording;
        stopButton.disabled = !recording;
        isRecordingMacro = recording;
    }
}

function saveMacroRecording(){
    sendWebSocketMessage("save_macro");
}

function playMacro(){
    sendWebSocketMessage("play_macro");
}

// Learning Functions
function startLearning() {
    if (!isLearning) {
        isLearning = true;
        sendWebSocketMessage("start_learning");
        updateLearningStatus(true);
    }
}

function stopLearning() {
    if (isLearning) {
        isLearning = false;
        sendWebSocketMessage("stop_learning");
        updateLearningStatus(false);
    }
}

function updateLearningStatus(learning) {
    const startButton = document.getElementById("start-learning");
    const stopButton = document.getElementById("stop-learning");
    if (startButton && stopButton) {
        startButton.disabled = learning;
        stopButton.disabled = !learning;
        isLearning = learning;
    }
}

function resetLearning() {
    sendWebSocketMessage("reset_learning");
}

// Sound Recording Functions
function startSoundRecording() {
    if (!isRecordingSound) {
        isRecordingSound = true;
        sendWebSocketMessage("start_sound");
        updateSoundStatus(true);
    }
}

function stopSoundRecording() {
    if (isRecordingSound) {
        isRecordingSound = false;
        sendWebSocketMessage("stop_sound");
        updateSoundStatus(false);
    }
}

function updateSoundStatus(recording) {
    const recordButton = document.getElementById("record-sound-btn");
    const stopButton = document.getElementById("stop-sound-btn");
    if (recordButton && stopButton) {
        recordButton.disabled = recording;
        stopButton.disabled = !recording;
        isRecordingSound = recording;
    }
}

function saveSoundRecording(){
    sendWebSocketMessage("save_sound");
}

function playSound(){
    sendWebSocketMessage("play_sound");
}

function toggleSoundMonitoring() {
    monitoring = !monitoring;
    sendWebSocketMessage("toggle_monitoring", { monitoring });
    updateMonitoringStatus(monitoring);
}

function updateMonitoringStatus(monitoring) {
    const monitoringBtn = document.getElementById("monitor-btn");
    if (monitoringBtn) {
        monitoringBtn.textContent = monitoring ? "Stop Monitoring" : "Start Monitoring";
    }
}



function updateStatus(type, status, isActive = false) {
    const indicator = document.getElementById(`${type}-status-indicator`);
    const text = document.getElementById(`${type}-status`);
    if (indicator && text) {
        indicator.className = `status-indicator ${isActive ? 'active' : status === 'error' ? 'error' : 'ready'}`;
        text.textContent = status;
    }
}


// Placeholder functions -  Replace with actual implementations
function refreshMacroList() {}
function refreshSoundList() {}
function updateMacroList(macros) {}
function updateSoundList(sounds) {}



// Initialize everything when the document is ready
document.addEventListener("DOMContentLoaded", () => {
    // Initialize WebSocket first
    initializeWebSocket();

    // Set initial status indicators
    updateStatus('system', 'Initializing');
    updateStatus('window', 'Not Detected');
    updateStatus('bot', 'Stopped');
    updateStatus('learning', 'Inactive');
    updateStatus('macro', 'Ready');
    updateStatus('sound', 'Ready');
    updateStatus('monitoring', 'Stopped');

    // Initialize UI elements
    setupEventListeners();
});

function setupEventListeners() {
    // Bot controls
    const startBotBtn = document.getElementById("start-bot");
    const stopBotBtn = document.getElementById("stop-bot");
    const emergencyStopBtn = document.getElementById("emergency-stop");

    if (startBotBtn) startBotBtn.onclick = () => sendWebSocketMessage('start_bot');
    if (stopBotBtn) stopBotBtn.onclick = () => sendWebSocketMessage('stop_bot');
    if (emergencyStopBtn) emergencyStopBtn.onclick = () => sendWebSocketMessage('emergency_stop');

    // Learning controls
    const startLearningBtn = document.getElementById("start-learning");
    const stopLearningBtn = document.getElementById("stop-learning");
    const resetLearningBtn = document.getElementById("reset-learning");

    if (startLearningBtn) startLearningBtn.onclick = () => {
        isLearning = true;
        sendWebSocketMessage('start_learning');
        updateStatus('learning', 'Active', true);
    };

    if (stopLearningBtn) stopLearningBtn.onclick = () => {
        isLearning = false;
        sendWebSocketMessage('stop_learning');
        updateStatus('learning', 'Inactive');
    };

    if (resetLearningBtn) resetLearningBtn.onclick = () => sendWebSocketMessage('reset_learning');

    // Macro controls
    const recordMacroBtn = document.getElementById("record-macro-btn");
    const stopMacroBtn = document.getElementById("stop-macro-btn");
    const saveMacroBtn = document.getElementById("save-macro-btn");
    const playMacroBtn = document.getElementById("play-macro-btn");

    if (recordMacroBtn) recordMacroBtn.onclick = startMacroRecording;
    if (stopMacroBtn) stopMacroBtn.onclick = stopMacroRecording;
    if (saveMacroBtn) saveMacroBtn.onclick = saveMacroRecording;
    if (playMacroBtn) playMacroBtn.onclick = playMacro;

    // Sound controls
    const recordSoundBtn = document.getElementById("record-sound-btn");
    const stopSoundBtn = document.getElementById("stop-sound-btn");
    const saveSoundBtn = document.getElementById("save-sound-btn");
    const playSoundBtn = document.getElementById("play-sound-btn");
    const monitoringBtn = document.getElementById("monitor-btn");

    if (recordSoundBtn) recordSoundBtn.onclick = startSoundRecording;
    if (stopSoundBtn) stopSoundBtn.onclick = stopSoundRecording;
    if (saveSoundBtn) saveSoundBtn.onclick = saveSoundRecording;
    if (playSoundBtn) playSoundBtn.onclick = playSound;
    if (monitoringBtn) monitoringBtn.onclick = toggleSoundMonitoring;
}