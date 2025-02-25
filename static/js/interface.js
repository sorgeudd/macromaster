// Initialize variables first
let ws = null;
let isRecordingMacro = false;
let isRecordingSound = false;
let isLearning = false;
let monitoring = false;

// Initialize WebSocket connection
function initializeWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    // Use port 5003 where our test server is running successfully
    const wsUrl = `${protocol}//${window.location.hostname}:5003/ws`;

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connection established");
            updateConnectionStatus(true);
            refreshMacroList();
            refreshSoundList();
        };

        ws.onclose = (event) => {
            console.log("WebSocket connection closed", event);
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
            case 'recording_complete':
                if (data.recording_type === 'macro') {
                    handleMacroRecordingComplete();
                } else if (data.recording_type === 'sound') {
                    handleSoundRecordingComplete();
                }
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

// Bot Controls
function startBot() {
    sendWebSocketMessage('start_bot');
    updateStatus('bot', 'Running', true);
}

function stopBot() {
    sendWebSocketMessage('stop_bot');
    updateStatus('bot', 'Stopped');
}

function emergencyStop() {
    sendWebSocketMessage('emergency_stop');
    updateStatus('bot', 'Stopped');
}

// Position Controls
function moveTo() {
    const x = document.getElementById('pos-x').value;
    const y = document.getElementById('pos-y').value;
    sendWebSocketMessage('move_to', { x: parseInt(x), y: parseInt(y) });
}

// Resource Controls
function loadMap(type) {
    sendWebSocketMessage('load_map', { map_type: type });
}

function findNearby() {
    sendWebSocketMessage('find_nearby');
}

// Macro Recording Functions
function startMacroRecording() {
    const macroName = document.getElementById('macro-name').value.trim();
    if (!macroName) {
        addLog('Please enter a macro name', 'error');
        return;
    }

    isRecordingMacro = true;
    updateMacroStatus(true);
    addLog('Recording macro in 3 seconds...', 'info');

    setTimeout(() => {
        sendWebSocketMessage('start_macro_recording', { macro_name: macroName });
        addLog('Started recording macro...', 'info');
    }, 3000);
}

function stopMacroRecording() {
    if (!isRecordingMacro) return;

    const macroName = document.getElementById('macro-name').value.trim();
    sendWebSocketMessage('stop_macro_recording', { macro_name: macroName });
    addLog('Stopping macro recording...', 'info');
}

function updateMacroStatus(recording) {
    const recordButton = document.getElementById("record-macro-btn");
    const stopButton = document.getElementById("stop-macro-btn");
    const saveButton = document.getElementById("save-macro-btn");

    if (recordButton && stopButton) {
        recordButton.disabled = recording;
        stopButton.disabled = !recording;
        if (saveButton) saveButton.disabled = recording;
        isRecordingMacro = recording;
    }
}

function handleMacroRecordingComplete() {
    updateMacroStatus(false);
    addLog('Macro recording completed', 'info');
}

function saveMacroRecording() {
    const macroName = document.getElementById('macro-name').value.trim();
    sendWebSocketMessage('save_macro', { macro_name: macroName });

    const saveBtn = document.getElementById('save-macro-btn');
    if (saveBtn) saveBtn.disabled = true;
    document.getElementById('macro-name').value = '';
    addLog('Saving macro...', 'info');
}

function playMacro() {
    const macroName = document.getElementById('macro-list').value;
    if (!macroName) {
        addLog('Please select a macro', 'error');
        return;
    }
    sendWebSocketMessage('play_macro', { macro_name: macroName });
}

// Learning Functions
function startLearning() {
    if (!isLearning) {
        isLearning = true;
        sendWebSocketMessage("start_learning");
        updateStatus('learning', 'Active', true);
    }
}

function stopLearning() {
    if (isLearning) {
        isLearning = false;
        sendWebSocketMessage("stop_learning");
        updateStatus('learning', 'Inactive');
    }
}

function resetLearning() {
    sendWebSocketMessage("reset_learning");
}

// Sound Recording Functions
function startSoundRecording() {
    const soundName = document.getElementById('sound-name').value.trim();
    if (!soundName) {
        addLog('Please enter a sound trigger name', 'error');
        return;
    }

    isRecordingSound = true;
    updateSoundStatus(true);
    addLog('Recording sound in 3 seconds...', 'info');

    setTimeout(() => {
        sendWebSocketMessage('start_sound_recording', { sound_name: soundName });
        addLog('Recording sound...', 'info');
    }, 3000);
}

function stopSoundRecording() {
    if (!isRecordingSound) return;

    const soundName = document.getElementById('sound-name').value.trim();
    sendWebSocketMessage('stop_sound_recording', { sound_name: soundName });
    addLog('Stopping sound recording...', 'info');
}

function updateSoundStatus(recording) {
    const recordButton = document.getElementById("record-sound-btn");
    const stopButton = document.getElementById("stop-sound-btn");
    const saveButton = document.getElementById("save-sound-btn");

    if (recordButton && stopButton) {
        recordButton.disabled = recording;
        stopButton.disabled = !recording;
        if (saveButton) saveButton.disabled = recording;
        isRecordingSound = recording;
    }
}

function handleSoundRecordingComplete() {
    updateSoundStatus(false);
    addLog('Sound recording completed', 'info');
}

function saveSoundRecording() {
    const soundName = document.getElementById('sound-name').value.trim();
    sendWebSocketMessage('save_sound_recording', { sound_name: soundName });

    const saveBtn = document.getElementById('save-sound-btn');
    if (saveBtn) saveBtn.disabled = true;
    document.getElementById('sound-name').value = '';
    addLog('Saving sound...', 'info');
    refreshSoundList();
}

function playSound() {
    const soundName = document.getElementById('sound-list').value;
    if (!soundName) {
        addLog('Please select a sound', 'error');
        return;
    }
    sendWebSocketMessage('play_sound', { sound_name: soundName });
}

function toggleSoundMonitoring() {
    monitoring = !monitoring;
    const btn = document.getElementById("monitor-btn");

    if (monitoring) {
        sendWebSocketMessage('start_sound_monitoring');
        btn.textContent = "Stop Monitoring";
        updateStatus('monitoring', 'Active', true);
        addLog('Started sound monitoring', 'info');
    } else {
        sendWebSocketMessage('stop_sound_monitoring');
        btn.textContent = "Start Monitoring";
        updateStatus('monitoring', 'Stopped');
        addLog('Stopped sound monitoring', 'info');
    }
}

function mapSoundToMacro() {
    const soundName = document.getElementById('map-sound').value;
    const macroName = document.getElementById('map-macro').value;
    if (!soundName || !macroName) {
        addLog('Please select both a sound and a macro', 'error');
        return;
    }
    sendWebSocketMessage('map_sound_to_macro', {
        sound_name: soundName,
        macro_name: macroName
    });
}

function updateStatus(type, status, isActive = false) {
    const indicator = document.getElementById(`${type}-status-indicator`);
    const text = document.getElementById(`${type}-status`);
    if (indicator && text) {
        indicator.className = `status-indicator ${isActive ? 'active' : status === 'error' ? 'error' : 'ready'}`;
        text.textContent = status;
    }
}

function refreshMacroList() {
    fetch('/macros')
        .then(response => response.json())
        .then(data => {
            if (data.macros) {
                updateMacroList(data.macros);
                addLog('Macro list refreshed', 'info');
            }
        })
        .catch(error => addLog('Failed to refresh macros: ' + error, 'error'));
}

function refreshSoundList() {
    fetch('/sounds')
        .then(response => response.json())
        .then(data => {
            if (data.sounds) {
                updateSoundList(data.sounds);
                addLog('Sound list refreshed', 'info');
            }
        })
        .catch(error => addLog('Failed to refresh sounds: ' + error, 'error'));
}

function updateMacroList(macros) {
    const macroList = document.getElementById('macro-list');
    const mapMacro = document.getElementById('map-macro');
    [macroList, mapMacro].forEach(select => {
        if (!select) return;
        select.innerHTML = '<option value="">Select a macro</option>';
        macros.forEach(macro => {
            const option = document.createElement('option');
            option.value = macro;
            option.textContent = macro;
            select.appendChild(option);
        });
    });
}

function updateSoundList(sounds) {
    const soundList = document.getElementById('sound-list');
    const mapSound = document.getElementById('map-sound');
    [soundList, mapSound].forEach(select => {
        if (!select) return;
        select.innerHTML = '<option value="">Select a sound</option>';
        sounds.forEach(sound => {
            const option = document.createElement('option');
            option.value = sound;
            option.textContent = sound;
            select.appendChild(option);
        });
    });
}

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

    if (startBotBtn) startBotBtn.onclick = startBot;
    if (stopBotBtn) stopBotBtn.onclick = stopBot;
    if (emergencyStopBtn) emergencyStopBtn.onclick = emergencyStop;

    // Learning controls
    const startLearningBtn = document.getElementById("start-learning");
    const stopLearningBtn = document.getElementById("stop-learning");
    const resetLearningBtn = document.getElementById("reset-learning");

    if (startLearningBtn) startLearningBtn.onclick = startLearning;
    if (stopLearningBtn) stopLearningBtn.onclick = stopLearning;
    if (resetLearningBtn) resetLearningBtn.onclick = resetLearning;

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
    const mapSoundMacroBtn = document.getElementById("map-sound-macro-btn");

    if (recordSoundBtn) recordSoundBtn.onclick = startSoundRecording;
    if (stopSoundBtn) stopSoundBtn.onclick = stopSoundRecording;
    if (saveSoundBtn) saveSoundBtn.onclick = saveSoundRecording;
    if (playSoundBtn) playSoundBtn.onclick = playSound;
    if (monitoringBtn) monitoringBtn.onclick = toggleSoundMonitoring;
    if (mapSoundMacroBtn) mapSoundMacroBtn.onclick = mapSoundToMacro;

    // Position and Resource controls
    const moveToBtn = document.getElementById("move-to-btn");
    const loadFishMapBtn = document.getElementById("load-fish-map");
    const loadOreMapBtn = document.getElementById("load-ore-map");
    const findNearbyBtn = document.getElementById("find-nearby");

    if (moveToBtn) moveToBtn.onclick = moveTo;
    if (loadFishMapBtn) loadFishMapBtn.onclick = () => loadMap('fish');
    if (loadOreMapBtn) loadOreMapBtn.onclick = () => loadMap('ore');
    if (findNearbyBtn) findNearbyBtn.onclick = findNearby;
}