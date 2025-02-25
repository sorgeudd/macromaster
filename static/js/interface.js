// Initialize variables first
let ws = null;
let isRecordingMacro = false;
let isRecordingSound = false;
let monitoring = false;
let reconnectAttempts = 0;
let maxReconnectAttempts = 5;
let reconnectTimeout = null;
let pingInterval = null;
let lastPongTime = Date.now();
let isRecordingHotkey = false;
let currentHotkeyRecorder = null;

// Initialize WebSocket connection
function initializeWebSocket() {
    if (ws && ws.readyState === WebSocket.CONNECTING) {
        return; // Already trying to connect
    }

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    try {
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            console.log("WebSocket connection established");
            reconnectAttempts = 0; // Reset reconnect attempts on successful connection

            // Start ping interval
            if (pingInterval) {
                clearInterval(pingInterval);
            }
            pingInterval = setInterval(sendPing, 30000); // Send ping every 30 seconds
            lastPongTime = Date.now();

            updateStatus('system', 'Connected', true);
        };

        ws.onclose = (event) => {
            console.log("WebSocket connection closed", event);
            updateStatus('system', 'Disconnected', false);
            clearInterval(pingInterval);

            // Clean up the existing WebSocket
            if (ws) {
                ws.onopen = null;
                ws.onclose = null;
                ws.onerror = null;
                ws.onmessage = null;
                ws = null;
            }

            // Try to reconnect with exponential backoff
            if (reconnectAttempts < maxReconnectAttempts) {
                const backoffTime = Math.min(1000 * Math.pow(2, reconnectAttempts), 10000);
                console.log(`Attempting to reconnect in ${backoffTime}ms...`);

                if (reconnectTimeout) {
                    clearTimeout(reconnectTimeout);
                }

                reconnectTimeout = setTimeout(() => {
                    reconnectAttempts++;
                    initializeWebSocket();
                }, backoffTime);
            } else {
                addLog("WebSocket connection failed. Please refresh the page.", "error");
            }
        };

        ws.onerror = (error) => {
            console.error("WebSocket error:", error);
            updateStatus('system', 'Error', false);
        };

        ws.onmessage = handleWebSocketMessage;
    } catch (error) {
        console.error("Error initializing WebSocket:", error);
        updateStatus('system', 'Error', false);
        addLog("Failed to initialize WebSocket connection", "error");
    }
}

function sendPing() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        const timeSinceLastPong = Date.now() - lastPongTime;
        if (timeSinceLastPong > 60000) { // No pong for 60 seconds
            console.log("No pong received for 60 seconds, reconnecting...");
            if (ws) {
                ws.close();
            }
            return;
        }
        try {
            sendWebSocketMessage('ping');
        } catch (error) {
            console.error("Error sending ping:", error);
        }
    }
}

function handleWebSocketMessage(event) {
    try {
        const data = JSON.parse(event.data);
        console.log("Received message:", data);

        switch (data.type) {
            case 'pong':
                lastPongTime = Date.now();
                break;
            case 'status_update':
                updateStatus('macro', data.macro_status, data.macro_status === 'Recording');
                updateStatus('sound', data.sound_status, data.sound_status === 'Recording');
                break;
            case 'log':
                addLog(data.message, data.level);
                break;
            case 'error':
                addLog(data.message, 'error');
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
            case 'hotkey_updated':
                updateHotkeyDisplay(data.macro_name, data.hotkey);
                break;
            case 'screenshot_taken':
                addLog(`Screenshot saved: ${data.filename}`, 'info');
                break;
        }
    } catch (error) {
        console.error("Error handling WebSocket message:", error);
        addLog("Error processing server message", "error");
    }
}

function sendWebSocketMessage(type, data = {}) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
        addLog("Connection lost. Attempting to reconnect...", "error");
        initializeWebSocket();
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

// Hotkey Management Functions
function startHotkeyRecording() {
    const macroName = document.getElementById('macro-name').value.trim();
    if (!macroName) {
        addLog('Please enter a macro name first', 'error');
        return;
    }

    isRecordingHotkey = true;
    const hotkeyInput = document.getElementById('hotkey-input');
    const assignButton = document.getElementById('assign-hotkey-btn');

    hotkeyInput.value = '';
    hotkeyInput.placeholder = 'Press keys...';
    hotkeyInput.classList.add('recording');
    assignButton.textContent = 'Recording...';
    assignButton.classList.add('recording');

    // Store the keys being pressed
    let pressedKeys = new Set();

    function handleKeyDown(e) {
        e.preventDefault();
        const key = e.key.toLowerCase();
        if (!pressedKeys.has(key)) {
            pressedKeys.add(key);
            updateHotkeyDisplay();
        }
    }

    function handleKeyUp(e) {
        e.preventDefault();
        const key = e.key.toLowerCase();
        pressedKeys.delete(key);

        // If no keys are pressed, stop recording
        if (pressedKeys.size === 0) {
            stopHotkeyRecording();
        } else {
            updateHotkeyDisplay();
        }
    }

    function updateHotkeyDisplay() {
        const hotkey = Array.from(pressedKeys).join('+');
        hotkeyInput.value = hotkey;
    }

    function stopHotkeyRecording() {
        document.removeEventListener('keydown', handleKeyDown);
        document.removeEventListener('keyup', handleKeyUp);
        isRecordingHotkey = false;
        hotkeyInput.placeholder = 'Enter hotkey (e.g. ctrl+shift+a)';
        hotkeyInput.classList.remove('recording');
        assignButton.textContent = 'Assign Hotkey';
        assignButton.classList.remove('recording');

        // If we have a valid hotkey combination, assign it
        if (hotkeyInput.value) {
            assignHotkey();
        }
    }

    document.addEventListener('keydown', handleKeyDown);
    document.addEventListener('keyup', handleKeyUp);
    currentHotkeyRecorder = { stop: stopHotkeyRecording };
}

function assignHotkey() {
    const macroName = document.getElementById('macro-name').value.trim();
    const hotkey = document.getElementById('hotkey-input').value.trim();

    if (!macroName) {
        addLog('Please enter a macro name', 'error');
        return;
    }
    if (!hotkey) {
        addLog('Please enter a hotkey', 'error');
        return;
    }

    // Validate the hotkey format
    const hotkeyParts = hotkey.toLowerCase().split('+');
    const validModifiers = ['ctrl', 'alt', 'shift'];
    const hasValidModifier = hotkeyParts.some(part => validModifiers.includes(part));

    if (!hasValidModifier) {
        addLog('Hotkey must include at least one modifier key (Ctrl, Alt, or Shift)', 'error');
        return;
    }

    sendWebSocketMessage('assign_hotkey', {
        macro_name: macroName,
        hotkey: hotkey
    });
}

function updateHotkeyDisplay(macroName, hotkey) {
    const hotkeyDisplay = document.getElementById('hotkey-display');
    if (hotkeyDisplay) {
        hotkeyDisplay.textContent = `Current Hotkey: ${hotkey || 'None'}`;
    }

    const hotkeyInput = document.getElementById('hotkey-input');
    if (hotkeyInput) {
        hotkeyInput.value = hotkey || '';
    }
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

    if (recordButton && stopButton) {
        recordButton.disabled = recording;
        stopButton.disabled = !recording;
        isRecordingMacro = recording;
    }
}

function handleMacroRecordingComplete() {
    updateMacroStatus(false);
    addLog('Macro recording completed', 'info');
}

function playMacro() {
    const macroName = document.getElementById('macro-name').value.trim();
    if (!macroName) {
        addLog('Please enter a macro name', 'error');
        return;
    }
    sendWebSocketMessage('play_macro', { macro_name: macroName });
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

    if (recordButton && stopButton) {
        recordButton.disabled = recording;
        stopButton.disabled = !recording;
        isRecordingSound = recording;
    }
}

function handleSoundRecordingComplete() {
    updateSoundStatus(false);
    addLog('Sound recording completed', 'info');
}

function playSound() {
    const soundName = document.getElementById('sound-name').value.trim();
    if (!soundName) {
        addLog('Please enter a sound name', 'error');
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
        updateStatus('sound', 'Monitoring', true);
        addLog('Started sound monitoring', 'info');
    } else {
        sendWebSocketMessage('stop_sound_monitoring');
        btn.textContent = "Start Monitoring";
        updateStatus('sound', 'Ready');
        addLog('Stopped sound monitoring', 'info');
    }
}

function updateStatus(type, status, isActive = false) {
    const indicator = document.getElementById(`${type}-status-indicator`);
    const text = document.getElementById(`${type}-status`);
    if (indicator && text) {
        indicator.className = `status-indicator ${isActive ? 'active' : status.toLowerCase() === 'error' ? 'error' : 'ready'}`;
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
    if (!macroList) return;
    macroList.innerHTML = '<option value="">Select a macro</option>';
    macros.forEach(macro => {
        const option = document.createElement('option');
        option.value = macro;
        option.textContent = macro;
        macroList.appendChild(option);
    });
}

function updateSoundList(sounds) {
    const soundList = document.getElementById('sound-list');
    if (!soundList) return;
    soundList.innerHTML = '<option value="">Select a sound</option>';
    sounds.forEach(sound => {
        const option = document.createElement('option');
        option.value = sound;
        option.textContent = sound;
        soundList.appendChild(option);
    });
}

// Initialize everything when the document is ready
document.addEventListener("DOMContentLoaded", () => {
    // Initialize WebSocket first
    initializeWebSocket();

    // Set initial status indicators
    updateStatus('system', 'Initializing');
    updateStatus('macro', 'Ready');
    updateStatus('sound', 'Ready');
});

// Add these functions to handle screenshots

function takeScreenshot() {
    const macroName = document.getElementById('macro-name').value.trim();
    if (!macroName) {
        addLog('Please enter a macro name before taking a screenshot', 'error');
        return;
    }

    // Show flash effect
    const flash = document.getElementById('screenshot-flash');
    flash.style.animation = 'none';
    flash.offsetHeight; // Trigger reflow
    flash.style.animation = null;

    sendWebSocketMessage('take_screenshot', { macro_name: macroName });
}

function clearHotkey() {
    const macroName = document.getElementById('macro-name').value.trim();
    if (!macroName) {
        addLog('Please enter a macro name', 'error');
        return;
    }

    sendWebSocketMessage('clear_hotkey', { macro_name: macroName });
}

function setupEventListeners() {
    // Macro controls
    const recordMacroBtn = document.getElementById("record-macro-btn");
    const stopMacroBtn = document.getElementById("stop-macro-btn");
    const playMacroBtn = document.getElementById("play-macro-btn");
    const assignHotkeyBtn = document.getElementById("assign-hotkey-btn");
    const clearHotkeyBtn = document.getElementById("clear-hotkey-btn");

    if (recordMacroBtn) recordMacroBtn.onclick = startMacroRecording;
    if (stopMacroBtn) stopMacroBtn.onclick = stopMacroRecording;
    if (playMacroBtn) playMacroBtn.onclick = playMacro;
    if (assignHotkeyBtn) {
        assignHotkeyBtn.onclick = function() {
            if (isRecordingHotkey) {
                if (currentHotkeyRecorder) {
                    currentHotkeyRecorder.stop();
                }
            } else {
                startHotkeyRecording();
            }
        };
    }
    if (clearHotkeyBtn) clearHotkeyBtn.onclick = clearHotkey;

    // Sound controls
    const recordSoundBtn = document.getElementById("record-sound-btn");
    const stopSoundBtn = document.getElementById("stop-sound-btn");
    const playSoundBtn = document.getElementById("play-sound-btn");
    const monitoringBtn = document.getElementById("monitor-btn");

    if (recordSoundBtn) recordSoundBtn.onclick = startSoundRecording;
    if (stopSoundBtn) stopSoundBtn.onclick = stopSoundRecording;
    if (playSoundBtn) playSoundBtn.onclick = playSound;
    if (monitoringBtn) monitoringBtn.onclick = toggleSoundMonitoring;

    const screenshotBtn = document.getElementById("screenshot-btn");
    if (screenshotBtn) screenshotBtn.onclick = takeScreenshot;
}

setupEventListeners();