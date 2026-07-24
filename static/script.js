// ============================================
// State
// ============================================
let sessionId = null;
let isProcessing = false;
let isUploading = false;
let selectedFiles = [];

// ============================================
// DOM Elements
// ============================================
const fileInput = document.getElementById('file-input');
const selectBtn = document.getElementById('select-btn');
const uploadBtn = document.getElementById('upload-btn');
const clearBtn = document.getElementById('clear-btn');
const fileList = document.getElementById('file-list');
const selectedFilesList = document.getElementById('selected-files');
const chatSection = document.getElementById('chat-section');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendBtn = document.getElementById('send-btn');
const sessionInfo = document.getElementById('session-info');
const sessionIdDisplay = document.getElementById('session-id');
const ocrIndicator = document.getElementById('ocr-indicator');
const uploadArea = document.getElementById('upload-area');
const notificationContainer = document.getElementById('notification-container');

// ============================================
// File Selection
// ============================================
selectBtn.addEventListener('click', () => fileInput.click());
uploadArea.addEventListener('click', (e) => {
    // Don't trigger if clicking on buttons inside
    if (e.target.closest('button')) return;
    fileInput.click();
});

fileInput.addEventListener('change', () => {
    const files = fileInput.files;
    if (files.length > 0) {
        addFiles(files);
        fileInput.value = '';
    }
});

// ============================================
// Drag and Drop
// ============================================
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        addFiles(files);
    }
});

// ============================================
// Add Files to List
// ============================================
function addFiles(files) {
    const imageExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'];
    let hasImage = false;

    for (let file of files) {
        const ext = '.' + file.name.split('.').pop().toLowerCase();
        if (imageExtensions.includes(ext)) hasImage = true;
        
        // Check if file already exists
        const exists = selectedFiles.some(f => f.name === file.name && f.size === file.size);
        if (!exists) {
            selectedFiles.push(file);
        }
    }

    renderFileList();
    updateOCRIndicator(hasImage);
    fileList.style.display = 'block';
    showNotification('info', 'Files Added', `${selectedFiles.length} file(s) selected`);
}

// ============================================
// Render File List
// ============================================
function renderFileList() {
    selectedFilesList.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const li = document.createElement('li');
        const fileSize = (file.size / 1024).toFixed(1);
        const icon = getFileIcon(file.name);
        
        li.innerHTML = `
            <span class="file-name">
                <i class="${icon}"></i>
                ${file.name}
            </span>
            <span class="file-size">${fileSize} KB</span>
            <button class="remove-file" data-index="${index}">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        li.querySelector('.remove-file').addEventListener('click', (e) => {
            e.stopPropagation();
            removeFile(index);
        });
        
        selectedFilesList.appendChild(li);
    });
}

function getFileIcon(filename) {
    const ext = filename.split('.').pop().toLowerCase();
    const icons = {
        'pdf': 'fas fa-file-pdf',
        'docx': 'fas fa-file-word',
        'doc': 'fas fa-file-word',
        'txt': 'fas fa-file-alt',
        'png': 'fas fa-file-image',
        'jpg': 'fas fa-file-image',
        'jpeg': 'fas fa-file-image',
        'bmp': 'fas fa-file-image',
        'tiff': 'fas fa-file-image',
        'pptx': 'fas fa-file-powerpoint',
        'csv': 'fas fa-file-csv',
        'xlsx': 'fas fa-file-excel',
        'xls': 'fas fa-file-excel',
        'md': 'fas fa-file-code'
    };
    return icons[ext] || 'fas fa-file';
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    if (selectedFiles.length === 0) {
        fileList.style.display = 'none';
        ocrIndicator.style.display = 'none';
        showNotification('info', 'Cleared', 'All files removed');
    } else {
        renderFileList();
        updateOCRIndicator();
    }
}

function updateOCRIndicator(hasImage) {
    const imageExtensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff'];
    let hasImageFile = hasImage !== undefined ? hasImage : selectedFiles.some(f => 
        imageExtensions.includes('.' + f.name.split('.').pop().toLowerCase())
    );
    ocrIndicator.style.display = hasImageFile ? 'flex' : 'none';
}

// ============================================
// Clear Files
// ============================================
clearBtn.addEventListener('click', () => {
    selectedFiles = [];
    fileList.style.display = 'none';
    ocrIndicator.style.display = 'none';
    showNotification('info', 'Cleared', 'All files removed');
});

// ============================================
// Upload Handler (ChatGPT-style notifications)
// ============================================
uploadBtn.addEventListener('click', handleUpload);

async function handleUpload() {
    if (selectedFiles.length === 0) {
        showNotification('warning', 'No Files', 'Please select files to upload');
        return;
    }

    const formData = new FormData();
    for (let file of selectedFiles) {
        formData.append('files', file);
    }

    showNotification('info', 'Uploading...', `Uploading ${selectedFiles.length} file(s)`);
    uploadBtn.disabled = true;
    clearBtn.disabled = true;
    isUploading = true;

    try {
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            sessionId = data.session_id;
            const ocrMsg = data.ocr_used ? ' (with OCR for images)' : '';
            
            // Show success notification
            showNotification('success', '✅ Upload Complete!', 
                `${data.file_count} file(s) indexed successfully${ocrMsg}`
            );
            
            sessionInfo.innerHTML = `📌 Session: <span>${sessionId}</span>`;
            sessionIdDisplay.textContent = sessionId;
            
            // Show chat section
            chatSection.style.display = 'block';
            chatMessages.innerHTML = '';
            addMessage('assistant', '✅ Documents indexed successfully! Ask me anything about your documents.');
            
            // Enable chat
            chatInput.disabled = false;
            sendBtn.disabled = false;
            chatInput.focus();
            
            // Clear file list
            selectedFiles = [];
            fileList.style.display = 'none';
            ocrIndicator.style.display = 'none';
            
        } else {
            showNotification('error', '❌ Upload Failed', data.detail || 'Unknown error');
        }
    } catch (error) {
        showNotification('error', '❌ Upload Failed', error.message);
    } finally {
        uploadBtn.disabled = false;
        clearBtn.disabled = false;
        isUploading = false;
    }
}

// ============================================
// Notification System (ChatGPT-style)
// ============================================
function showNotification(type, title, message) {
    const icons = {
        success: 'fa-check-circle',
        error: 'fa-exclamation-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };
    
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    notification.innerHTML = `
        <div class="notification-icon">
            <i class="fas ${icons[type] || icons.info}"></i>
        </div>
        <div class="notification-content">
            <div class="notification-title">${title}</div>
            <div class="notification-message">${message}</div>
        </div>
        <button class="notification-close"><i class="fas fa-times"></i></button>
    `;
    
    notification.querySelector('.notification-close').addEventListener('click', () => {
        removeNotification(notification);
    });
    
    notificationContainer.appendChild(notification);
    
    // Auto remove after 6 seconds
    setTimeout(() => {
        removeNotification(notification);
    }, 6000);
}

function removeNotification(notification) {
    notification.classList.add('fade-out');
    setTimeout(() => {
        notification.remove();
    }, 300);
}

// ============================================
// Chat Handler (same as before)
// ============================================
sendBtn.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') sendMessage();
});

async function sendMessage() {
    const message = chatInput.value.trim();
    if (!message || !sessionId || isProcessing) return;

    addMessage('user', message);
    chatInput.value = '';
    chatInput.disabled = true;
    sendBtn.disabled = true;
    isProcessing = true;

    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: sessionId, message: message })
        });

        const data = await response.json();

        if (response.ok) {
            addMessage('assistant', data.answer);
        } else {
            addMessage('assistant', `❌ Error: ${data.detail}`);
            showNotification('error', 'Chat Error', data.detail);
        }
    } catch (error) {
        addMessage('assistant', `❌ Error: ${error.message}`);
        showNotification('error', 'Chat Error', error.message);
    } finally {
        chatInput.disabled = false;
        sendBtn.disabled = false;
        isProcessing = false;
        chatInput.focus();
    }
}

function addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user' 
        ? '<i class="fas fa-user"></i>' 
        : '<i class="fas fa-robot"></i>';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    contentDiv.innerHTML = `<p>${content}</p>`;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// ============================================
// Console Welcome
// ============================================
console.log('🚀 MultiDocChat loaded!');
console.log('📸 OCR support: Enabled');
console.log('💡 Select files, then click Upload');
console.log('🔍 RAG with FAISS + MMR');