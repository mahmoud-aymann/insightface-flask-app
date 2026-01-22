// DOM Elements
const uploadBox = document.getElementById('uploadBox');
const imageInput = document.getElementById('imageInput');
const previewSection = document.getElementById('previewSection');
const previewImage = document.getElementById('previewImage');
const removeBtn = document.getElementById('removeBtn');
const recognizeBtn = document.getElementById('recognizeBtn');
const thresholdSlider = document.getElementById('threshold');
const thresholdValue = document.getElementById('thresholdValue');
const resultSection = document.getElementById('resultSection');
const loadingOverlay = document.getElementById('loadingOverlay');

// Stats elements
const totalPersons = document.getElementById('totalPersons');
const totalEmbeddings = document.getElementById('totalEmbeddings');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    
    // Upload box click
    uploadBox.addEventListener('click', () => {
        imageInput.click();
    });
    
    // File input change
    imageInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.classList.add('dragover');
    });
    
    uploadBox.addEventListener('dragleave', () => {
        uploadBox.classList.remove('dragover');
    });
    
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFile(files[0]);
        }
    });
    
    // Remove button
    removeBtn.addEventListener('click', removeImage);
    
    // Recognize button
    recognizeBtn.addEventListener('click', recognizeFace);
    
    // Threshold slider
    thresholdSlider.addEventListener('input', (e) => {
        thresholdValue.textContent = parseFloat(e.target.value).toFixed(2);
    });
});

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file');
        return;
    }
    
    // Validate file size (16MB)
    if (file.size > 16 * 1024 * 1024) {
        alert('File size must be less than 16MB');
        return;
    }
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewSection.style.display = 'block';
        uploadBox.style.display = 'none';
        recognizeBtn.disabled = false;
    };
    reader.readAsDataURL(file);
}

function removeImage() {
    imageInput.value = '';
    previewSection.style.display = 'none';
    uploadBox.style.display = 'block';
    recognizeBtn.disabled = true;
    resultSection.style.display = 'none';
}

async function recognizeFace() {
    if (!imageInput.files[0]) {
        alert('Please select an image first');
        return;
    }
    
    // Show loading
    loadingOverlay.style.display = 'flex';
    recognizeBtn.disabled = true;
    
    try {
        // Create form data
        const formData = new FormData();
        formData.append('image', imageInput.files[0]);
        formData.append('threshold', thresholdSlider.value);
        
        // Send request
        const response = await fetch('/recognize', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Recognition failed');
        }
        
        // Display result
        displayResult(data);
        
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Recognition error:', error);
    } finally {
        loadingOverlay.style.display = 'none';
        recognizeBtn.disabled = false;
    }
}

function displayResult(data) {
    const personNameEl = document.getElementById('personName');
    const confidenceEl = document.getElementById('confidence');
    const statusEl = document.getElementById('status');
    
    personNameEl.textContent = data.person_name;
    confidenceEl.textContent = (data.confidence * 100).toFixed(2) + '%';
    
    if (data.status === 'recognized') {
        statusEl.textContent = 'Recognized';
        statusEl.className = 'value recognized';
        personNameEl.className = 'value recognized';
    } else {
        statusEl.textContent = 'Unknown';
        statusEl.className = 'value unknown';
        personNameEl.className = 'value unknown';
    }
    
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        if (response.ok) {
            totalPersons.textContent = data.total_persons || 0;
            totalEmbeddings.textContent = data.total_embeddings || 0;
        }
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}
