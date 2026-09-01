// OmniVid AI Client Application Logic

let currentVideoId = null;
let currentResultData = null;
let pollInterval = null;
let chatHistory = [];

// DOM Elements
const selectionSection = document.getElementById('selectionSection');
const progressSection = document.getElementById('progressSection');
const workspaceSection = document.getElementById('workspaceSection');

const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const videoGrid = document.getElementById('videoGrid');
const refreshVideosBtn = document.getElementById('refreshVideosBtn');

const progressStageTitle = document.getElementById('progressStageTitle');
const progressMessage = document.getElementById('progressMessage');
const progressPctText = document.getElementById('progressPctText');
const progressBarFill = document.getElementById('progressBarFill');

const stepAudio = document.getElementById('stepAudio');
const stepWhisper = document.getElementById('stepWhisper');
const stepFrames = document.getElementById('stepFrames');
const stepGemma = document.getElementById('stepGemma');

const videoPlayer = document.getElementById('videoPlayer');
const currentVideoTitle = document.getElementById('currentVideoTitle');
const metaDuration = document.getElementById('metaDuration');
const metaResolution = document.getElementById('metaResolution');
const metaFps = document.getElementById('metaFps');

const statSegments = document.getElementById('statSegments');
const statKeyframes = document.getElementById('statKeyframes');
const statTime = document.getElementById('statTime');

const gemmaReportContainer = document.getElementById('gemmaReportContainer');
const transcriptList = document.getElementById('transcriptList');
const transcriptSearch = document.getElementById('transcriptSearch');
const framesGrid = document.getElementById('framesGrid');

const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const chatInput = document.getElementById('chatInput');

const changeVideoBtn = document.getElementById('changeVideoBtn');
const copySummaryBtn = document.getElementById('copySummaryBtn');
const exportSrtBtn = document.getElementById('exportSrtBtn');
const exportTxtBtn = document.getElementById('exportTxtBtn');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
  setupTabs();
  setupDropzone();
  setupChat();
  setupVideoPlayerSync();
  loadVideoList();

  refreshVideosBtn.addEventListener('click', loadVideoList);
  changeVideoBtn.addEventListener('click', () => {
    workspaceSection.style.display = 'none';
    progressSection.style.display = 'none';
    selectionSection.style.display = 'block';
    videoPlayer.pause();
    loadVideoList();
  });

  transcriptSearch.addEventListener('input', filterTranscripts);
  copySummaryBtn.addEventListener('click', copySummaryReport);
  exportSrtBtn.addEventListener('click', downloadSrt);
  exportTxtBtn.addEventListener('click', downloadTxt);
});

// Tab Switching
function setupTabs() {
  const tabs = document.querySelectorAll('.tab-btn');
  tabs.forEach(tab => {
    tab.addEventListener('click', () => {
      tabs.forEach(t => t.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tab.classList.add('active');
      const targetPane = document.getElementById(tab.dataset.tab);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

// Dropzone & File Upload
function setupDropzone() {
  dropzone.addEventListener('click', () => fileInput.click());

  ['dragenter', 'dragover'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    dropzone.addEventListener(name, (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
    });
  });

  dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });
}

async function handleFileUpload(file) {
  const formData = new FormData();
  formData.append('file', file);

  showProgress('Uploading Video', `Uploading ${file.name}...`, 5);

  try {
    const res = await fetch('/api/upload', {
      method: 'POST',
      body: formData,
    });
    const data = await res.json();
    if (data.video_id) {
      startProcessingVideo(data.filepath, data.video_id);
    }
  } catch (err) {
    alert(`Upload failed: ${err.message}`);
    hideProgress();
  }
}

// Load Video Library
async function loadVideoList() {
  try {
    const res = await fetch('/api/videos');
    const data = await res.json();
    renderVideoGrid(data.videos || []);
  } catch (err) {
    console.error('Error fetching videos:', err);
    videoGrid.innerHTML = '<p style="color: var(--text-muted)">Could not load videos.</p>';
  }
}

function renderVideoGrid(videos) {
  if (!videos.length) {
    videoGrid.innerHTML = '<p style="color: var(--text-muted)">No videos found in data/videos. Upload one above!</p>';
    return;
  }

  videoGrid.innerHTML = videos.map(v => `
    <div class="video-card" onclick="selectVideo('${v.filepath}', '${v.filename.replace(/\.[^/.]+$/, '')}', ${v.is_processed})">
      <div class="video-card-header">
        <div class="video-card-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
            <polygon points="5 3 19 12 5 21 5 3"></polygon>
          </svg>
        </div>
        <div>
          <div class="video-card-name">${escapeHtml(v.filename)}</div>
          <div class="video-card-meta">
            <span>⏱️ ${v.duration_formatted}</span>
            <span>📐 ${v.width}x${v.height}</span>
            <span>🎙️ ${v.has_audio ? 'Audio track' : 'No audio'}</span>
          </div>
        </div>
      </div>
      <div class="video-card-footer">
        <span class="badge" style="color: ${v.is_processed ? 'var(--accent-emerald)' : 'var(--accent-cyan)'}">
          ${v.is_processed ? '✓ Ready (Processed)' : '⚡ Click to Analyze'}
        </span>
        <button class="btn-primary btn-sm">
          ${v.is_processed ? 'Open' : 'Analyze Now'}
        </button>
      </div>
    </div>
  `).join('');
}

// Start or View Video Processing
async function selectVideo(filepath, videoId, isProcessed) {
  currentVideoId = videoId;
  if (isProcessed) {
    // Already processed, load results directly
    try {
      showProgress('Loading Results', 'Fetching saved analysis...', 90);
      const res = await fetch(`/api/results/${videoId}`);
      const data = await res.json();
      displayWorkspace(data, filepath);
      hideProgress();
    } catch (err) {
      console.warn('Saved result fetch failed, re-processing:', err);
      startProcessingVideo(filepath, videoId);
    }
  } else {
    startProcessingVideo(filepath, videoId);
  }
}

async function startProcessingVideo(filepath, videoId) {
  currentVideoId = videoId;
  showProgress('Starting Analysis Pipeline', 'Initializing Whisper and Gemma models...', 10);

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ video_path: filepath, video_id: videoId }),
    });
    const job = await res.json();

    if (job.job_id) {
      pollJobStatus(job.job_id, filepath, videoId);
    }
  } catch (err) {
    alert(`Error starting pipeline: ${err.message}`);
    hideProgress();
  }
}

function pollJobStatus(jobId, filepath, videoId) {
  if (pollInterval) clearInterval(pollInterval);

  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/api/jobs/${jobId}`);
      const job = await res.json();

      updateProgressUI(job);

      if (job.status === 'completed') {
        clearInterval(pollInterval);
        setTimeout(() => {
          hideProgress();
          displayWorkspace(job.result, filepath);
        }, 600);
      } else if (job.status === 'failed') {
        clearInterval(pollInterval);
        alert(`Processing error: ${job.error || job.message}`);
        hideProgress();
      }
    } catch (err) {
      console.error('Job polling error:', err);
    }
  }, 1000);
}

function updateProgressUI(job) {
  const pct = Math.round(job.progress || 0);
  progressPctText.innerText = `${pct}%`;
  progressBarFill.style.width = `${pct}%`;
  progressMessage.innerText = job.message || 'Processing...';

  // Highlight step badges
  const step = job.step || '';
  stepAudio.className = 'step-badge' + (pct >= 15 ? ' done' : (pct >= 5 ? ' active' : ''));
  stepWhisper.className = 'step-badge' + (pct >= 50 ? ' done' : (pct >= 15 ? ' active' : ''));
  stepFrames.className = 'step-badge' + (pct >= 75 ? ' done' : (pct >= 50 ? ' active' : ''));
  stepGemma.className = 'step-badge' + (pct >= 95 ? ' done' : (pct >= 75 ? ' active' : ''));
}

function showProgress(title, msg, initialPct) {
  selectionSection.style.display = 'none';
  workspaceSection.style.display = 'none';
  progressSection.style.display = 'block';

  progressStageTitle.innerText = title;
  progressMessage.innerText = msg;
  progressPctText.innerText = `${initialPct}%`;
  progressBarFill.style.width = `${initialPct}%`;
}

function hideProgress() {
  progressSection.style.display = 'none';
}

// Display Complete Workspace
function displayWorkspace(data, filepath) {
  currentResultData = data;
  currentVideoId = data.video_id;

  selectionSection.style.display = 'none';
  progressSection.style.display = 'none';
  workspaceSection.style.display = 'grid';

  const meta = data.metadata || {};
  currentVideoTitle.innerText = meta.filename || data.video_id;
  metaDuration.innerText = `⏱️ ${meta.duration_formatted || '--:--'}`;
  metaResolution.innerText = `📐 ${meta.width}x${meta.height}`;
  metaFps.innerText = `⚡ ${meta.fps || 30} fps`;

  statSegments.innerText = data.transcription?.segments?.length || 0;
  statKeyframes.innerText = data.keyframes?.length || 0;
  statTime.innerText = `${data.elapsed_seconds || 0}s`;

  // Set video player src
  videoPlayer.src = `/media/video/${meta.filename || data.video_id + '.mp4'}`;
  videoPlayer.load();

  // 1. Render Gemma 4 "What's Inside" Report
  renderGemmaReport(data.gemma_analysis?.report_markdown || '');

  // 2. Render Audio Transcript
  renderTranscript(data.transcription?.segments || []);

  // 3. Render Visual Frames
  renderFrames(data.keyframes || []);

  // 4. Reset Chat
  chatHistory = [];
  chatMessages.innerHTML = `
    <div class="chat-bubble bot">
      <div class="avatar">G4</div>
      <div class="bubble-content">
        I'm ready! Ask me anything about <strong>${escapeHtml(meta.filename || 'this video')}</strong>, its speech, or its visual contents.
      </div>
    </div>
  `;
}

// Render Gemma Markdown Report
function renderGemmaReport(mdText) {
  if (!mdText) {
    gemmaReportContainer.innerHTML = '<p class="placeholder-text">No summary generated.</p>';
    return;
  }

  // Basic lightweight markdown to HTML converter for sections & formatting
  let html = mdText
    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/^- (.*$)/gim, '<li>$1</li>')
    .replace(/\n\n/g, '<p></p>')
    .replace(/\n/g, '<br>');

  gemmaReportContainer.innerHTML = html;
}

// Render Transcript List
function renderTranscript(segments) {
  if (!segments.length) {
    transcriptList.innerHTML = '<p style="color: var(--text-muted); padding: 1rem;">No speech detected in audio stream.</p>';
    return;
  }

  transcriptList.innerHTML = segments.map(seg => `
    <div class="transcript-row" data-start="${seg.start_sec}" data-end="${seg.end_sec}" onclick="seekTo(${seg.start_sec})">
      <span class="transcript-time">${seg.start_timestamp}</span>
      <span class="transcript-text">${escapeHtml(seg.text)}</span>
    </div>
  `).join('');
}

function filterTranscripts() {
  const query = transcriptSearch.value.toLowerCase();
  const rows = transcriptList.querySelectorAll('.transcript-row');
  rows.forEach(row => {
    const text = row.querySelector('.transcript-text').innerText.toLowerCase();
    row.style.display = text.includes(query) ? 'flex' : 'none';
  });
}

// Render Visual Frames Gallery
function renderFrames(frames) {
  if (!frames.length) {
    framesGrid.innerHTML = '<p style="color: var(--text-muted); padding: 1rem;">No keyframes extracted.</p>';
    return;
  }

  framesGrid.innerHTML = frames.map(f => `
    <div class="frame-item" onclick="seekTo(${f.timestamp})">
      <img class="frame-thumb" src="/media/frames/${currentVideoId}/${f.filename}" alt="Frame at ${f.timestamp_formatted}" loading="lazy">
      <div class="frame-caption">
        <span>⏱️ ${f.timestamp_formatted}</span>
        <span style="color: var(--primary)">Seek ↗</span>
      </div>
    </div>
  `).join('');
}

// Synchronize video player time with transcript
function setupVideoPlayerSync() {
  videoPlayer.addEventListener('timeupdate', () => {
    const curr = videoPlayer.currentTime;
    const rows = transcriptList.querySelectorAll('.transcript-row');

    rows.forEach(row => {
      const start = parseFloat(row.dataset.start);
      const end = parseFloat(row.dataset.end);
      if (curr >= start && curr <= end) {
        row.classList.add('active');
        // Auto scroll if needed
        row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
      } else {
        row.classList.remove('active');
      }
    });
  });
}

function seekTo(seconds) {
  videoPlayer.currentTime = seconds;
  videoPlayer.play();
}

// Chat with Gemma 4
function setupChat() {
  chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = chatInput.value.trim();
    if (!query || !currentVideoId) return;

    chatInput.value = '';

    // Append user message
    appendChatMessage('user', query);
    chatHistory.push({ role: 'user', content: query });

    // Append bot thinking placeholder
    const botMsgEl = appendChatMessage('bot', '<span class="loader" style="width:16px;height:16px;margin-right:8px;vertical-align:middle;"></span> Gemma 4 is thinking...');

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: currentVideoId,
          message: query,
          chat_history: chatHistory,
        }),
      });

      const data = await res.json();
      const answer = data.answer || 'No response from model.';
      botMsgEl.innerHTML = escapeHtml(answer).replace(/\n/g, '<br>');
      chatHistory.push({ role: 'assistant', content: answer });
    } catch (err) {
      botMsgEl.innerHTML = `<span style="color: #ef4444">Error: ${escapeHtml(err.message)}</span>`;
    }
  });
}

function appendChatMessage(role, htmlContent) {
  const bubble = document.createElement('div');
  bubble.className = `chat-bubble ${role}`;
  bubble.innerHTML = `
    <div class="avatar">${role === 'user' ? 'YOU' : 'G4'}</div>
    <div class="bubble-content">${htmlContent}</div>
  `;
  chatMessages.appendChild(bubble);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return bubble.querySelector('.bubble-content');
}

// Helpers
function copySummaryReport() {
  const text = currentResultData?.gemma_analysis?.report_markdown || '';
  navigator.clipboard.writeText(text).then(() => {
    alert('Video report copied to clipboard!');
  });
}

function downloadSrt() {
  const segments = currentResultData?.transcription?.segments || [];
  if (!segments.length) return alert('No transcript available.');

  let srtContent = '';
  segments.forEach((seg, i) => {
    const sTime = formatSrtTime(seg.start_sec);
    const eTime = formatSrtTime(seg.end_sec);
    srtContent += `${i + 1}\n${sTime} --> ${eTime}\n${seg.text}\n\n`;
  });

  triggerDownload(`${currentVideoId}.srt`, srtContent);
}

function downloadTxt() {
  const fullText = currentResultData?.transcription?.full_text || '';
  triggerDownload(`${currentVideoId}_transcript.txt`, fullText);
}

function formatSrtTime(totalSec) {
  const sec = Math.max(0, totalSec);
  const hrs = Math.floor(sec / 3600);
  const mins = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec % 1) * 1000);
  return `${String(hrs).padStart(2, '0')}:${String(mins).padStart(2, '0')}:${String(s).padStart(2, '0')},${String(ms).padStart(3, '0')}`;
}

function triggerDownload(filename, content) {
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function escapeHtml(str) {
  return (str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}
