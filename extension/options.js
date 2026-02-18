const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const DEFAULT_WEB_APP_BASE = "http://127.0.0.1:5500/index.html";

const apiBaseInput = document.getElementById("apiBase");
const webAppBaseInput = document.getElementById("webAppBase");
const saveBtn = document.getElementById("saveBtn");
const statusEl = document.getElementById("status");

function setStatus(msg) {
  statusEl.textContent = msg;
}

function load() {
  chrome.storage.sync.get(["apiBase", "webAppBase"], (res) => {
    apiBaseInput.value = res.apiBase || DEFAULT_API_BASE;
    webAppBaseInput.value = res.webAppBase || DEFAULT_WEB_APP_BASE;
    setStatus("Loaded");
  });
}

function save() {
  const value = (apiBaseInput.value || "").trim() || DEFAULT_API_BASE;
  const webAppValue = (webAppBaseInput.value || "").trim() || DEFAULT_WEB_APP_BASE;
  chrome.storage.sync.set({ apiBase: value, webAppBase: webAppValue }, () => {
    setStatus("Saved");
  });
}

saveBtn.addEventListener("click", save);
load();
