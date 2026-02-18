const DEFAULT_API_BASE = "http://127.0.0.1:8000";
const DEFAULT_WEB_APP_BASE = "http://127.0.0.1:5500/index.html";

function pageSelectionFunction() {
  return new Promise((resolve) => {
    const existing = document.getElementById("__outfit_finder_overlay__");
    if (existing) existing.remove();

    const overlay = document.createElement("div");
    overlay.id = "__outfit_finder_overlay__";
    overlay.style.position = "fixed";
    overlay.style.inset = "0";
    overlay.style.zIndex = "2147483647";
    overlay.style.cursor = "crosshair";
    overlay.style.background = "rgba(0,0,0,0.18)";
    overlay.style.userSelect = "none";

    const box = document.createElement("div");
    box.style.position = "absolute";
    box.style.border = "2px solid #0a66ff";
    box.style.background = "rgba(10,102,255,0.16)";
    box.style.display = "none";
    overlay.appendChild(box);

    const tip = document.createElement("div");
    tip.textContent = "Drag to select area. Press Esc to cancel.";
    tip.style.position = "absolute";
    tip.style.top = "12px";
    tip.style.left = "50%";
    tip.style.transform = "translateX(-50%)";
    tip.style.padding = "8px 10px";
    tip.style.borderRadius = "8px";
    tip.style.font = "13px -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif";
    tip.style.color = "white";
    tip.style.background = "rgba(0,0,0,0.7)";
    overlay.appendChild(tip);

    let startX = 0;
    let startY = 0;
    let currX = 0;
    let currY = 0;
    let dragging = false;

    function cleanup(result) {
      window.removeEventListener("keydown", onKeyDown, true);
      overlay.removeEventListener("mousedown", onMouseDown, true);
      overlay.removeEventListener("mousemove", onMouseMove, true);
      overlay.removeEventListener("mouseup", onMouseUp, true);
      overlay.remove();
      resolve(result);
    }

    function onKeyDown(ev) {
      if (ev.key === "Escape") {
        ev.preventDefault();
        ev.stopPropagation();
        cleanup(null);
      }
    }

    function onMouseDown(ev) {
      ev.preventDefault();
      ev.stopPropagation();
      dragging = true;
      startX = ev.clientX;
      startY = ev.clientY;
      currX = ev.clientX;
      currY = ev.clientY;
      box.style.display = "block";
      box.style.left = `${startX}px`;
      box.style.top = `${startY}px`;
      box.style.width = "0px";
      box.style.height = "0px";
    }

    function onMouseMove(ev) {
      if (!dragging) return;
      ev.preventDefault();
      ev.stopPropagation();
      currX = ev.clientX;
      currY = ev.clientY;
      const x = Math.min(startX, currX);
      const y = Math.min(startY, currY);
      const w = Math.abs(currX - startX);
      const h = Math.abs(currY - startY);
      box.style.left = `${x}px`;
      box.style.top = `${y}px`;
      box.style.width = `${w}px`;
      box.style.height = `${h}px`;
    }

    function onMouseUp(ev) {
      if (!dragging) return;
      ev.preventDefault();
      ev.stopPropagation();
      dragging = false;
      currX = ev.clientX;
      currY = ev.clientY;
      const x = Math.min(startX, currX);
      const y = Math.min(startY, currY);
      const w = Math.abs(currX - startX);
      const h = Math.abs(currY - startY);
      if (w < 8 || h < 8) {
        cleanup(null);
        return;
      }
      cleanup({
        x,
        y,
        w,
        h,
        viewportWidth: window.innerWidth,
        viewportHeight: window.innerHeight,
      });
    }

    window.addEventListener("keydown", onKeyDown, true);
    overlay.addEventListener("mousedown", onMouseDown, true);
    overlay.addEventListener("mousemove", onMouseMove, true);
    overlay.addEventListener("mouseup", onMouseUp, true);
    document.documentElement.appendChild(overlay);
  });
}

function abToB64(ab) {
  let binary = "";
  const bytes = new Uint8Array(ab);
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    const part = bytes.subarray(i, i + chunk);
    binary += String.fromCharCode(...part);
  }
  return btoa(binary);
}

async function blobToDataUrl(blob) {
  const ab = await blob.arrayBuffer();
  const b64 = abToB64(ab);
  return `data:${blob.type || "image/jpeg"};base64,${b64}`;
}

async function getSettings() {
  return new Promise((resolve) => {
    chrome.storage.sync.get(["apiBase", "webAppBase"], (res) => {
      resolve({
        apiBase: (res.apiBase || DEFAULT_API_BASE).replace(/\/$/, ""),
        webAppBase: res.webAppBase || DEFAULT_WEB_APP_BASE,
      });
    });
  });
}

async function uploadCaptureAndOpenWebApp(croppedBlob, fullBlob) {
  const { apiBase, webAppBase } = await getSettings();
  const formData = new FormData();
  formData.append("file", croppedBlob, "selection.jpg");
  if (fullBlob) {
    formData.append("full_file", fullBlob, "full_screenshot.jpg");
  }

  const uploadRes = await fetch(`${apiBase}/upload-capture`, {
    method: "POST",
    body: formData,
  });
  const payload = await uploadRes.json().catch(() => ({}));
  if (!uploadRes.ok) {
    const msg = payload.detail || payload.error || "Upload failed";
    throw new Error(msg);
  }
  const captureId = payload.capture_id;
  if (!captureId) {
    throw new Error("Backend did not return capture_id");
  }

  const webUrl = new URL(webAppBase);
  webUrl.searchParams.set("capture_id", captureId);
  await chrome.tabs.create({ url: webUrl.toString() });
}

async function runSelectionCaptureWorkflow() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (!tab || !tab.id) {
    throw new Error("No active tab found");
  }

  const injected = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: pageSelectionFunction,
  });
  const selected = injected?.[0]?.result || null;
  if (!selected) {
    await chrome.storage.local.set({
      latestCaptureStatus: "Selection canceled",
      latestCaptureError: "",
    });
    return;
  }

  const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "jpeg", quality: 90 });
  const response = await fetch(dataUrl);
  const fullBlob = await response.blob();
  const bitmap = await createImageBitmap(fullBlob);

  const scaleX = bitmap.width / selected.viewportWidth;
  const scaleY = bitmap.height / selected.viewportHeight;
  const sx = Math.max(0, Math.floor(selected.x * scaleX));
  const sy = Math.max(0, Math.floor(selected.y * scaleY));
  const sw = Math.max(1, Math.floor(selected.w * scaleX));
  const sh = Math.max(1, Math.floor(selected.h * scaleY));

  const off = new OffscreenCanvas(sw, sh);
  const ctx = off.getContext("2d");
  ctx.drawImage(bitmap, sx, sy, sw, sh, 0, 0, sw, sh);
  const croppedBlob = await off.convertToBlob({ type: "image/jpeg", quality: 0.92 });
  const croppedDataUrl = await blobToDataUrl(croppedBlob);

  await chrome.storage.local.set({
    latestCropDataUrl: croppedDataUrl,
    latestCropAt: Date.now(),
    latestCaptureStatus: "Selection captured. Opening web app...",
    latestCaptureError: "",
  });

  await uploadCaptureAndOpenWebApp(croppedBlob, fullBlob);
  await chrome.storage.local.set({
    latestCaptureStatus: "Opened web app with capture results.",
    latestCaptureError: "",
  });
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.type !== "startSelectionCapture") return;
  (async () => {
    try {
      await chrome.storage.local.set({
        latestCaptureStatus: "Select a region on the page...",
        latestCaptureError: "",
      });
      await runSelectionCaptureWorkflow();
      sendResponse({ ok: true });
    } catch (err) {
      const message = err?.message || String(err);
      await chrome.storage.local.set({
        latestCaptureStatus: "Capture failed",
        latestCaptureError: message,
      });
      sendResponse({ ok: false, error: message });
    }
  })();
  return true;
});
