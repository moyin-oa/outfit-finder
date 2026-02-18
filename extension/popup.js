const captureBtn = document.getElementById("captureBtn");
const openSettingsBtn = document.getElementById("openSettings");
const previewCanvas = document.getElementById("previewCanvas");
const canvasEmpty = document.getElementById("canvasEmpty");
const ctx = previewCanvas.getContext("2d");

function setStatus(msg) {
  console.debug("[Outfit Finder]", msg);
}

function dataURLtoImage(dataUrl) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });
}

function drawPreview(image) {
  ctx.clearRect(0, 0, previewCanvas.width, previewCanvas.height);
  if (!image) return;
  const cw = previewCanvas.width;
  const ch = previewCanvas.height;
  const r = Math.min(cw / image.width, ch / image.height);
  const w = image.width * r;
  const h = image.height * r;
  const x = (cw - w) / 2;
  const y = (ch - h) / 2;
  ctx.fillStyle = "#f2f2f4";
  ctx.fillRect(0, 0, cw, ch);
  ctx.drawImage(image, x, y, w, h);
}

async function loadLastCaptureFromStorage() {
  return new Promise((resolve) => {
    chrome.storage.local.get(
      ["latestCropDataUrl", "latestCaptureStatus", "latestCaptureError"],
      async (res) => {
        const status = res.latestCaptureError
          ? `Capture failed: ${res.latestCaptureError}`
          : res.latestCaptureStatus || "";
        if (status) setStatus(status);

        if (res.latestCropDataUrl) {
          try {
            const img = await dataURLtoImage(res.latestCropDataUrl);
            drawPreview(img);
            canvasEmpty.classList.add("hidden");
          } catch (_err) {
            // ignore stale data
          }
        }
        resolve();
      }
    );
  });
}

async function requestSelectionCapture() {
  setStatus("Select region on page. The extension will auto-open web app results.");
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type: "startSelectionCapture" }, (resp) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      if (!resp?.ok) {
        reject(new Error(resp?.error || "Capture failed"));
        return;
      }
      resolve();
    });
  });
}

captureBtn.addEventListener("click", async () => {
  try {
    await requestSelectionCapture();
  } catch (err) {
    setStatus(`Capture failed: ${err.message || err}`);
  }
});

openSettingsBtn.addEventListener("click", () => {
  chrome.runtime.openOptionsPage();
});

setStatus("Ready");
loadLastCaptureFromStorage();
