let model;
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const startBtn = document.getElementById('startCam');
const stopBtn = document.getElementById('stopCam');
const imageInput = document.getElementById('imageInput');
const totalsBox = document.getElementById('totals');
const thresholdEl = document.getElementById('threshold');
const downloadCSVBtn = document.getElementById('downloadCSV');

let stream = null;
let running = false;
let lastCounts = {};

(async function init() {
  model = await cocoSsd.load({ base: 'lite_mobilenet_v2' }); // fast + light
  totalsBox.textContent = 'Ready. Start camera or upload an image.';
})();

function fitCanvasTo(el) {
  const rect = el.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = rect.height;
}

function drawDetections(dets, scaleX=1, scaleY=1) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2;
  ctx.font = '12px ui-monospace, monospace';
  dets.forEach(d => {
    const [x, y, w, h] = d.bbox;
    ctx.strokeStyle = 'white';
    ctx.fillStyle = 'white';
    ctx.globalAlpha = 0.9;
    ctx.strokeRect(x*scaleX, y*scaleY, w*scaleX, h*scaleY);
    const tag = `${d.class} ${(d.score*100).toFixed(0)}%`;
    const tx = x*scaleX, ty = Math.max(12, y*scaleY - 4);
    ctx.fillRect(tx, ty - 12, ctx.measureText(tag).width + 6, 14);
    ctx.fillStyle = 'black';
    ctx.fillText(tag, tx + 3, ty);
  });
  ctx.globalAlpha = 1;
}

function countByClass(dets, thresh) {
  const counts = {};
  for (const d of dets) {
    if (d.score >= thresh) counts[d.class] = (counts[d.class] || 0) + 1;
  }
  counts.__total = Object.values(counts).reduce((a,b)=>a+b,0);
  return counts;
}

function prettyCounts(counts) {
  if (!counts || Object.keys(counts).length === 0) return 'No objects.';
  const { __total, ...byClass } = counts;
  const lines = [`Total: ${__total}`];
  Object.entries(byClass).sort((a,b)=>b[1]-a[1]).forEach(([k,v])=>lines.push(`${k}: ${v}`));
  return lines.join('\n');
}

async function predictFromVideo() {
  if (!running) return;
  fitCanvasTo(video);
  const dets = await model.detect(video);
  const counts = countByClass(dets, parseFloat(thresholdEl.value || '0.5'));
  lastCounts = counts;
  totalsBox.textContent = prettyCounts(counts);
  drawDetections(dets);
  requestAnimationFrame(predictFromVideo);
}

startBtn.addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' }, audio: false });
    video.srcObject = stream;
    await video.play();
    startBtn.disabled = true;
    stopBtn.disabled = false;
    downloadCSVBtn.disabled = false;
    running = true;
    predictFromVideo();
  } catch (e) {
    totalsBox.textContent = 'Camera error: ' + e.message;
  }
});

stopBtn.addEventListener('click', () => {
  running = false;
  if (stream) stream.getTracks().forEach(t => t.stop());
  startBtn.disabled = false;
  stopBtn.disabled = true;
});

imageInput.addEventListener('change', async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const img = new Image();
  img.onload = async () => {
    // Draw image on video-sized canvas area
    stopBtn.click(); // stop camera if running
    fitCanvasTo(canvas); // ensure canvas has layout size
    canvas.width = img.width; canvas.height = img.height;
    ctx.drawImage(img, 0, 0);
    const dets = await model.detect(img);
    const counts = countByClass(dets, parseFloat(thresholdEl.value || '0.5'));
    lastCounts = counts;
    totalsBox.textContent = prettyCounts(counts);
    // Rescale drawing to canvas size (currently 1:1 with img)
    drawDetections(dets, 1, 1);
    downloadCSVBtn.disabled = false;
  };
  img.src = URL.createObjectURL(file);
});

thresholdEl.addEventListener('input', () => {
  totalsBox.textContent = prettyCounts(lastCounts);
});

downloadCSVBtn.addEventListener('click', () => {
  const rows = [['class','count']];
  Object.entries(lastCounts).forEach(([k,v])=>{
    if (k !== '__total') rows.push([k, v]);
  });
  rows.unshift(['total', lastCounts.__total || 0]);
  const csv = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'object_counts.csv';
  a.click();
});
