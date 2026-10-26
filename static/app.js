const canvas = document.querySelector('#drawing');
const ctx = canvas.getContext('2d');
const preview = document.querySelector('#preview').getContext('2d');
const predictButton = document.querySelector('#predict');
const status = document.querySelector('#status');
const pixels = new Array(784).fill(0);
let drawing = false, lastPoint = null, revision = 0, requestId = 0;
const rows = Array.from({length: 10}, (_, digit) => {
  const row = document.createElement('div');
  row.className = 'score-row';
  row.innerHTML = `<span>${digit}</span><div class="track"><div class="fill"></div></div><span class="value">—</span>`;
  document.querySelector('#scores').append(row);
  return row;
});
function paint(context, values) {
  context.fillStyle = '#101b16'; context.fillRect(0, 0, 28, 28);
  values.forEach((value, i) => {
    if (value > 0) { context.fillStyle = `rgb(${Math.round(value*255)},${Math.round(value*255)},${Math.round(value*255)})`; context.fillRect(i%28, Math.floor(i/28), 1, 1); }
  });
}
function resetResult() {
  document.querySelector('#digit').textContent = '—';
  rows.forEach(row => { row.classList.remove('winner'); row.querySelector('.fill').style.width='0%'; row.querySelector('.value').textContent='—'; });
  paint(preview, []);
}
function point(event) {
  const rect = canvas.getBoundingClientRect();
  return {x: Math.max(0, Math.min(27, Math.floor((event.clientX-rect.left)/rect.width*28))), y: Math.max(0, Math.min(27, Math.floor((event.clientY-rect.top)/rect.height*28)))};
}
function dab(x,y) {
  for(let dy=-1;dy<=1;dy++) for(let dx=-1;dx<=1;dx++) {
    if(dx*dx+dy*dy<=1 && x+dx>=0 && x+dx<28 && y+dy>=0 && y+dy<28) pixels[(y+dy)*28+x+dx]=1;
  }
}
function stroke(event) {
  const next = point(event), start = lastPoint || next;
  const steps = Math.max(Math.abs(next.x-start.x), Math.abs(next.y-start.y), 1);
  for(let i=0;i<=steps;i++) dab(Math.round(start.x+(next.x-start.x)*i/steps), Math.round(start.y+(next.y-start.y)*i/steps));
  lastPoint=next; revision++; paint(ctx,pixels); resetResult(); predictButton.disabled=false;
  status.textContent='';
}
canvas.addEventListener('pointerdown', event => {
  if (event.button !== 0 || drawing) return;
  drawing=true; lastPoint=null; canvas.setPointerCapture(event.pointerId); stroke(event);
});
canvas.addEventListener('pointermove', event => { if(drawing && canvas.hasPointerCapture(event.pointerId)) stroke(event); });
function stop(){ drawing=false; lastPoint=null; }
canvas.addEventListener('pointerup',stop); canvas.addEventListener('pointercancel',stop); canvas.addEventListener('lostpointercapture',stop);
document.querySelector('#clear').addEventListener('click', () => {
  pixels.fill(0); revision++; requestId++; stop(); paint(ctx,pixels); resetResult(); predictButton.disabled=true; status.textContent='';
});
predictButton.addEventListener('click', async () => {
  const currentRevision=revision, currentRequest=++requestId;
  predictButton.disabled=true; status.textContent='Predicting…';
  try {
    const response=await fetch('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({pixels})});
    const result=await response.json();
    if(currentRequest!==requestId || currentRevision!==revision) return;
    if(!response.ok) throw new Error(typeof result.detail==='string' ? result.detail : 'Unable to read this drawing. Please try again.');
    document.querySelector('#digit').textContent=result.digit;
    rows.forEach((row,i)=> {row.classList.toggle('winner',i===result.digit); row.querySelector('.fill').style.width=`${result.scores[i]*100}%`; row.querySelector('.value').textContent=result.scores[i].toFixed(3);});
    paint(preview,result.processed_pixels); status.textContent='';
  } catch(error) {
    if(currentRequest===requestId && currentRevision===revision) status.textContent=error.message || 'Connection failed. Please try again.';
  } finally {
    if(currentRequest===requestId) predictButton.disabled=!pixels.some(Boolean);
  }
});
paint(ctx,pixels); paint(preview,[]);
