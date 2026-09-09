const message = document.querySelector('#message');
const count = document.querySelector('#char-count');
const run = document.querySelector('#run');
const resultPanel = document.querySelector('#result-panel');
const examples = document.querySelector('#example-list');
message.addEventListener('input', () => count.textContent = `${message.value.length} characters`);
count.textContent = `${message.value.length} characters`;

function esc(value) { return String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
function renderResult(data) {
  const pct = Math.round(data.confidence * 100);
  const evidence = data.evidence.length ? data.evidence.map(item => `<div class="evidence-item"><span class="evidence-meta">${item.score} match</span>${esc(item.text)}</div>`).join('') : '<div class="evidence-item">No close historical precedent found.</div>';
  const mode = data.mode === 'llm' ? 'LLM' : 'KEYWORD';
  resultPanel.innerHTML = `<div class="result-head"><div><p class="eyebrow accent">ANALYSIS COMPLETE · ${mode} MODE</p><h2>Recommended action</h2></div><span class="decision ${data.decision === 'auto-handle' ? 'auto' : 'escalate'}">${data.decision.toUpperCase()}</span></div><div class="intent-row"><div><div class="eyebrow">DETECTED INTENT</div><div class="intent-name">${esc(data.intent.replaceAll('_',' '))}</div></div><div class="confidence">${pct}% confidence</div></div><div class="bar"><i style="width:${pct}%"></i></div><p class="reason">${esc(data.reason)}</p><div class="draft"><strong>DRAFT REPLY</strong>${esc(data.reply)}</div><div class="evidence"><h3>HISTORICAL EVIDENCE · ${data.evidence.length} retrieved replies</h3>${evidence}</div>`;
}
run.addEventListener('click', async () => {
  run.disabled = true; run.textContent = 'Analyzing…';
  try { const res = await fetch('/api/agent', {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:message.value})}); renderResult(await res.json()); }
  finally { run.disabled = false; run.innerHTML = 'Analyze message <span>→</span>'; }
});
fetch('/api/examples').then(r=>r.json()).then(items => items.forEach(item => { const b=document.createElement('button'); b.className='example'; b.textContent=item.text.slice(0,70)+(item.text.length>70?'…':''); b.onclick=()=>{message.value=item.text;message.dispatchEvent(new Event('input'));}; examples.appendChild(b); }));
fetch('/api/evaluation').then(r=>r.json()).then(data => {
  const grid = document.querySelector('#audit-grid');
  const intentAcc = data.intent_accuracy != null ? `${Math.round(data.intent_accuracy*100)}%` : '—';
  const escAcc = data.escalation_accuracy != null ? `${Math.round(data.escalation_accuracy*100)}%` : '—';
  const source = data.source || 'proxy audit';
  grid.innerHTML = `<div class="audit-card"><span>${data.sample_size}</span><small>golden eval examples</small></div><div class="audit-card"><span>${intentAcc}</span><small>intent accuracy</small></div><div class="audit-card"><span>${escAcc}</span><small>escalation accuracy</small></div><div class="audit-card"><span>${Math.round(data.high_confidence_rate*100)}%</span><small>high-confidence intents</small></div><div class="audit-card"><span>${data.retrieval_corpus.toLocaleString()}</span><small>historical replies indexed</small></div><div class="audit-card"><span>7</span><small>intent categories</small></div><p class="audit-source">${esc(source)}</p>`;
});
