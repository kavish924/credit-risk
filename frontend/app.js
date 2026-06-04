

const API_BASE = window.location.origin;

const form = document.getElementById('predict-form');
const btnPredict = document.getElementById('btn-predict');
const btnReset = document.getElementById('btn-reset');
const resultPlaceholder = document.getElementById('results-placeholder');
const resultContent = document.getElementById('results-content');
const apiStatusBadge = document.getElementById('api-status');
const toastContainer = document.getElementById('toast-container');

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(5000) });
    const data = await res.json();
    if (data.status === 'healthy') {
      apiStatusBadge.className = 'status-badge status-healthy';
      apiStatusBadge.querySelector('.status-text').textContent = `Model v${data.model_version} · Healthy`;
    } else {
      apiStatusBadge.className = 'status-badge status-unhealthy';
      apiStatusBadge.querySelector('.status-text').textContent = 'Model Unhealthy';
    }
  } catch (err) {
    apiStatusBadge.className = 'status-badge status-unhealthy';
    apiStatusBadge.querySelector('.status-text').textContent = 'API Offline';
  }
}

document.addEventListener('DOMContentLoaded', checkHealth);


document.querySelectorAll('.toggle-buttons').forEach(group => {
  group.querySelectorAll('.toggle-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      group.querySelectorAll('.toggle-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
    });
  });
});


function buildPayload() {
  const payload = {};

  // Credit amount
  const credit = parseFloat(document.getElementById('input-credit').value);
  if (!credit || credit <= 0) throw new Error('Credit Amount is required and must be > 0');
  payload.AMT_CREDIT = credit;

  // Income
  const income = parseFloat(document.getElementById('input-income').value);
  if (!income || income <= 0) throw new Error('Annual Income is required and must be > 0');
  payload.AMT_INCOME_TOTAL = income;

  // Age → DAYS_BIRTH (negative)
  const age = parseInt(document.getElementById('input-age').value, 10);
  if (!age || age < 18) throw new Error('Age is required and must be ≥ 18');
  payload.DAYS_BIRTH = -(age * 365);

  // Annuity (optional)
  const annuity = document.getElementById('input-annuity').value;
  if (annuity) payload.AMT_ANNUITY = parseFloat(annuity);

  // Employment years → DAYS_EMPLOYED (negative)
  const emp = document.getElementById('input-employment').value;
  if (emp) payload.DAYS_EMPLOYED = -(parseFloat(emp) * 365);

  // Contract type
  payload.NAME_CONTRACT_TYPE = document.getElementById('input-contract-type').value;

  // Gender
  payload.CODE_GENDER = document.getElementById('input-gender').value;

  // Goods price (optional)
  const goods = document.getElementById('input-goods-price').value;
  if (goods) payload.AMT_GOODS_PRICE = parseFloat(goods);

  // Region population (optional)
  const regPop = document.getElementById('input-region-pop').value;
  if (regPop) payload.REGION_POPULATION_RELATIVE = parseFloat(regPop);

  // Days registration (optional)
  const daysReg = document.getElementById('input-days-reg').value;
  if (daysReg) payload.DAYS_REGISTRATION = parseFloat(daysReg);

  // Days ID publish (optional)
  const daysId = document.getElementById('input-days-id').value;
  if (daysId) payload.DAYS_ID_PUBLISH = parseInt(daysId, 10);

  // Children
  const children = document.getElementById('input-children').value;
  if (children !== '') payload.CNT_CHILDREN = parseInt(children, 10);

  // Family members
  const family = document.getElementById('input-family').value;
  if (family !== '') payload.CNT_FAM_MEMBERS = parseFloat(family);

  // Toggle buttons
  document.querySelectorAll('.toggle-btn.active').forEach(btn => {
    const name = btn.dataset.name;
    const val = btn.dataset.value;
    if (name === 'LIVE_CITY_NOT_WORK_CITY') {
      payload[name] = parseInt(val, 10);
    } else {
      payload[name] = val;
    }
  });

  return payload;
}

function renderResults(data) {
  resultPlaceholder.classList.add('hidden');
  resultContent.classList.remove('hidden');
  resultContent.classList.remove('animate-in');
  // Force reflow for re-animation
  void resultContent.offsetWidth;
  resultContent.classList.add('animate-in');

  // Gauge
  const score = data.risk_score;
  const pct = score / 1000;  // 0 (worst) → 1 (best)
  const needleAngle = -90 + (1 - pct) * 180; // -90 (best/left) → +90 (worst/right)
  document.getElementById('gauge-needle').style.transform = `rotate(${needleAngle}deg)`;
  document.getElementById('gauge-value').textContent = score;

  // Metrics
  document.getElementById('val-probability').textContent =
    (data.default_probability * 100).toFixed(2) + '%';

  const riskEl = document.getElementById('val-risk-label');
  riskEl.textContent = data.risk_label;
  riskEl.className = `metric-value risk-badge risk-${data.risk_label}`;

  document.getElementById('val-score').textContent = score;
  document.getElementById('val-model').textContent = data.model_name;
  document.getElementById('val-model-version').textContent = `v${data.model_version}`;

  // Interpretation
  const interpBox = document.getElementById('interpretation-box');
  const interpTitle = document.getElementById('interpretation-title');
  const interpText = document.getElementById('interpretation-text');

  if (data.risk_label === 'LOW') {
    interpBox.style.borderLeftColor = 'var(--risk-low)';
    interpTitle.textContent = '✅ Low Risk';
    interpText.textContent =
      `This application has a low default probability of ${(data.default_probability * 100).toFixed(2)}%. ` +
      `With a credit score of ${score}/1000, the applicant demonstrates strong creditworthiness. ` +
      `The model recommends approval, subject to standard verification.`;
  } else if (data.risk_label === 'MEDIUM') {
    interpBox.style.borderLeftColor = 'var(--risk-medium)';
    interpTitle.textContent = '⚠️ Medium Risk';
    interpText.textContent =
      `This application has a moderate default probability of ${(data.default_probability * 100).toFixed(2)}%. ` +
      `With a credit score of ${score}/1000, the applicant sits in the medium-risk tier. ` +
      `Consider requesting additional documentation or offering adjusted terms.`;
  } else {
    interpBox.style.borderLeftColor = 'var(--risk-high)';
    interpTitle.textContent = '🚨 High Risk';
    interpText.textContent =
      `This application has a high default probability of ${(data.default_probability * 100).toFixed(2)}%. ` +
      `With a credit score of ${score}/1000, the model flags significant default risk. ` +
      `Recommend further manual review or declining the application.`;
  }
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  toastContainer.appendChild(toast);
  setTimeout(() => {
    toast.classList.add('toast-exit');
    toast.addEventListener('animationend', () => toast.remove());
  }, 4000);
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();

  let payload;
  try {
    payload = buildPayload();
  } catch (err) {
    showToast(err.message, 'error');
    return;
  }

  // Loading state
  btnPredict.classList.add('loading');
  btnPredict.disabled = true;

  try {
    const res = await fetch(`${API_BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errData = await res.json().catch(() => ({}));
      throw new Error(errData.detail || `Server error (${res.status})`);
    }

    const data = await res.json();
    renderResults(data);
    showToast('Prediction complete!', 'success');
  } catch (err) {
    showToast(err.message || 'Failed to get prediction', 'error');
  } finally {
    btnPredict.classList.remove('loading');
    btnPredict.disabled = false;
  }
});


form.addEventListener('reset', () => {
  // Small delay so the browser default reset fires first
  setTimeout(() => {
    resultContent.classList.add('hidden');
    resultPlaceholder.classList.remove('hidden');

    // Reset toggle buttons to default
    document.querySelectorAll('.toggle-buttons').forEach(group => {
      group.querySelectorAll('.toggle-btn').forEach((btn, i) => {
        btn.classList.toggle('active', i === 0);
      });
    });

    // Reset gauge
    document.getElementById('gauge-needle').style.transform = 'rotate(0deg)';
    document.getElementById('gauge-value').textContent = '—';
  }, 0);
});
