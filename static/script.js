// Same origin by default (FastAPI serves this page).
// For a separately hosted frontend, set e.g. "https://your-api.onrender.com"
const API_BASE = "";

const RING = 2 * Math.PI * 100;
const CX = 120, CY = 120;

const $ = (id) => document.getElementById(id);
const form = $("loan-form");
const submitBtn = $("submit-btn");
const formError = $("form-error");
const panel = document.querySelector(".panel");
const arc = $("arc");
const marker = $("marker");
const tick = $("tick");
const pct = $("pct");
const verdict = $("verdict");
const summary = $("summary");

const EXAMPLES = {
  steady: {
    person_age: 36, person_income: 92000, person_emp_length: 9,
    person_home_ownership: "MORTGAGE", loan_amnt: 8000, loan_int_rate: 7.5,
    loan_intent: "EDUCATION", loan_grade: "A",
    cb_person_cred_hist_length: 12, cb_person_default_on_file: "N",
  },
  stretched: {
    person_age: 23, person_income: 26000, person_emp_length: 1,
    person_home_ownership: "RENT", loan_amnt: 14000, loan_int_rate: 17.8,
    loan_intent: "MEDICAL", loan_grade: "E",
    cb_person_cred_hist_length: 2, cb_person_default_on_file: "Y",
  },
};

/* ---------- Helpers ---------- */

const fmtPct = (v, d = 1) => `${(v * 100).toFixed(d)}%`;

const probDigits = (p) => (p * 100 < 1 ? 2 : 1);
const fmtProb = (p) => `${(p * 100).toFixed(probDigits(p))}%`;

// Ring starts at 12 o'clock and runs clockwise: 0% at the top, 100% back at the top.
function setGauge(p) {
  arc.style.strokeDashoffset = RING * (1 - p);
  marker.style.transform = `rotate(${p * 360}deg)`;
}

function placeTick(t) {
  const a = t * 2 * Math.PI;
  const [r1, r2] = [84, 116];
  tick.setAttribute("x1", CX + r1 * Math.sin(a));
  tick.setAttribute("y1", CY - r1 * Math.cos(a));
  tick.setAttribute("x2", CX + r2 * Math.sin(a));
  tick.setAttribute("y2", CY - r2 * Math.cos(a));
  tick.classList.add("show");
}

function countUp(p, duration = 1300) {
  const digits = probDigits(p);
  const start = performance.now();
  const ease = (x) => 1 - Math.pow(1 - x, 4);
  (function frame(now) {
    const k = Math.min((now - start) / duration, 1);
    pct.textContent = (p * 100 * ease(k)).toFixed(digits);
    if (k < 1) requestAnimationFrame(frame);
  })(start);
}

function showError(msg) {
  formError.textContent = msg;
  formError.hidden = false;
}

/* ---------- Loan-to-income readout ---------- */

function loanRatio() {
  const income = parseFloat(form.person_income.value);
  const amount = parseFloat(form.loan_amnt.value);
  if (!income || !amount || income <= 0) return null;
  return amount / income;
}

function updateRatio() {
  const r = loanRatio();
  const fill = $("ratio-fill");
  if (r === null) {
    $("ratio-value").textContent = "–";
    fill.style.width = "0";
    return;
  }
  $("ratio-value").textContent = fmtPct(r, 0);
  fill.style.width = `${Math.min(r, 1) * 100}%`;
  fill.style.background = r > 0.4 ? "var(--risk)" : r > 0.25 ? "var(--warn)" : "var(--accent)";
}

form.person_income.addEventListener("input", updateRatio);
form.loan_amnt.addEventListener("input", updateRatio);

/* ---------- Examples ---------- */

document.querySelectorAll("[data-example]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const data = EXAMPLES[btn.dataset.example];
    for (const [name, value] of Object.entries(data)) {
      const els = form.elements[name];
      if (els instanceof RadioNodeList) els.value = value;
      else els.value = value;
    }
    clearInvalid();
    updateRatio();
  });
});

/* ---------- Validation ---------- */

function clearInvalid() {
  formError.hidden = true;
  form.querySelectorAll(".invalid").forEach((el) => el.classList.remove("invalid"));
}

function validate() {
  clearInvalid();
  let firstBad = null;

  form.querySelectorAll("input[type=number], select").forEach((el) => {
    if (!el.checkValidity()) {
      el.classList.add("invalid");
      firstBad = firstBad || el;
    }
  });

  ["person_home_ownership", "loan_grade", "cb_person_default_on_file"].forEach((name) => {
    if (!form.elements[name].value) {
      const group = form.querySelector(`input[name=${name}]`).closest(".chips");
      group.classList.add("invalid");
      firstBad = firstBad || group;
    }
  });

  if (firstBad) {
    showError("Some fields are empty or out of range. Complete the highlighted fields and try again.");
    firstBad.scrollIntoView({ block: "center", behavior: "smooth" });
    return false;
  }
  return true;
}

form.addEventListener("input", (e) => {
  e.target.classList.remove("invalid");
  e.target.closest(".chips")?.classList.remove("invalid");
});

/* ---------- Submit ---------- */

function buildPayload() {
  const f = form.elements;
  return {
    person_age: parseInt(f.person_age.value, 10),
    person_income: parseFloat(f.person_income.value),
    person_home_ownership: f.person_home_ownership.value,
    person_emp_length: parseFloat(f.person_emp_length.value),
    loan_intent: f.loan_intent.value,
    loan_grade: f.loan_grade.value,
    loan_amnt: parseFloat(f.loan_amnt.value),
    loan_int_rate: parseFloat(f.loan_int_rate.value),
    loan_percent_income: Math.round(loanRatio() * 100) / 100,
    cb_person_default_on_file: f.cb_person_default_on_file.value,
    cb_person_cred_hist_length: parseInt(f.cb_person_cred_hist_length.value, 10),
  };
}

function renderResult(res, payload) {
  const p = res.default_probability;
  const t = res.threshold;
  const high = res.default_prediction === 1;

  placeTick(t);
  setGauge(p);
  countUp(p);

  panel.classList.remove("low", "high");
  panel.classList.add(high ? "high" : "low");

  verdict.className = `verdict ${high ? "high" : "low"} pop`;
  verdict.textContent = high ? "High risk" : "Low risk";

  const gap = Math.abs(p - t);
  summary.textContent = high
    ? `The estimated chance of default is ${fmtProb(p)}, which is above the ${fmtPct(t)} threshold. This application should be reviewed closely.`
    : `The estimated chance of default is ${fmtProb(p)}, which is below the ${fmtPct(t)} threshold. This application fits the low-risk profile.`;

  $("fact-threshold").textContent = fmtPct(t);
  $("fact-margin").textContent = `${(gap * 100).toFixed(1)} pts ${high ? "above" : "below"}`;
  $("fact-ratio").textContent = fmtPct(payload.loan_percent_income, 0);

  if (window.matchMedia("(max-width: 960px)").matches) {
    $("result").scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!validate()) return;

  const payload = buildPayload();
  submitBtn.disabled = true;
  submitBtn.classList.add("loading");
  submitBtn.querySelector(".label").textContent = "Assessing";

  try {
    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      let detail = "";
      try {
        const body = await response.json();
        if (Array.isArray(body.detail)) {
          detail = body.detail.map((d) => `${d.loc?.slice(-1)[0]}: ${d.msg}`).join("; ");
        }
      } catch (_) { /* response was not JSON */ }
      throw new Error(detail || `The server returned an error (${response.status}).`);
    }

    renderResult(await response.json(), payload);
  } catch (err) {
    showError(
      err instanceof TypeError
        ? "Could not reach the model. Check that the server is running, then try again."
        : err.message
    );
  } finally {
    submitBtn.disabled = false;
    submitBtn.classList.remove("loading");
    submitBtn.querySelector(".label").textContent = "Assess risk";
  }
});

/* ---------- Page load: one sweep of the ring ---------- */

window.addEventListener("load", () => {
  setTimeout(() => {
    setGauge(0.35);
    setTimeout(() => setGauge(0), 900);
  }, 200);
});
