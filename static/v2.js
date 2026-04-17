(() => {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const els = {
    dropzone:  $("#dropzone"),
    fileInput: $("#file-input"),
    preview:   $("#preview"),
    dzHint:    $("#dz-hint"),
    go:        $("#go"),
    hair:      $("#hair-length"),
    k:         $("#k"),
    status:    $("#status"),

    panelAnalysis: $("#panel-analysis"),
    panelRecs:     $("#panel-recs"),

    annotated:     $("#annotated"),
    primary:       $("#primary-shape"),
    secondaryWrap: $("#secondary-wrap"),
    secondary:     $("#secondary-shape"),
    confidence:    $("#confidence"),
    advice:        $("#advice"),
    bars:          $("#prob-bars"),
    features:      $("#feature-table"),
    recs:          $("#recs"),
  };

  let uploaded = null; // { filename, url }

  // --- upload ---------------------------------------------------------------

  function setStatus(msg, kind) {
    if (!msg) { els.status.hidden = true; return; }
    els.status.hidden = false;
    els.status.className = "status " + (kind || "");
    els.status.textContent = msg;
  }

  function showPreview(file) {
    const url = URL.createObjectURL(file);
    els.preview.src = url;
    els.preview.hidden = false;
    els.dzHint.hidden = true;
  }

  async function upload(file) {
    showPreview(file);
    setStatus("Uploading…", "busy");
    els.go.disabled = true;

    const fd = new FormData();
    fd.append("file", file);
    const res = await fetch("/api/v2/upload", { method: "POST", body: fd });
    const body = await res.json();
    if (!res.ok) {
      setStatus(body.error || "Upload failed", "error");
      return;
    }
    uploaded = body;
    els.go.disabled = false;
    setStatus(null);
  }

  els.dropzone.addEventListener("click", () => els.fileInput.click());
  els.fileInput.addEventListener("change", (e) => {
    if (e.target.files[0]) upload(e.target.files[0]);
  });
  ["dragenter", "dragover"].forEach(ev =>
    els.dropzone.addEventListener(ev, e => { e.preventDefault(); els.dropzone.classList.add("drag"); })
  );
  ["dragleave", "drop"].forEach(ev =>
    els.dropzone.addEventListener(ev, e => { e.preventDefault(); els.dropzone.classList.remove("drag"); })
  );
  els.dropzone.addEventListener("drop", (e) => {
    const f = e.dataTransfer.files[0];
    if (f) { els.fileInput.files = e.dataTransfer.files; upload(f); }
  });

  // --- analyze --------------------------------------------------------------

  els.go.addEventListener("click", async () => {
    if (!uploaded) return;
    setStatus("Analyzing face & ranking styles… (first run builds the CLIP index)", "busy");
    els.go.disabled = true;

    try {
      const res = await fetch("/api/v2/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          filename: uploaded.filename,
          hair_length: els.hair.value || null,
          k: parseInt(els.k.value, 10),
        }),
      });
      const body = await res.json();
      if (!res.ok || !body.success) {
        setStatus(body.error || "Could not detect a face in this image.", "error");
        return;
      }
      render(body);
      setStatus(null);
    } catch (err) {
      setStatus("Request failed: " + err.message, "error");
    } finally {
      els.go.disabled = false;
    }
  });

  // --- render ---------------------------------------------------------------

  function render(data) {
    const face = data.face;

    els.annotated.src = data.annotated_img || uploaded.url;
    els.primary.textContent = face.primary_shape;
    els.confidence.textContent = (face.confidence * 100).toFixed(0) + "%";

    const showSecondary = face.confidence < 0.2 && face.secondary_shape;
    els.secondaryWrap.hidden = !showSecondary;
    if (showSecondary) els.secondary.textContent = face.secondary_shape;

    els.advice.textContent = face.advice || "";

    renderBars(face.shape_probabilities, face.primary_shape);
    renderFeatures(face.features);
    renderRecs(data.recommendations);

    els.panelAnalysis.hidden = false;
    els.panelRecs.hidden = false;
    els.panelAnalysis.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  function renderBars(probs, primary) {
    const entries = Object.entries(probs).sort((a, b) => b[1] - a[1]);
    els.bars.innerHTML = entries.map(([shape, p]) => {
      const pct = (p * 100).toFixed(1);
      const top = shape === primary ? " top" : "";
      return (
        `<div class="bar${top}">` +
          `<span>${shape}</span>` +
          `<span class="track"><span class="fill" style="width:${pct}%"></span></span>` +
          `<span class="pct">${pct}%</span>` +
        `</div>`
      );
    }).join("");
  }

  function renderFeatures(features) {
    els.features.innerHTML = Object.entries(features)
      .map(([k, v]) => `<tr><td>${k}</td><td>${Number(v).toFixed(3)}</td></tr>`)
      .join("");
  }

  function renderRecs(recs) {
    els.recs.innerHTML = recs.map(r => (
      `<article class="rec">` +
        `<img src="${r.url}" alt="${r.face_shape} ${r.hair_length}" loading="lazy">` +
        `<div class="meta">` +
          `<div class="tags">` +
            `<span class="tag">${r.face_shape}</span>` +
            `<span class="tag">${r.hair_length}</span>` +
          `</div>` +
          `<span class="score">match ${r.score.toFixed(3)} ` +
            `(clip ${r.clip_similarity.toFixed(3)} + shape ${r.shape_bonus.toFixed(3)})</span>` +
          `<span class="reason">${r.reason || ""}</span>` +
        `</div>` +
      `</article>`
    )).join("");
  }
})();
