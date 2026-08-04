/* Atelier AI — results page */
(function () {
  "use strict";

  var C = window.ATELIER || {};
  var uid = C.uid || "";
  var resultUrl = C.resultUrl || null;
  var alpha = (C.resultAlpha !== null && C.resultAlpha !== undefined) ? C.resultAlpha : 0.7;

  var processingScreen = document.getElementById("processing-screen");
  var resultBody = document.getElementById("result-body");
  var compare = document.getElementById("compare");
  var alphaInput = document.getElementById("alpha");
  var alphaValue = document.getElementById("alpha-value");
  var alphaLabel = document.getElementById("alpha-label");
  var resultImg = document.getElementById("cmp-result");
  var downloadBtn = document.getElementById("btn-download");
  var restyleStatus = document.getElementById("restyle-status");
  var btnRestyle = document.getElementById("btn-restyle");
  var polling = null;

  function setAlphaUI(v) {
    alphaValue.textContent = v.toFixed(2);
    if (alphaLabel) alphaLabel.textContent = "α = " + v.toFixed(2);
    alphaInput.value = v;
    alphaInput.style.setProperty("--fill", ((v - 0.1) / 0.9) * 100 + "%");
  }

  function setResult(url) {
    resultUrl = url;
    var sep = url.indexOf("?") === -1 ? "?" : "&";
    resultImg.src = url + sep + "v=" + Date.now();
    downloadBtn.href = url;
  }

  /* ---------- Before / After comparison slider ---------- */
  function initCompare() {
    var range = document.getElementById("compare-range");
    if (!range) return;
    function setPos(v) {
      compare.style.setProperty("--pos", v + "%");
    }
    setPos(range.value);
    range.addEventListener("input", function () { setPos(range.value); });
  }

  /* ---------- Re-stylize with new alpha ---------- */
  function doRestyle() {
    if (restyleStatus) restyleStatus.hidden = false;
    alphaInput.disabled = true;
    if (btnRestyle) btnRestyle.disabled = true;
    resultImg.style.opacity = "0.45";

    fetch("/restylize", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-Requested-With": "XMLHttpRequest" },
      body: JSON.stringify({ uid: uid, alpha: Number(alphaInput.value) })
    })
      .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
      .then(function (res) {
        if (!res.ok) throw new Error(res.data.error || "Re-stylize failed.");
        alpha = Number(res.data.alpha);
        setAlphaUI(alpha);
        setResult(res.data.result_url);
      })
      .catch(function (err) { alert(err.message); })
      .finally(function () {
        if (restyleStatus) restyleStatus.hidden = true;
        alphaInput.disabled = false;
        if (btnRestyle) btnRestyle.disabled = false;
        resultImg.style.opacity = "1";
      });
  }

  function initRestyle() {
    alphaInput.addEventListener("change", doRestyle);
    alphaInput.addEventListener("input", function () {
      setAlphaUI(Number(alphaInput.value));
    });
    if (btnRestyle) btnRestyle.addEventListener("click", doRestyle);
  }

  /* ---------- Poll until ready (used when navigating before render) ---------- */
  function pollStatus() {
    var steps = ["Encoding features…", "Matching style statistics…", "Rendering output…"];
    var i = 0;
    var txt = document.getElementById("p-text");
    var bar = document.getElementById("p-bar");
    if (bar) bar.style.width = "40%";
    polling = setInterval(function () {
      fetch("/api/status?uid=" + encodeURIComponent(uid))
        .then(function (r) { return r.json(); })
        .then(function (d) {
          if (d.ready) {
            clearInterval(polling);
            if (txt) txt.textContent = "Done";
            if (bar) bar.style.width = "100%";
            setTimeout(function () {
              resultUrl = d.result_url;
              alpha = parseFloat((d.alpha !== undefined) ? d.alpha : alpha);
              setAlphaUI(alpha);
              setResult(d.result_url);
              showResult();
            }, 350);
          } else {
            if (txt) txt.textContent = steps[i++ % steps.length];
            if (bar) bar.style.width = Math.min((i * 18), 80) + "%";
          }
        })
        .catch(function () {});
    }, 1500);
  }

  function showResult() {
    processingScreen.hidden = true;
    resultBody.hidden = false;
  }

  /* ---------- Share ---------- */
  function initShare() {
    var shareUrl = window.location.href;
    var title = "My artwork made with Atelier AI";
    var text = "I turned a photo into art with Atelier AI — a local neural style transfer studio.";

    var copyBtn = document.getElementById("btn-copy");
    if (copyBtn) {
      copyBtn.addEventListener("click", function () {
        var done = function () {
          copyBtn.textContent = "Copied!";
          setTimeout(function () { copyBtn.textContent = "Copy link"; }, 1600);
        };
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(shareUrl).then(done).catch(function () { fallbackCopy(shareUrl); done(); });
        } else {
          fallbackCopy(shareUrl);
          done();
        }
      });
    }

    var shareBtn = document.getElementById("btn-share");
    if (shareBtn) {
      if (navigator.share) {
        shareBtn.addEventListener("click", function () {
          navigator.share({ title: title, text: text, url: shareUrl }).catch(function () {});
        });
      } else {
        shareBtn.hidden = true;
      }
    }

    function setHref(id, url) {
      var el = document.getElementById(id);
      if (el) el.href = url;
    }
    setHref("share-wa", "https://wa.me/?text=" + encodeURIComponent(text + " " + shareUrl));
    setHref("share-x", "https://twitter.com/intent/tweet?text=" + encodeURIComponent(text) + "&url=" + encodeURIComponent(shareUrl));
    setHref("share-fb", "https://www.facebook.com/sharer/sharer.php?u=" + encodeURIComponent(shareUrl));
  }

  function fallbackCopy(text) {
    var ta = document.createElement("textarea");
    ta.value = text;
    ta.style.position = "fixed";
    ta.style.opacity = "0";
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand("copy"); } catch (e) {}
    document.body.removeChild(ta);
  }

  /* ---------- Boot ---------- */
  function boot() {
    initCompare();
    initShare();
    setAlphaUI(alpha);

    if (resultUrl) {
      showResult();
      setResult(resultUrl);
    } else {
      pollStatus();
    }

    if (initRestyle) initRestyle();
  }

  document.addEventListener("DOMContentLoaded", boot);
})();
