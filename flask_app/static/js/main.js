/* Atelier AI — landing page */
(function () {
  "use strict";

  /* ---------- Hero canvas: neural particle field ---------- */
  function initHero() {
    var canvas = document.getElementById("hero-canvas");
    if (!canvas) return;
    var ctx = canvas.getContext("2d");
    var W, H, dpr = Math.min(window.devicePixelRatio || 1, 2);
    var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    var particles = [];
    var N = 70;

    function resize() {
      W = canvas.offsetWidth;
      H = canvas.offsetHeight;
      canvas.width = W * dpr;
      canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }

    function makeParticle(i) {
      return {
        x: Math.random() * W,
        y: Math.random() * H,
        vx: (Math.random() - 0.5) * 0.35,
        vy: (Math.random() - 0.5) * 0.35,
        r: Math.random() * 1.6 + 0.6,
        a: i % 5 === 0
      };
    }

    function init() {
      resize();
      particles = [];
      for (var i = 0; i < N; i++) particles.push(makeParticle(i));
    }

    function draw() {
      ctx.clearRect(0, 0, W, H);
      var i, j, p, q, d;
      for (i = 0; i < particles.length; i++) {
        p = particles[i];
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < -20) p.x = W + 20;
        if (p.x > W + 20) p.x = -20;
        if (p.y < -20) p.y = H + 20;
        if (p.y > H + 20) p.y = -20;

        for (j = i + 1; j < particles.length; j++) {
          q = particles[j];
          d = Math.hypot(p.x - q.x, p.y - q.y);
          if (d < 130) {
            var t = 1 - d / 130;
            ctx.strokeStyle = p.a ? "rgba(253,186,116," + (t * 0.28) + ")" : "rgba(234,88,12," + (t * 0.18) + ")";
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
          }
        }
        ctx.fillStyle = p.a ? "rgba(253,186,116,0.9)" : "rgba(255,255,255,0.55)";
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    function loop() {
      draw();
      requestAnimationFrame(loop);
    }

    var debounce;
    window.addEventListener("resize", function () {
      clearTimeout(debounce);
      debounce = setTimeout(function () { init(); }, 150);
    });

    init();
    if (reduced) {
      draw();
    } else {
      loop();
    }
  }

  /* ---------- Reveal on scroll ---------- */
  function initReveal() {
    var els = document.querySelectorAll(".reveal");
    if (!("IntersectionObserver" in window)) {
      els.forEach(function (el) { el.classList.add("in"); });
      return;
    }
    var io = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) {
          entry.target.classList.add("in");
          io.unobserve(entry.target);
        }
      });
    }, { threshold: 0.15 });
    els.forEach(function (el) { io.observe(el); });
  }

  /* ---------- Upload form: dropzones + progress + submit ---------- */
  var state = { content: null, style: null };

  function initUpload() {
    var form = document.getElementById("stylize-form");
    if (!form) return;
    var alpha = document.getElementById("alpha");
    var alphaVal = document.getElementById("alpha-value");

    function updateAlpha() {
      alphaVal.textContent = Number(alpha.value).toFixed(2);
      alpha.style.setProperty("--fill", ((alpha.value - 0.1) / 0.9) * 100 + "%");
    }
    alpha.addEventListener("input", updateAlpha);
    updateAlpha();

    document.querySelectorAll(".dropzone").forEach(function (dz) {
      var input = dz.querySelector('input[type="file"]');
      var role = dz.dataset.role;

      dz.addEventListener("click", function () { input.click(); });
      dz.addEventListener("dragover", function (e) { e.preventDefault(); dz.classList.add("drag"); });
      dz.addEventListener("dragleave", function () { dz.classList.remove("drag"); });
      dz.addEventListener("drop", function (e) {
        e.preventDefault();
        dz.classList.remove("drag");
        if (e.dataTransfer.files.length) setFile(input, e.dataTransfer.files[0]);
      });
      input.addEventListener("change", function () {
        if (input.files.length) setFile(input, input.files[0]);
      });
    });

    function setFile(input, file) {
      if (!file.type.startsWith("image/")) return;
      var dz = input.closest(".dropzone");
      var role = dz.dataset.role;
      var prev = dz.querySelector(".dz-preview");
      var icon = dz.querySelector(".dz-icon");
      var url = URL.createObjectURL(file);
      prev.src = url;
      prev.hidden = false;
      icon.hidden = true;
      dz.classList.add("has-image");
      state[role] = file;
    }

    form.addEventListener("submit", function (e) {
      e.preventDefault();
      if (!state.content || !state.style) {
        showError("Upload both a content image and a style image first.");
        return;
      }
      var btn = document.getElementById("generate-btn");
      var btnLabel = document.getElementById("btn-label");
      var progress = document.getElementById("progress");
      var bar = document.getElementById("progress-bar");
      var text = document.getElementById("progress-text");

      btn.disabled = true;
      btnLabel.textContent = "Rendering…";
      progress.hidden = false;
      hideError();

      animateProgress(bar, text);

      var fd = new FormData();
      fd.append("content_image", state.content);
      fd.append("style_image", state.style);
      fd.append("alpha", alpha.value);

      fetch("/stylize", {
        method: "POST",
        headers: { "X-Requested-With": "XMLHttpRequest" },
        body: fd
      })
        .then(function (r) { return r.json().then(function (d) { return { ok: r.ok, data: d }; }); })
        .then(function (res) {
          if (!res.ok) throw new Error(res.data.error || "Processing failed.");
          text.textContent = "Done";
          bar.style.width = "100%";
          window.location.href = "/stylize?uid=" + res.data.uid;
        })
        .catch(function (err) {
          btn.disabled = false;
          btnLabel.textContent = "Generate artwork";
          progress.hidden = true;
          showError(err.message);
        });
    });
  }

  function animateProgress(bar, text) {
    var steps = ["Uploading images…", "Encoding features…", "Stylizing…", "Rendering output…"];
    var p = 4;
    var i = 0;
    bar.style.width = "4%";
    bar.classList.add("running");
    var timer = setInterval(function () {
      if (i < steps.length) {
        text.textContent = steps[i];
        i++;
      }
      p = Math.min(p + Math.random() * 9, 92);
      bar.style.width = p + "%";
    }, 550);
    window.__progressTimer = timer;
  }

  function showError(msg) {
    var el = document.getElementById("form-error");
    el.textContent = msg;
    el.hidden = false;
  }
  function hideError() {
    var el = document.getElementById("form-error");
    el.hidden = true;
  }

  /* ---------- Gallery from real data ---------- */
  function initGallery() {
    var grid = document.getElementById("gallery-grid");
    if (!grid) return;
    fetch("/api/samples")
      .then(function (r) { return r.json(); })
      .then(function (items) {
        if (!items.length) return;
        grid.innerHTML = "";
        items.forEach(function (item) {
          var card = document.createElement("article");
          card.className = "gallery-card card reveal in";
          card.innerHTML =
            '<div class="gallery-media">' +
            '<img src="' + item.result_url + '" alt="' + item.title + '" loading="lazy">' +
            '<div class="gallery-overlay"><span>CONTENT × ' + (item.alpha * 100).toFixed(0) + '% STYLE</span></div>' +
            "</div>" +
            '<div class="gallery-cap"><strong>' + item.title + '</strong><em>α ' + item.alpha.toFixed(2) + "</em></div>";
          grid.appendChild(card);
        });
      })
      .catch(function () { /* keep skeletons off */ grid.innerHTML = ""; });
  }

  /* ---------- Carousel from real data ---------- */
  function initCarousel() {
    var windowEl = document.getElementById("carousel-window");
    var prevBtn = document.getElementById("carousel-prev");
    var nextBtn = document.getElementById("carousel-next");
    var dotsEl = document.getElementById("carousel-dots");
    if (!windowEl) return;

    fetch("/api/samples")
      .then(function (r) { return r.json(); })
      .then(function (items) {
        if (!items.length) return;
        var grouped = groupSamples(items);
        if (!grouped.length) return;
        var idx = 0;

        var track = document.createElement("div");
        track.className = "carousel-track";
        grouped.forEach(function (g) {
          var slide = document.createElement("div");
          slide.className = "carousel-slide";
          slide.innerHTML =
            '<div class="carousel-img"><img src="' + g.before + '" alt="Before"></div>' +
            '<div class="carousel-img"><img src="' + g.after + '" alt="After"></div>' +
            '<div class="carousel-caption">' +
            '<span class="kicker">Before → After</span>' +
            '<h3>' + g.title + '</h3>' +
            '<p>Same content and style rendered at α ' + g.alpha.toFixed(2) + ' — produced live by the local AdaIN model.</p>' +
            '<div class="carousel-tags"><span class="carousel-tag">CONTENT</span><span class="carousel-tag">STYLE ' + (g.alpha * 100).toFixed(0) + '%</span><span class="carousel-tag">ADAIN</span></div>' +
            "</div>";
          track.appendChild(slide);
        });
        windowEl.appendChild(track);

        var slides = track.children.length;
        grouped.forEach(function (g, i) {
          var dot = document.createElement("button");
          dot.className = "carousel-dot" + (i === 0 ? " active" : "");
          dot.setAttribute("aria-label", "Slide " + (i + 1));
          dot.addEventListener("click", function () { go(i); });
          dotsEl.appendChild(dot);
        });
        var dots = dotsEl.querySelectorAll(".carousel-dot");

        function go(i) {
          idx = (i + slides) % slides;
          track.style.transform = "translateX(-" + idx * 100 + "%)";
          dots.forEach(function (d, di) { d.classList.toggle("active", di === idx); });
        }
        prevBtn.addEventListener("click", function () { go(idx - 1); });
        nextBtn.addEventListener("click", function () { go(idx + 1); });

        var auto = setInterval(function () { go(idx + 1); }, 5000);
        windowEl.addEventListener("mouseenter", function () { clearInterval(auto); });
        windowEl.addEventListener("mouseleave", function () {
          clearInterval(auto);
          auto = setInterval(function () { go(idx + 1); }, 5000);
        });
      })
      .catch(function () {});
  }

  function groupSamples(items) {
    var byId = {};
    items.forEach(function (item) {
      if (!byId[item.id]) byId[item.id] = [];
      byId[item.id].push(item);
    });
    var out = [];
    Object.keys(byId).forEach(function (id) {
      var list = byId[id].sort(function (a, b) { return a.alpha - b.alpha; });
      var first = list[0];
      var last = list[list.length - 1];
      out.push({ title: first.title, before: first.content_url, after: last.result_url, alpha: last.alpha });
    });
    return out;
  }

  /* ---------- Boot ---------- */
  document.addEventListener("DOMContentLoaded", function () {
    initHero();
    initReveal();
    initUpload();
    initGallery();
    initCarousel();
  });
})();
