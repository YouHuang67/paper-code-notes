document.addEventListener("DOMContentLoaded", function () {
  var papersEl = document.getElementById("home-papers");
  var codeEl = document.getElementById("home-code");
  if (!papersEl && !codeEl) return;

  var scripts = document.getElementsByTagName("script");
  var base = "";
  for (var i = 0; i < scripts.length; i++) {
    var src = scripts[i].src;
    var idx = src.indexOf("/javascripts/home.js");
    if (idx !== -1) {
      base = src.substring(0, idx);
      break;
    }
  }

  fetch(base + "/data/tag_index.json")
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (docs) {
      var papers = [];
      var code = [];
      docs.forEach(function (d) {
        if (d.type === "论文阅读") papers.push(d);
        else if (d.type === "代码分析") code.push(d);
      });
      if (papersEl) papersEl.innerHTML = renderList(papers, base);
      if (codeEl) codeEl.innerHTML = renderList(code, base);
    })
    .catch(function () {});

  function renderList(items, base) {
    if (items.length === 0) return "";
    var html = '<div class="home-items">';
    items.forEach(function (d) {
      html += '<a class="home-item" href="' + base + "/" + d.path + '">' + d.title + "</a>";
    });
    html += "</div>";
    return html;
  }
});
