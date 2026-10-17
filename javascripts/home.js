document.addEventListener("DOMContentLoaded", function () {
  var papersEl = document.getElementById("home-papers");
  var codeEl = document.getElementById("home-code");
  if (!papersEl && !codeEl) return;

  var scripts = document.getElementsByTagName("script");
  var base = "";
  var cacheQuery = "";
  for (var i = 0; i < scripts.length; i++) {
    var src = scripts[i].src;
    var idx = src.indexOf("/javascripts/home.js");
    if (idx !== -1) {
      base = src.substring(0, idx);
      var qidx = src.indexOf("?");
      cacheQuery = qidx !== -1 ? src.substring(qidx) : "";
      break;
    }
  }

  fetch(base + "/data/tag_index.json" + cacheQuery, { cache: "no-store" })
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (docs) {
      var papers = [];
      var code = [];
      var codeProjects = {};
      docs.forEach(function (d) {
        if (d.type === "论文阅读") {
          papers.push(d);
        } else if (d.type === "代码分析") {
          // group by project: use first path segment under code_analysis/
          var parts = d.path.split("/");
          var project = parts.length >= 3 ? parts[1] : parts[0];
          if (!codeProjects[project]) {
            codeProjects[project] = d;
          }
        }
      });
      for (var k in codeProjects) code.push(codeProjects[k]);
      if (papersEl) papersEl.innerHTML = renderList(papers, base);
      if (codeEl) codeEl.innerHTML = renderList(code, base);
    })
    .catch(function () {});

  function renderList(items, base) {
    if (items.length === 0) return "";
    var html = '<div class="home-items">';
    items.forEach(function (d) {
      var label = d.title.indexOf(" - ") !== -1 ? d.title.split(" - ")[0] : d.title;
      var url = base + "/" + d.path.replace(/\.md$/, "/");
      html += '<a class="home-item" href="' + url + '">' + label + "</a>";
    });
    html += "</div>";
    return html;
  }
});
