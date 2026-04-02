document.addEventListener("DOMContentLoaded", function () {
  var container = document.getElementById("paper-index");
  if (!container) return;

  var scripts = document.getElementsByTagName("script");
  var base = "";
  for (var i = 0; i < scripts.length; i++) {
    var src = scripts[i].src;
    var idx = src.indexOf("/javascripts/paper_index.js");
    if (idx !== -1) {
      base = src.substring(0, idx);
      break;
    }
  }

  var docs = [];
  var currentSort = "added";

  fetch(base + "/data/tag_index.json")
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (data) {
      docs = data.filter(function (d) { return d.type === "论文阅读"; });
      render();
    })
    .catch(function () {
      container.innerHTML = '<p style="opacity:0.5">索引加载失败</p>';
    });

  var buttons = document.querySelectorAll(".sort-btn");
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].addEventListener("click", function () {
      for (var j = 0; j < buttons.length; j++) buttons[j].classList.remove("active");
      this.classList.add("active");
      currentSort = this.getAttribute("data-sort");
      render();
    });
  }

  function render() {
    var groups = {};
    var catOrder = [];
    docs.forEach(function (d) {
      var cat = d.category || "其他";
      if (!groups[cat]) {
        groups[cat] = [];
        catOrder.push(cat);
      }
      groups[cat].push(d);
    });

    catOrder.forEach(function (cat) {
      groups[cat].sort(function (a, b) {
        var da, db;
        if (currentSort === "arxiv") {
          da = a.date_arxiv || "";
          db = b.date_arxiv || "";
        } else {
          da = a.date_added || "";
          db = b.date_added || "";
        }
        return da > db ? -1 : da < db ? 1 : 0;
      });
    });

    var html = "";
    catOrder.forEach(function (cat) {
      html += '<h2>' + cat + '</h2>';
      html += '<div class="paper-list">';
      groups[cat].forEach(function (d) {
        var url = base + "/" + d.path.replace(/\.md$/, "/");
        var dateLabel = currentSort === "arxiv" ? d.date_arxiv : d.date_added;
        html += '<div class="paper-item">'
          + '<a href="' + url + '">' + d.title + '</a>'
          + '<span class="paper-date">' + (dateLabel || "") + '</span>'
          + '</div>';
      });
      html += '</div>';
    });

    container.innerHTML = html;
  }
});
