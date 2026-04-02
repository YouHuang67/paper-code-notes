document.addEventListener("DOMContentLoaded", function () {
  var input = document.getElementById("tag-search-input");
  var results = document.getElementById("tag-search-results");
  var sortControls = document.getElementById("tag-sort-controls");
  if (!input || !results) return;

  var docs = [];
  var lastResults = [];
  var currentSort = "relevance";

  var scripts = document.getElementsByTagName("script");
  var siteBase = "";
  for (var i = 0; i < scripts.length; i++) {
    var src = scripts[i].src;
    var idx = src.indexOf("/javascripts/tag_search.js");
    if (idx !== -1) {
      siteBase = src.substring(0, idx);
      break;
    }
  }

  fetch(siteBase + "/data/tag_index.json")
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (data) { docs = data; })
    .catch(function () {
      results.innerHTML = '<p class="tag-search-error">索引加载失败</p>';
    });

  var buttons = sortControls ? sortControls.querySelectorAll(".sort-btn") : [];
  for (var i = 0; i < buttons.length; i++) {
    buttons[i].addEventListener("click", function () {
      for (var j = 0; j < buttons.length; j++) buttons[j].classList.remove("active");
      this.classList.add("active");
      currentSort = this.getAttribute("data-sort");
      renderResults(lastResults);
    });
  }

  input.addEventListener("input", function () {
    var query = input.value.trim().toLowerCase();
    if (!query) {
      results.innerHTML = "";
      if (sortControls) sortControls.style.display = "none";
      lastResults = [];
      return;
    }

    var exact = [];
    var fuzzy = [];

    docs.forEach(function (doc) {
      var matchedTags = [];
      var isExact = false;

      doc.tags.forEach(function (tag) {
        var tagLower = tag.toLowerCase();
        if (tagLower === query) {
          matchedTags.push(tag);
          isExact = true;
        } else if (tagLower.indexOf(query) !== -1) {
          matchedTags.push(tag);
        }
      });

      if (matchedTags.length > 0) {
        var item = { doc: doc, matchedTags: matchedTags, isExact: isExact };
        if (isExact) {
          exact.push(item);
        } else {
          fuzzy.push(item);
        }
      }
    });

    lastResults = exact.concat(fuzzy);

    if (lastResults.length === 0) {
      results.innerHTML = '<p class="tag-search-empty">未找到匹配的文档</p>';
      if (sortControls) sortControls.style.display = "none";
      return;
    }

    if (sortControls) sortControls.style.display = "flex";
    renderResults(lastResults);
  });

  function renderResults(all) {
    if (all.length === 0) return;

    var sorted = all.slice();
    if (currentSort === "added") {
      sorted.sort(function (a, b) {
        var da = a.doc.date_added || "";
        var db = b.doc.date_added || "";
        return da > db ? -1 : da < db ? 1 : 0;
      });
    } else if (currentSort === "arxiv") {
      sorted.sort(function (a, b) {
        var da = a.doc.date_arxiv || "";
        var db = b.doc.date_arxiv || "";
        return da > db ? -1 : da < db ? 1 : 0;
      });
    }

    var linkBase = siteBase + "/";
    var html = '<p class="tag-search-count">' + sorted.length + ' 个符合条件的结果</p>';
    sorted.forEach(function (item) {
      var doc = item.doc;
      var tagsHtml = doc.tags.map(function (t) {
        var cls = item.matchedTags.indexOf(t) !== -1 ? "tag-chip tag-chip-match" : "tag-chip";
        return '<span class="' + cls + '">' + t + '</span>';
      }).join(" ");

      var dateHtml = "";
      if (doc.date_added || doc.date_arxiv) {
        dateHtml = '<div class="tag-search-dates">';
        if (doc.date_added) dateHtml += '<span>添加: ' + doc.date_added + '</span>';
        if (doc.date_arxiv) dateHtml += '<span>arXiv: ' + doc.date_arxiv + '</span>';
        dateHtml += '</div>';
      }

      html += '<div class="tag-search-card">'
        + '<a class="tag-search-title" href="' + linkBase + doc.path.replace(/\.md$/, "/") + '">' + doc.title + '</a>'
        + '<span class="tag-search-type">' + doc.type + '</span>'
        + '<div class="tag-search-tags">' + tagsHtml + '</div>'
        + dateHtml
        + '<p class="tag-search-summary">' + doc.summary + '</p>'
        + '</div>';
    });

    results.innerHTML = html;
  }
});
