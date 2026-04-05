document.addEventListener("DOMContentLoaded", function () {
  var container = document.getElementById("paper-index");
  var filterArea = document.getElementById("paper-tag-filter");
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
  var currentSort = "arxiv";
  var STORAGE_KEY = "paper_index_selected_tags";
  var selectedTags = (function () {
    try {
      var saved = sessionStorage.getItem(STORAGE_KEY);
      return saved ? JSON.parse(saved) : [];
    } catch (e) { return []; }
  }());

  fetch(base + "/data/tag_index.json")
    .then(function (r) {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    })
    .then(function (data) {
      docs = data.filter(function (d) { return d.type === "论文阅读"; });
      renderTagFilter();
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

  function saveSelectedTags() {
    try { sessionStorage.setItem(STORAGE_KEY, JSON.stringify(selectedTags)); } catch (e) {}
  }

  function renderTagFilter() {
    if (!filterArea) return;
    var tagSet = {};
    docs.forEach(function (d) {
      (d.tags || []).forEach(function (t) { tagSet[t] = true; });
    });
    var tags = Object.keys(tagSet).sort();

    var html = '<div class="filter-tag-area">';
    tags.forEach(function (t) {
      html += '<button class="filter-tag-chip" data-tag="' + t + '">' + t + '</button>';
    });
    html += '</div>';
    filterArea.innerHTML = html;

    var chips = filterArea.querySelectorAll(".filter-tag-chip");
    for (var i = 0; i < chips.length; i++) {
      // restore highlight for previously selected tags
      if (selectedTags.indexOf(chips[i].getAttribute("data-tag")) !== -1) {
        chips[i].classList.add("active");
      }
      chips[i].addEventListener("click", function () {
        var tag = this.getAttribute("data-tag");
        var idx = selectedTags.indexOf(tag);
        if (idx === -1) {
          selectedTags.push(tag);
        } else {
          selectedTags.splice(idx, 1);
        }
        this.classList.toggle("active", selectedTags.indexOf(tag) !== -1);
        saveSelectedTags();
        render();
      });
    }
  }

  function sortDocs(arr) {
    return arr.slice().sort(function (a, b) {
      var da = currentSort === "arxiv" ? (a.date_arxiv || "") : (a.date_added || "");
      var db = currentSort === "arxiv" ? (b.date_arxiv || "") : (b.date_added || "");
      return da > db ? -1 : da < db ? 1 : 0;
    });
  }

  function paperItemHtml(d, linkBase, dateKey) {
    var url = linkBase + d.path.replace(/\.md$/, "/");
    var dateVal = d[dateKey] || "";
    var orgHtml = d.organizations ? '<span class="paper-org">' + d.organizations + '</span>' : '';
    return '<div class="paper-item">'
      + '<div class="paper-item-header">'
      + '<a href="' + url + '">' + d.title + '</a>'
      + '<span class="paper-date">' + dateVal + '</span>'
      + '</div>'
      + orgHtml
      + '<a class="paper-summary" href="' + url + '">' + (d.summary || "") + '</a>'
      + '</div>';
  }

  function render() {
    var linkBase = base + "/";
    var dateKey = currentSort === "arxiv" ? "date_arxiv" : "date_added";

    if (selectedTags.length > 0) {
      var filtered = docs.filter(function (d) {
        return selectedTags.every(function (t) {
          return (d.tags || []).indexOf(t) !== -1;
        });
      });
      var sorted = sortDocs(filtered);
      var html = '<p class="filter-count">' + sorted.length + ' 篇匹配</p>';
      html += '<div class="paper-list">';
      sorted.forEach(function (d) { html += paperItemHtml(d, linkBase, dateKey); });
      html += '</div>';
      container.innerHTML = html;
      return;
    }

    var groups = {};
    var catOrder = [];
    docs.forEach(function (d) {
      var cat = d.category || "其他";
      if (!groups[cat]) { groups[cat] = []; catOrder.push(cat); }
      groups[cat].push(d);
    });
    catOrder.forEach(function (cat) { groups[cat] = sortDocs(groups[cat]); });

    var html = "";
    catOrder.forEach(function (cat) {
      html += '<h2>' + cat + '</h2><div class="paper-list">';
      groups[cat].forEach(function (d) { html += paperItemHtml(d, linkBase, dateKey); });
      html += '</div>';
    });
    container.innerHTML = html;
  }
});
