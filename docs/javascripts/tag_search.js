document.addEventListener("DOMContentLoaded", function () {
  var input = document.getElementById("tag-search-input");
  var results = document.getElementById("tag-search-results");
  if (!input || !results) return;

  var docs = [];

  // find base URL from this script's own src path
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

  input.addEventListener("input", function () {
    var query = input.value.trim().toLowerCase();
    if (!query) {
      results.innerHTML = "";
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
        var item = { doc: doc, matchedTags: matchedTags };
        if (isExact) {
          exact.push(item);
        } else {
          fuzzy.push(item);
        }
      }
    });

    var all = exact.concat(fuzzy);

    if (all.length === 0) {
      results.innerHTML = '<p class="tag-search-empty">未找到匹配的文档</p>';
      return;
    }

    var linkBase = siteBase + "/";

    var html = "";
    all.forEach(function (item) {
      var doc = item.doc;
      var tagsHtml = doc.tags.map(function (t) {
        var cls = item.matchedTags.indexOf(t) !== -1 ? "tag-chip tag-chip-match" : "tag-chip";
        return '<span class="' + cls + '">' + t + '</span>';
      }).join(" ");

      html += '<div class="tag-search-card">'
        + '<a class="tag-search-title" href="' + linkBase + doc.path + '">' + doc.title + '</a>'
        + '<span class="tag-search-type">' + doc.type + '</span>'
        + '<div class="tag-search-tags">' + tagsHtml + '</div>'
        + '<p class="tag-search-summary">' + doc.summary + '</p>'
        + '</div>';
    });

    results.innerHTML = html;
  });
});
