document.addEventListener("DOMContentLoaded", function () {
  var input = document.getElementById("tag-search-input");
  var results = document.getElementById("tag-search-results");
  var sortControls = document.getElementById("tag-sort-controls");
  if (!input || !results) return;

  // Inject search hint below input
  var hint = document.createElement("div");
  hint.className = "tag-search-hint";
  hint.innerHTML =
    "支持布尔检索：<code>AND</code>（与）&nbsp;<code>OR</code>（或）&nbsp;括号控制优先级" +
    " &nbsp;·&nbsp; 示例：<code>sparse AND diffusion</code> &nbsp;·&nbsp;" +
    " <code>(video OR image) AND attention</code>";
  input.parentNode.insertBefore(hint, input.nextSibling);

  var docs = [];
  var lastResults = [];
  var currentSort = "arxiv";

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

  // Sync button active state with currentSort default
  var buttons = sortControls ? sortControls.querySelectorAll(".sort-btn") : [];
  for (var i = 0; i < buttons.length; i++) {
    (function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-sort") === currentSort);
      btn.addEventListener("click", function () {
        for (var j = 0; j < buttons.length; j++) buttons[j].classList.remove("active");
        this.classList.add("active");
        currentSort = this.getAttribute("data-sort");
        renderResults(lastResults);
      });
    })(buttons[i]);
  }

  // ── Boolean query parser ──────────────────────────────────────────────────
  //
  // Grammar:
  //   Expr   = Or
  //   Or     = And ( OR And )*
  //   And    = Factor ( [AND] Factor )*   (AND keyword or implicit adjacency)
  //   Factor = LPAREN Expr RPAREN | TERM
  //
  // A TERM matches a doc tag if the tag (lowercase) contains the term.

  function tokenize(query) {
    var tokens = [];
    var i = 0;
    while (i < query.length) {
      if (/\s/.test(query[i])) { i++; continue; }
      if (query[i] === "(") { tokens.push({ type: "LPAREN" }); i++; continue; }
      if (query[i] === ")") { tokens.push({ type: "RPAREN" }); i++; continue; }
      var j = i;
      while (j < query.length && !/\s/.test(query[j]) && query[j] !== "(" && query[j] !== ")") j++;
      var word = query.slice(i, j);
      var up = word.toUpperCase();
      if (up === "AND") tokens.push({ type: "AND" });
      else if (up === "OR") tokens.push({ type: "OR" });
      else tokens.push({ type: "TERM", value: word });
      i = j;
    }
    return tokens;
  }

  function buildAST(tokens) {
    var pos = 0;

    function peek() { return tokens[pos]; }
    function consume() { return tokens[pos++]; }

    function parseExpr() { return parseOr(); }

    function parseOr() {
      var left = parseAnd();
      while (peek() && peek().type === "OR") {
        consume();
        var right = parseAnd();
        left = { type: "OR", left: left, right: right };
      }
      return left;
    }

    function parseAnd() {
      var left = parseFactor();
      while (peek() && peek().type !== "OR" && peek().type !== "RPAREN") {
        if (peek().type === "AND") {
          consume();
        } else if (peek().type !== "TERM" && peek().type !== "LPAREN") {
          break;
        }
        var right = parseFactor();
        if (!right) break;
        left = { type: "AND", left: left, right: right };
      }
      return left;
    }

    function parseFactor() {
      var t = peek();
      if (!t) return null;
      if (t.type === "LPAREN") {
        consume();
        var expr = parseExpr();
        if (peek() && peek().type === "RPAREN") consume();
        return expr;
      }
      if (t.type === "TERM") {
        consume();
        return { type: "TERM", value: t.value };
      }
      return null;
    }

    return parseExpr();
  }

  function evalAST(node, tags) {
    if (!node) return { match: false, matchedTags: [] };

    if (node.type === "TERM") {
      var term = node.value;
      var matched = tags.filter(function (tag) {
        return tag.toLowerCase().indexOf(term) !== -1;
      });
      return { match: matched.length > 0, matchedTags: matched };
    }

    if (node.type === "AND") {
      var l = evalAST(node.left, tags);
      var r = evalAST(node.right, tags);
      if (l.match && r.match) {
        var combined = l.matchedTags.slice();
        r.matchedTags.forEach(function (t) { if (combined.indexOf(t) === -1) combined.push(t); });
        return { match: true, matchedTags: combined };
      }
      return { match: false, matchedTags: [] };
    }

    if (node.type === "OR") {
      var l = evalAST(node.left, tags);
      var r = evalAST(node.right, tags);
      var combined = l.matchedTags.slice();
      r.matchedTags.forEach(function (t) { if (combined.indexOf(t) === -1) combined.push(t); });
      return { match: l.match || r.match, matchedTags: combined };
    }

    return { match: false, matchedTags: [] };
  }

  // ── Search handler ────────────────────────────────────────────────────────

  input.addEventListener("input", function () {
    var raw = input.value.trim();
    if (!raw) {
      results.innerHTML = "";
      if (sortControls) sortControls.style.display = "none";
      lastResults = [];
      return;
    }

    var tokens = tokenize(raw.toLowerCase());
    var ast = buildAST(tokens);

    var matched = [];
    docs.forEach(function (doc) {
      var ev = evalAST(ast, doc.tags);
      if (ev.match) matched.push({ doc: doc, matchedTags: ev.matchedTags });
    });

    lastResults = matched;

    if (matched.length === 0) {
      results.innerHTML = '<p class="tag-search-empty">未找到匹配的文档</p>';
      if (sortControls) sortControls.style.display = "none";
      return;
    }

    if (sortControls) sortControls.style.display = "flex";
    renderResults(lastResults);
  });

  // ── Render ────────────────────────────────────────────────────────────────

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
    } else {
      // relevance: more matched tags first, then arxiv date desc
      sorted.sort(function (a, b) {
        var diff = b.matchedTags.length - a.matchedTags.length;
        if (diff !== 0) return diff;
        var da = a.doc.date_arxiv || "";
        var db = b.doc.date_arxiv || "";
        return da > db ? -1 : da < db ? 1 : 0;
      });
    }

    var linkBase = siteBase + "/";
    var html = '<p class="tag-search-count">' + sorted.length + " 个符合条件的结果</p>";
    sorted.forEach(function (item) {
      var doc = item.doc;
      var tagsHtml = doc.tags.map(function (t) {
        var cls = item.matchedTags.indexOf(t) !== -1 ? "tag-chip tag-chip-match" : "tag-chip";
        return '<span class="' + cls + '">' + t + "</span>";
      }).join(" ");

      var dateHtml = "";
      if (doc.date_added || doc.date_arxiv) {
        dateHtml = '<div class="tag-search-dates">';
        if (doc.date_added) dateHtml += "<span>添加: " + doc.date_added + "</span>";
        if (doc.date_arxiv) dateHtml += "<span>arXiv: " + doc.date_arxiv + "</span>";
        dateHtml += "</div>";
      }

      html += '<div class="tag-search-card">'
        + '<a class="tag-search-title" href="' + linkBase + doc.path.replace(/\.md$/, "/") + '">' + doc.title + "</a>"
        + '<span class="tag-search-type">' + doc.type + "</span>"
        + '<div class="tag-search-tags">' + tagsHtml + "</div>"
        + dateHtml
        + '<p class="tag-search-summary">' + doc.summary + "</p>"
        + "</div>";
    });

    results.innerHTML = html;
  }
});
