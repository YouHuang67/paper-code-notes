document.addEventListener("DOMContentLoaded", function () {
  var filter = document.getElementById("daily-filter");
  if (!filter) return;

  var groups = Array.prototype.slice.call(filter.querySelectorAll(".daily-filter-group"));
  var cards = Array.prototype.slice.call(document.querySelectorAll(".daily-paper-card"));
  var sections = Array.prototype.slice.call(document.querySelectorAll(".md-content h2"));
  var count = document.getElementById("daily-filter-count");
  var selections = {};

  groups.forEach(function (group) {
    selections[group.getAttribute("data-filter-group")] = [];
    group.querySelectorAll(".daily-filter-chip").forEach(function (chip) {
      var label = document.createElement("span");
      label.className = "daily-filter-chip-label";
      label.textContent = chip.textContent;
      chip.textContent = "";
      chip.appendChild(label);
      var chipCount = document.createElement("span");
      chipCount.className = "daily-filter-chip-count";
      chipCount.setAttribute("aria-label", "匹配数量");
      chip.appendChild(chipCount);
    });
  });

  function matchesOtherGroups(card, currentKey) {
    return groups.every(function (group) {
      var key = group.getAttribute("data-filter-group");
      if (key === currentKey) return true;
      var selected = selections[key];
      return selected.length === 0 || selected.indexOf(card.getAttribute("data-" + key)) !== -1;
    });
  }

  function countFor(group, value) {
    var key = group.getAttribute("data-filter-group");
    var selected = selections[key];
    var currentCards = cards.filter(function (card) {
      return groups.every(function (otherGroup) {
        var otherKey = otherGroup.getAttribute("data-filter-group");
        var otherSelected = selections[otherKey];
        return otherSelected.length === 0 || otherSelected.indexOf(card.getAttribute("data-" + otherKey)) !== -1;
      });
    });
    if (selected.length > 0) {
      if (value === "all") return currentCards.length;
      if (selected.indexOf(value) === -1) return 0;
      return currentCards.filter(function (card) {
        return card.getAttribute("data-" + key) === value;
      }).length;
    }
    return cards.filter(function (card) {
      return matchesOtherGroups(card, key) && (value === "all" || card.getAttribute("data-" + key) === value);
    }).length;
  }

  function renderChipCounts() {
    groups.forEach(function (group) {
      group.querySelectorAll(".daily-filter-chip").forEach(function (chip) {
        var value = chip.getAttribute("data-filter-value");
        var countNode = chip.querySelector(".daily-filter-chip-count");
        if (countNode) countNode.textContent = "(" + countFor(group, value) + ")";
      });
    });
  }

  function render() {
    var visible = 0;
    cards.forEach(function (card) {
      var show = groups.every(function (group) {
        var key = group.getAttribute("data-filter-group");
        var selected = selections[key];
        return selected.length === 0 || selected.indexOf(card.getAttribute("data-" + key)) !== -1;
      });
      card.hidden = !show;
      if (show) visible += 1;
    });
    sections.forEach(function (heading) {
      if (heading.textContent.indexOf("当日概览") !== -1) return;
      var hasVisibleCard = false;
      var sibling = heading.nextElementSibling;
      while (sibling && sibling.tagName !== "H2") {
        if ((sibling.matches && sibling.matches(".daily-paper-card:not([hidden])")) ||
            (sibling.querySelector && sibling.querySelector(".daily-paper-card:not([hidden])"))) {
          hasVisibleCard = true;
          break;
        }
        sibling = sibling.nextElementSibling;
      }
      heading.hidden = !hasVisibleCard;
    });
    if (count) count.textContent = visible + " 篇匹配";
    renderChipCounts();
  }

  function updateGroup(group, value) {
    var key = group.getAttribute("data-filter-group");
    var selected = selections[key];
    if (value === "all") {
      selections[key] = [];
    } else if (selected.indexOf(value) === -1) {
      selected.push(value);
    } else {
      selected.splice(selected.indexOf(value), 1);
    }
    group.querySelectorAll(".daily-filter-chip").forEach(function (chip) {
      var chipValue = chip.getAttribute("data-filter-value");
      chip.classList.toggle("active", chipValue === "all" ? selections[key].length === 0 : selections[key].indexOf(chipValue) !== -1);
      chip.setAttribute("aria-pressed", chip.classList.contains("active") ? "true" : "false");
    });
    render();
  }

  groups.forEach(function (group) {
    group.querySelectorAll(".daily-filter-chip").forEach(function (chip) {
      chip.addEventListener("click", function () {
        updateGroup(group, chip.getAttribute("data-filter-value"));
      });
    });
  });

  var reset = filter.querySelector(".daily-filter-reset");
  if (reset) reset.addEventListener("click", function () {
    groups.forEach(function (group) { selections[group.getAttribute("data-filter-group")] = []; });
    groups.forEach(function (group) {
      group.querySelectorAll(".daily-filter-chip").forEach(function (chip) {
        var active = chip.getAttribute("data-filter-value") === "all";
        chip.classList.toggle("active", active);
        chip.setAttribute("aria-pressed", active ? "true" : "false");
      });
    });
    render();
  });

  var topButton = document.querySelector(".daily-to-top");
  if (topButton) topButton.addEventListener("click", function () {
    window.scrollTo({top: 0, behavior: "smooth"});
  });
  render();
});
