(function () {
  "use strict";

  // Render every Plotly figure embedded as JSON next to its container.
  var plotConfig = {
    responsive: true,
    displaylogo: false,
    modeBarButtonsToRemove: ["lasso2d", "select2d"],
    toImageButtonOptions: { format: "png", scale: 2 }
  };

  function renderCharts() {
    if (!window.Plotly) return;
    document.querySelectorAll(".chart").forEach(function (el) {
      var source = document.getElementById(el.id + "-data");
      if (!source) return;
      try {
        var figure = JSON.parse(source.textContent);
        window.Plotly.newPlot(el, figure.data, figure.layout, plotConfig);
      } catch (error) {
        el.textContent = "Não foi possível desenhar o gráfico.";
        console.error(error);
      }
    });
  }

  // Selects inside a control form update the page immediately.
  function autoSubmit() {
    document.querySelectorAll("form.controls").forEach(function (form) {
      form.querySelectorAll("select").forEach(function (select) {
        select.addEventListener("change", function () {
          form.submit();
        });
      });
    });
  }

  // Mobile navigation drawer.
  function menu() {
    var toggle = document.querySelector(".menu-toggle");
    if (!toggle) return;
    toggle.addEventListener("click", function () {
      var open = document.body.classList.toggle("nav-open");
      toggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    renderCharts();
    autoSubmit();
    menu();
  });
})();
