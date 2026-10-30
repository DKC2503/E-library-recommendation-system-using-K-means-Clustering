const $ = s => document.querySelector(s);

const esc = s =>
  String(s ?? "").replace(/[&<>"']/g, c => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[c]));

const fmt = n =>
  Number(n ?? 0).toLocaleString(undefined, {
    maximumFractionDigits: 3
  });

// Update the cluster count display.
$("#k").addEventListener("input", () => {
  $("#kout").textContent = $("#k").value;
});

// Update the rating threshold display.
$("#threshold").addEventListener("input", () => {
  $("#tout").textContent =
    Number($("#threshold").value).toFixed(1);
});

// Display selected CSV filenames.
for (const [name, id] of [
  ["books_file", "#bookname"],
  ["ratings_file", "#ratingname"]
]) {
  document.querySelector(`[name=${name}]`)
    .addEventListener("change", e => {
      $(id).textContent =
        e.target.files[0]?.name || "Choose CSV";
    });
}

// Safely read JSON API responses.
async function readApiResponse(response) {
  const contentType =
    response.headers.get("content-type") || "";

  if (contentType.includes("application/json")) {
    return await response.json();
  }

  const responseText = await response.text();

  throw new Error(
    `Server returned HTTP ${response.status}: ` +
    responseText.slice(0, 250)
  );
}

// API and MongoDB health status.
async function health() {
  try {
    const response = await fetch("/api/health");
    const data = await readApiResponse(response);

    if (!response.ok) {
      throw new Error(
        data.detail || `Health check failed (${response.status}).`
      );
    }

    $("#health").textContent =
      data.mongodb?.connected
        ? "API + MongoDB connected"
        : data.mongodb?.configured
          ? "API online · MongoDB offline"
          : "API online · MongoDB not configured";

  } catch (error) {
    $("#health").textContent = "API unavailable";
  }
}

// Display clustering analysis results.
function showResults(d) {
  $("#results").classList.remove("hidden");

  const m = d.metrics || {};
  const p = d.parameters || {};

  const data = [
    [
      "Usable readers",
      fmt(m.usable_readers),
      "Readers included in the clustering matrix after filtering."
    ],
    [
      "Feature books",
      fmt(m.usable_books),
      "Most active books used as the model's input features."
    ],
    [
      "Clusters (K)",
      fmt(p.n_clusters),
      "Number of reader groups requested for this run."
    ],
    [
      "Inertia (J)",
      fmt(m.inertia),
      "Sum of squared distances to cluster centroids. Compare only with the same data and feature pipeline."
    ],
    [
      "Iterations",
      fmt(m.iterations),
      "Iterations performed by the selected clustering estimator."
    ],
    [
      "SVD variance",
      `${((m.explained_variance_ratio || 0) * 100).toFixed(2)}%`,
      "Variance represented by the selected latent components; this is not recommendation accuracy."
    ],
    [
      "Runtime",
      `${fmt(m.duration_seconds)} s`,
      "Measured analysis duration reported by the backend."
    ],
    [
      "Sparse ratings",
      fmt(m.sparse_nonzeros),
      "Non-zero entries in the sparse reader-book matrix."
    ]
  ];

  $("#metrics").innerHTML = data.map(x => `
    <div class="metric">
      <span>${esc(x[0])}</span>
      <strong>${esc(x[1])}</strong>
      <small>${esc(x[2])}</small>
    </div>
  `).join("");

  $("#runExplanation").innerHTML = `
    <b>How to read this run</b>
    <p>
      Readers are represented by their rating behaviour, projected
      to a smaller latent space with TruncatedSVD, then grouped by
      ${esc(p.algorithm || "the selected K-Means estimator")}
      (K=${esc(p.n_clusters)}).
      Inertia measures within-cluster compactness; it is not an
      accuracy score. SVD variance describes the projection, not
      whether recommendations are correct.
    </p>
    <p>
      <b>Configuration:</b>
      rating threshold ${esc(p.rating_threshold)} / 5 ·
      random seed ${esc(p.random_state)} ·
      algorithm ${esc(p.algorithm)}.
    </p>
  `;

  $("#clusters").innerHTML = (d.clusters || []).map(c => `
    <article class="cluster">
      <div class="clusterline">
        <span>Cluster ${esc(c.cluster_id)}</span>
        <span>${fmt(c.percentage)}%</span>
      </div>

      <div class="track">
        <span style="width:${
          Math.max(
            0,
            Math.min(100, Number(c.percentage) || 0)
          )
        }%"></span>
      </div>

      <p class="clusterstat">
        <b>${fmt(c.reader_count)}</b> readers ·
        mean rating <b>${fmt(c.average_rating)} / 5</b>
      </p>

      <p class="explain">
        This group contains ${fmt(c.reader_count)} of the usable
        readers. Books below are highly rated among ratings
        associated with this cluster; the rating count is shown
        to provide context.
      </p>

      <div class="bookchips">
        ${
          (c.top_books || []).map(b => `
            <span
              class="chip"
              title="Mean rating ${esc(b.average_rating)} from ${esc(b.rating_count)} ratings"
            >
              ${esc(b.title)}
              <b>${esc(b.average_rating)}/5</b>
              <small>${fmt(b.rating_count)} ratings</small>
            </span>
          `).join("")
          || "<span class='subtle'>No book summary available.</span>"
        }
      </div>
    </article>
  `).join("");

  history();
}

// Run the clustering analysis.
// Improved handling for HTTP 413, HTML error pages and invalid JSON.
$("#analysis").addEventListener("submit", async e => {
  e.preventDefault();

  const button = e.target.querySelector("button");
  button.disabled = true;

  $("#progress").classList.remove("hidden");
  $("#bar").style.width = "15%";
  $("#message").style.color = "";
  $("#message").textContent =
    "Processing may take some time for large datasets.";

  const timer = setInterval(() => {
    $("#bar").style.width =
      Math.min(
        90,
        parseInt($("#bar").style.width, 10) + 7
      ) + "%";

    $("#progressText").textContent =
      "Validating, vectorizing and clustering readers...";
  }, 1600);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      body: new FormData(e.target)
    });

    const data = await readApiResponse(response);

    if (!response.ok) {
      throw new Error(
        data.detail ||
        `Analysis failed with HTTP ${response.status}.`
      );
    }

    $("#bar").style.width = "100%";
    $("#progressText").textContent = "Analysis completed.";
    $("#message").style.color = "";

    $("#message").textContent =
      `Completed in ${data.metrics.duration_seconds}s · ` +
      `Run ${data.run_id}`;

    showResults(data);

  } catch (error) {
    $("#message").style.color = "#a9472c";
    $("#message").textContent =
      error.message || "An unexpected error occurred.";

  } finally {
    clearInterval(timer);
    button.disabled = false;

    health();

    setTimeout(() => {
      $("#progress").classList.add("hidden");
    }, 3000);
  }
});

// Get personalised recommendations for a reader.
$("#recform").addEventListener("submit", async e => {
  e.preventDefault();

  const reader = $("#reader").value.trim();

  $("#recmeta").textContent = "Finding recommendations...";
  $("#recommendations").innerHTML = "";

  try {
    if (!reader) {
      throw new Error("Please enter a reader ID.");
    }

    const response = await fetch(
      `/api/recommend/${encodeURIComponent(reader)}` +
      `?limit=${encodeURIComponent($("#limit").value)}`
    );

    const data = await readApiResponse(response);

    if (!response.ok) {
      throw new Error(
        data.detail ||
        `Recommendation request failed (${response.status}).`
      );
    }

    $("#recmeta").textContent = data.is_cold_start
      ? "Cold-start fallback: this reader was not found in the analysed ratings, so community favourites are shown."
      : `Reader ${data.reader_id} belongs to Cluster ${data.cluster_id}. Books already rated by this reader are excluded.`;

    $("#recommendations").innerHTML =
      (data.recommendations || []).map((b, i) => `
        <article class="rec">
          <span class="num">
            ${String(i + 1).padStart(2, "0")}
          </span>

          <div>
            <h3>${esc(b.title)}</h3>

            <p>
              ${esc(b.authors)} ·
              ${fmt(b.rating_count)} relevant ratings
            </p>

            <div class="score">
              Estimated ranking score: ${fmt(b.score)} / 5
            </div>

            <p>${esc(b.reason)}</p>

            <p class="fineprint">
              This is a ranking estimate based on historical
              ratings, not a guaranteed personal rating.
            </p>
          </div>
        </article>
      `).join("")
      || `
        <p class="empty">
          No eligible recommendations were found for this reader.
          Try another reader ID or increase the result count.
        </p>
      `;

  } catch (error) {
    $("#recmeta").textContent =
      error.message || "Could not load recommendations.";

    $("#recommendations").innerHTML = "";
  }
});

// Load analysis history.
async function history() {
  try {
    const response = await fetch("/api/history");
    const data = await readApiResponse(response);

    if (!response.ok) {
      throw new Error(
        data.detail ||
        `History request failed (${response.status}).`
      );
    }

    const runs = data.history || [];

    $("#history").innerHTML = runs.length
      ? runs.map(x => `
          <div class="historyrow">
            <b>${esc(x.run_id || "Analysis")}</b>
            <span>
              ${esc(x.created_at || "")} ·
              ${fmt(x.metrics?.usable_readers)} readers ·
              K=${x.parameters?.n_clusters ?? "—"}
            </span>
          </div>
        `).join("")
      : "No analysis history yet.";

  } catch (error) {
    $("#history").textContent = "History unavailable";
  }
}

// Refresh history manually.
$("#refresh").addEventListener("click", history);

// Initial page setup.
health();
history();