<?php $config = include __DIR__ . "/lmhelper_config.php"; ?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LM Helper – IT Outsourcing AI</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>

<h1>LM Helper – IT Outsourcing AI</h1>

<form id="promptForm">
    <label>Mode:</label>
    <select id="mode">
        <option value="solve">Solve (recommended)</option>
        <option value="search">Search (FAISS only)</option>
    </select>

    &nbsp;&nbsp;

    <label>Language:</label>
    <select id="lang">
        <option value="en" selected>English</option>
        <option value="cz">Czech</option>
        <option value="de">German</option>
        <option value="pl">Polish</option>
        <option value="it">Italian</option>
    </select>

    <br><br>

    <textarea
        id="prompt"
        placeholder="Describe the problem or incident..."
        required
    ></textarea>

    <button type="submit">Submit</button>
</form>

<div id="response"></div>

<script>
function fmtDist(d) {
    if (d === undefined || d === null || Number.isNaN(Number(d))) return "n/a";
    return Number(d).toFixed(3);
}

function renderSimilarCases(similarCases) {
    if (!Array.isArray(similarCases) || similarCases.length === 0) {
        return "";
    }

    let out = "\n=== SIMILAR CASES ===\n";
    similarCases.forEach((c, i) => {
        const id = (c && c.id !== undefined && c.id !== null) ? c.id : "n/a";
        const dist = fmtDist(c?.distance);
        const label = c?.distance_label ? `, ${c.distance_label}` : "";

        out += `\n#${i + 1} (ID ${id}, dist ${dist}${label})\n`;
        out += `Problem: ${c?.problem || ""}\n`;
        if (c?.symptoms) out += `Symptoms: ${c.symptoms}\n`;
        if (c?.solution) out += `Solution: ${c.solution}\n`;
    });

    return out;
}

document.getElementById("promptForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const query = document.getElementById("prompt").value.trim();
    const lang  = document.getElementById("lang").value;
    const mode  = document.getElementById("mode").value;

    if (!query) {
        alert("Please enter a problem description.");
        return;
    }

    const endpoint = (mode === "solve")
        ? "http://127.0.0.1:5001/solve"
        : "http://127.0.0.1:5001/search";

    const respBox = document.getElementById("response");
    respBox.innerText = "Working...";

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, lang: lang })
        });

        if (!response.ok) {
            const t = await response.text();
            respBox.innerText = `Backend error: HTTP ${response.status}\n${t}`;
            return;
        }

        const data = await response.json();
        let formatted = "";

        if (mode === "solve") {
            formatted += "=== FINAL ANSWER ===\n\n";
            formatted += (data?.llm_answer || "(no answer)") + "\n\n";

            formatted += renderSimilarCases(data?.similar_cases);

            const rt = (data?.response_time !== undefined) ? data.response_time : "n/a";
            const tt = (data?.total_time !== undefined) ? data.total_time : "n/a";
            formatted += `\nGenerated in: ${rt}s (total ${tt}s)\n`;

        } else {
            formatted += "=== SEARCH RESULTS ===\n";

            // /search vrací přímo pole
            const results = Array.isArray(data) ? data : [];
            if (results.length === 0) {
                formatted += "\n(no matches under threshold)\n";
            } else {
                results.forEach((c, i) => {
                    const id = (c && c.id !== undefined && c.id !== null) ? c.id : "n/a";
                    const dist = fmtDist(c?.distance);
                    const label = c?.distance_label ? `, ${c.distance_label}` : "";

                    formatted += `\n#${i + 1} (ID ${id}, dist ${dist}${label})\n`;
                    formatted += `Problem: ${c?.problem || ""}\n`;
                    if (c?.symptoms) formatted += `Symptoms: ${c.symptoms}\n`;
                    if (c?.solution) formatted += `Solution: ${c.solution}\n`;
                });
            }
        }

        respBox.innerText = formatted;

    } catch (err) {
        respBox.innerText = "Connection error or backend is not running.";
    }
});
</script>

</body>
</html>
