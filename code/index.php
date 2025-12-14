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
document.getElementById("promptForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const query = document.getElementById("prompt").value.trim();
    const lang  = document.getElementById("lang").value;
    const mode  = document.getElementById("mode").value;

    if (!query) {
        alert("Please enter a problem description.");
        return;
    }

    let endpoint = (mode === "solve")
        ? "http://127.0.0.1:5001/solve"
        : "http://127.0.0.1:5001/search";

    try {
        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ query: query, lang: lang })
        });

        const data = await response.json();

        let formatted = "";

        if (mode === "solve") {
            formatted += "=== FINAL ANSWER ===\n\n";
            formatted += data.llm_answer + "\n\n";
            formatted += "=== SIMILAR CASES ===\n";

            data.similar_cases.forEach((c, i) => {
                formatted += `\n#${i + 1} (${c.distance.toFixed(3)})\n`;
                formatted += `Problem: ${c.problem}\n`;
                formatted += `Symptoms: ${c.symptoms}\n`;
                formatted += `Solution: ${c.solution}\n`;
            });

            formatted += `\nGenerated in: ${data.response_time}s\n`;
        } else {
            formatted += "=== SEARCH RESULTS ===\n";

            data.forEach((c, i) => {
                formatted += `\n#${i + 1} (${c.distance.toFixed(3)})\n`;
                formatted += `Problem: ${c.problem}\n`;
                formatted += `Symptoms: ${c.symptoms}\n`;
                formatted += `Solution: ${c.solution}\n`;
            });
        }

        document.getElementById("response").innerText = formatted;

    } catch (err) {
        document.getElementById("response").innerText =
            "Connection error or backend is not running.";
    }
});
</script>

</body>
</html>
