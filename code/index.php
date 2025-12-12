<?php $config = include __DIR__ . "/lmhelper_config.php"; ?>
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>LM Helper</title>
    <link rel="stylesheet" href="style.css">
</head>
<body>

<h1>LM Helper – IT Outsourcing AI</h1>

<form id="promptForm">
    <textarea id="prompt" placeholder="Zadej problém nebo dotaz..." required></textarea>
    <button type="submit">Odeslat</button>
</form>

<div id="response"></div>

<script>
document.getElementById("promptForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const query = document.getElementById("prompt").value.trim();

    if (!query) {
        alert("Vyplň dotaz!");
        return;
    }

    try {
        const response = await fetch("http://127.0.0.1:5001/solve", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            mode: "cors",
            body: JSON.stringify({ query: query })
        });

        if (!response.ok) {
            document.getElementById("response").innerText =
                "Chyba serveru: " + response.status;
            return;
        }

        const data = await response.json();
        document.getElementById("response").innerText = JSON.stringify(data, null, 2);

    } catch (err) {
        document.getElementById("response").innerText = "Chyba spojení: " + err;
    }
});
</script>

</body>
</html>
