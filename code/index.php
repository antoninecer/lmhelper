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

<form method="POST" action="call_llm.php" id="promptForm">
    <textarea name="prompt" placeholder="Zadej problém nebo dotaz..." required></textarea>
    <button type="submit">Odeslat</button>
</form>

<div id="response"></div>

<script>
document.getElementById("promptForm").addEventListener("submit", async function(e) {
    e.preventDefault();

    const formData = new FormData(this);

    const response = await fetch("call_llm.php", {
        method: "POST",
        body: formData
    });

    const data = await response.json();
    document.getElementById("response").innerText = JSON.stringify(data, null, 2);
});
</script>

</body>
</html>
