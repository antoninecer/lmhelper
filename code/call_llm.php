<?php
$config = include __DIR__ . "/lmhelper_config.php";

$query = $_POST["prompt"] ?? "";
if (!$query) {
    echo json_encode(["error" => "No prompt given"]);
    exit;
}

// 1) Send query to RAG server
$ragResponse = file_get_contents("http://127.0.0.1:5001/search", false, stream_context_create([
    "http" => [
        "method"  => "POST",
        "header"  => "Content-Type: application/json",
        "content" => json_encode(["query" => $query])
    ]
]));

$ragData = json_decode($ragResponse, true);

// Build context text
$context = "";
foreach ($ragData as $item) {
    $context .= "Problem: {$item['problem']}\nSolution: {$item['solution']}\n\n";
}

// Final prompt
$finalPrompt = "Použij následující znalostní bázi k návrhu postupu:\n\n" .
               $context .
               "Uživatelský dotaz:\n$query\n\n" .
               "Navrhni nejpravděpodobnější řešení.";

// 2) Ask LLM via LM Studio
$payload = json_encode([
    "model" => $config["model"],
    "messages" => [
        ["role" => "user", "content" => $finalPrompt]
    ]
]);

$opts = [
    "http" => [
        "method" => "POST",
        "header" => "Content-Type: application/json",
        "content"=> $payload
    ]
];

$result = file_get_contents($config["api_url"], false, stream_context_create($opts));
echo $result;
