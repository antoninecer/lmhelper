<?php

header("Content-Type: application/json; charset=utf-8");

$config = include __DIR__ . "/lmhelper_config.php";

if (!isset($_POST["prompt"])) {
    echo json_encode(["error" => "Missing prompt"]);
    exit;
}

$prompt = trim($_POST["prompt"]);

$payload = [
    "model" => $config["model"],
    "messages" => [
        ["role" => "user", "content" => $prompt]
    ]
];

$ch = curl_init($config["lmstudio_api"] . "/v1/chat/completions");
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, ["Content-Type: application/json"]);
curl_setopt($ch, CURLOPT_POSTFIELDS, json_encode($payload));

$response = curl_exec($ch);
curl_close($ch);

echo $response;
