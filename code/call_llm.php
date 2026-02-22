<?php
header("Content-Type: application/json");

$api_url = "http://127.0.0.1:5001/search";

if (!isset($_POST["prompt"])) {
    echo json_encode(["error" => "Missing prompt"]);
    exit;
}

$payload = json_encode(["query" => $_POST["prompt"]]);

$ch = curl_init($api_url);
curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
curl_setopt($ch, CURLOPT_POST, true);
curl_setopt($ch, CURLOPT_HTTPHEADER, [
    "Content-Type: application/json",
    "Content-Length: " . strlen($payload)
]);
curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);

$response = curl_exec($ch);
curl_close($ch);

echo $response;
