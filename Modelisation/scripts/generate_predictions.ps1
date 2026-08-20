param(
    [int]$Count = 50,
    [string]$ApiUrl = "http://127.0.0.1:8001/predict",
    [string]$ApiKey = "trustpilot-secret-key"
)

$headers = @{
    "Content-Type" = "application/json"
    "X-API-Key" = $ApiKey
}

$texts = @(
    "This product is amazing and I love it",
    "Excellent service, I am very satisfied",
    "Great experience, fast delivery and good quality",
    "I recommend this company, everything was perfect",
    "Very happy with my purchase",
    "This is terrible, I am very disappointed",
    "Bad experience, the product arrived broken",
    "Worst service ever, I want a refund",
    "I am not satisfied with this order",
    "The delivery was late and the support was useless",
    "Very poor quality, I will never buy again",
    "Customer service was awful"
)

for ($i = 1; $i -le $Count; $i++) {
    $text = Get-Random -InputObject $texts

    $body = @{
        text = $text
    } | ConvertTo-Json

    try {
        $response = Invoke-RestMethod `
            -Method POST `
            -Uri $ApiUrl `
            -Headers $headers `
            -Body $body

        Write-Host "[$i/$Count] prediction=$($response.prediction) label=$($response.label)"
    }
    catch {
        Write-Host "[$i/$Count] Error: $($_.Exception.Message)" -ForegroundColor Red
    }

    Start-Sleep -Milliseconds 300
}