$body = Get-Content 'C:\Hirdal\WS_MLops\samples\predict_sample.json' -Raw
$response = Invoke-RestMethod -Uri 'http://localhost:8001/predict' -Method Post -ContentType 'application/json' -Body $body
$response | ConvertTo-Json -Depth 10
