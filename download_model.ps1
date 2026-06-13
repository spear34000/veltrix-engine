$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:NO_PROXY = "github.com,huggingface.co,raw.githubusercontent.com"

$url = "https://huggingface.co/llama.cpp/models/resolve/main/tinyllamas-1.1b-q4_0.gguf"
$output = "model.gguf"

Write-Host "Downloading from $url..."
try {
    Invoke-WebRequest -Uri $url -OutFile $output -TimeoutSec 300
    Write-Host "Download complete: $output"
    $file = Get-Item $output
    Write-Host "File size: $($file.Length) bytes"
} catch {
    Write-Error "Download failed: $($_.Exception.Message)"
    exit 1
}