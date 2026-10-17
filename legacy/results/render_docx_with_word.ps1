param(
    [Parameter(Mandatory = $true)][string]$InputDocx,
    [Parameter(Mandatory = $true)][string]$OutputPdf,
    [Parameter(Mandatory = $true)][string]$StatusFile
)

$ErrorActionPreference = 'Stop'
$word = $null
$document = $null

try {
    Set-Content -LiteralPath $StatusFile -Value 'starting'
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($InputDocx, $false, $true)
    Set-Content -LiteralPath $StatusFile -Value 'opened'
    $document.ExportAsFixedFormat($OutputPdf, 17)
    Set-Content -LiteralPath $StatusFile -Value 'complete'
}
catch {
    Set-Content -LiteralPath $StatusFile -Value ("error: " + $_.Exception.Message)
    throw
}
finally {
    if ($document -ne $null) {
        $document.Close(0)
    }
    if ($word -ne $null) {
        $word.Quit()
    }
}
