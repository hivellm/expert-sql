# Test script to compare expert-sql v0.2.1 vs v0.3.0
$cli = "F:\Node\hivellm\expert\cli\target\release\expert-cli.exe"

$testCases = @(
    @{prompt="Liste todos os usuarios"; category="SELECT"},
    @{prompt="Mostre produtos com preco menor que 100"; category="WHERE"},
    @{prompt="Quantos pedidos foram cancelados?"; category="COUNT"},
    @{prompt="Liste clientes e seus pedidos"; category="JOIN"},
    @{prompt="Qual a receita total por produto?"; category="GROUP BY"},
    @{prompt="Contas com total de transacoes acima de 10000"; category="HAVING"},
    @{prompt="Autores e seus livros, incluindo autores sem livros"; category="LEFT JOIN"},
    @{prompt="Classifique produtos como 'Caro' se preco > 1000, senao 'Barato'"; category="CASE"}
)

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "COMPARACAO: expert-sql v0.2.1 vs v0.3.0" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$results_v021 = @()
$results_v030 = @()

foreach ($test in $testCases) {
    Write-Host "`n[TESTE] $($test.category): $($test.prompt)" -ForegroundColor Yellow
    
    # Test v0.2.1
    Write-Host "  v0.2.1: " -NoNewline -ForegroundColor Gray
    $output_v021 = & $cli chat --experts sql@0.2.1 --prompt $test.prompt --max-tokens 100 2>&1
    $sql_v021 = ($output_v021 | Select-String -Pattern "SELECT.*" | Select-Object -First 1).Line
    if ($sql_v021) {
        Write-Host $sql_v021 -ForegroundColor Green
        $results_v021 += @{category=$test.category; sql=$sql_v021; valid=$true}
    } else {
        Write-Host "Nenhum SQL gerado" -ForegroundColor Red
        $results_v021 += @{category=$test.category; sql=""; valid=$false}
    }
    
    # Test v0.3.0
    Write-Host "  v0.3.0: " -NoNewline -ForegroundColor Gray
    $output_v030 = & $cli chat --experts sql@0.3.0 --prompt $test.prompt --max-tokens 100 2>&1
    $sql_v030 = ($output_v030 | Select-String -Pattern "SELECT.*" | Select-Object -First 1).Line
    if ($sql_v030) {
        Write-Host $sql_v030 -ForegroundColor Green
        $results_v030 += @{category=$test.category; sql=$sql_v030; valid=$true}
    } else {
        Write-Host "Nenhum SQL gerado" -ForegroundColor Red
        $results_v030 += @{category=$test.category; sql=""; valid=$false}
    }
    
    Start-Sleep -Seconds 2
}

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "RESUMO" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

$valid_v021 = ($results_v021 | Where-Object {$_.valid}).Count
$valid_v030 = ($results_v030 | Where-Object {$_.valid}).Count

Write-Host "v0.2.1: $valid_v021/$($testCases.Count) queries validas" -ForegroundColor $(if ($valid_v021 -eq $testCases.Count) {"Green"} else {"Yellow"})
Write-Host "v0.3.0: $valid_v030/$($testCases.Count) queries validas" -ForegroundColor $(if ($valid_v030 -eq $testCases.Count) {"Green"} else {"Yellow"})

Write-Host "`nDIFERENCAS:" -ForegroundColor Cyan
for ($i = 0; $i -lt $testCases.Count; $i++) {
    if ($results_v021[$i].valid -ne $results_v030[$i].valid) {
        Write-Host "  [$($testCases[$i].category)] v0.2.1: $($results_v021[$i].valid) | v0.3.0: $($results_v030[$i].valid)" -ForegroundColor Yellow
    }
    if ($results_v021[$i].sql -ne $results_v030[$i].sql -and $results_v021[$i].valid -and $results_v030[$i].valid) {
        Write-Host "  [$($testCases[$i].category)] Diferentes:" -ForegroundColor Gray
        Write-Host "    v0.2.1: $($results_v021[$i].sql)" -ForegroundColor Gray
        Write-Host "    v0.3.0: $($results_v030[$i].sql)" -ForegroundColor Gray
    }
}

