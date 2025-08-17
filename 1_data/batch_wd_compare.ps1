# Batch WD Comparison Script
# Processes all experimental results in the results folder using wd_compare.py

param(
    [string]$ResultsDir = "results",
    [string]$OutputDir = "wd_comparisons",
    [switch]$ContinueOnError
)

# Initialize counters
$totalExperiments = 0
$successfulComparisons = 0
$failedComparisons = 0
$processedResults = @()

Write-Host "=== Batch WD Comparison Processing ===" -ForegroundColor Cyan
Write-Host "Results Directory: $((Get-Item $ResultsDir).FullName)" -ForegroundColor Yellow
Write-Host "Output Directory: $((New-Item -ItemType Directory -Path $OutputDir -Force).FullName)" -ForegroundColor Yellow
Write-Host ""

# Find all experimental result files
Write-Host "Scanning for experimental results..." -ForegroundColor Green
$resultFiles = Get-ChildItem -Path $ResultsDir -Directory | Where-Object { $_.Name -like "refusal_*" } | ForEach-Object {
    $resultFile = Join-Path $_.FullName "refusal_results.json"
    if (Test-Path $resultFile) {
        [PSCustomObject]@{
            ExperimentDir = $_.Name
            ResultFile = $resultFile
            FullPath = $_.FullName
        }
    }
} | Sort-Object ExperimentDir

$totalExperiments = $resultFiles.Count

if ($totalExperiments -eq 0) {
    Write-Host "No experimental result files found in $ResultsDir" -ForegroundColor Red
    exit 1
}

Write-Host "Found $totalExperiments experimental result files" -ForegroundColor Green
Write-Host ("=" * 80) -ForegroundColor Cyan

# Process each result file
for ($i = 0; $i -lt $resultFiles.Count; $i++) {
    $current = $i + 1
    $experiment = $resultFiles[$i]
    
    Write-Host "[$($current.ToString().PadLeft(2))/$totalExperiments] Processing: $($experiment.ExperimentDir)" -ForegroundColor Yellow
    
    try {
        # Run wd_compare.py
        $process = Start-Process -FilePath "python" -ArgumentList "1_data\wd_compare.py", "`"$($experiment.ResultFile)`"" -Wait -PassThru -NoNewWindow -RedirectStandardOutput "temp_output.txt" -RedirectStandardError "temp_error.txt"
        
        $stdout = Get-Content "temp_output.txt" -Raw -ErrorAction SilentlyContinue
        $stderr = Get-Content "temp_error.txt" -Raw -ErrorAction SilentlyContinue
        
        # Clean up temp files
        Remove-Item "temp_output.txt" -ErrorAction SilentlyContinue
        Remove-Item "temp_error.txt" -ErrorAction SilentlyContinue
        
        if ($process.ExitCode -eq 0) {
            # Parse output to find the generated filename
            $outputLine = $stdout -split "`n" | Where-Object { $_ -like "*Wrote * comparisons + summary to *" }
            if ($outputLine) {
                $outputFileName = ($outputLine -split " to ")[-1].Trim()
                
                # Move file to output directory if it exists
                if (Test-Path $outputFileName) {
                    $destFile = Join-Path $OutputDir (Split-Path $outputFileName -Leaf)
                    Move-Item $outputFileName $destFile -Force
                    
                    $result = [PSCustomObject]@{
                        ExperimentDirectory = $experiment.ExperimentDir
                        ResultFile = $experiment.ResultFile
                        Success = $true
                        OutputFile = Split-Path $destFile -Leaf
                        ErrorMessage = ""
                    }
                    
                    Write-Host "    Success: $(Split-Path $destFile -Leaf)" -ForegroundColor Green
                    $successfulComparisons++
                } else {
                    $result = [PSCustomObject]@{
                        ExperimentDirectory = $experiment.ExperimentDir
                        ResultFile = $experiment.ResultFile
                        Success = $false
                        OutputFile = ""
                        ErrorMessage = "Output file not found: $outputFileName"
                    }
                    
                    Write-Host "    Failed: Output file not found" -ForegroundColor Red
                    $failedComparisons++
                }
            } else {
                $result = [PSCustomObject]@{
                    ExperimentDirectory = $experiment.ExperimentDir
                    ResultFile = $experiment.ResultFile
                    Success = $false
                    OutputFile = ""
                    ErrorMessage = "Could not parse output filename from stdout"
                }
                
                Write-Host "    Failed: Could not parse output" -ForegroundColor Red
                $failedComparisons++
            }
        } else {
            $errorMsg = if ($stderr) { $stderr.Trim() } else { "Process failed with exit code $($process.ExitCode)" }
            
            $result = [PSCustomObject]@{
                ExperimentDirectory = $experiment.ExperimentDir
                ResultFile = $experiment.ResultFile
                Success = $false
                OutputFile = ""
                ErrorMessage = $errorMsg
            }
            
            Write-Host "    Failed: $errorMsg" -ForegroundColor Red
            $failedComparisons++
        }
        
        $processedResults += $result
        
    } catch {
        $result = [PSCustomObject]@{
            ExperimentDirectory = $experiment.ExperimentDir
            ResultFile = $experiment.ResultFile
            Success = $false
            OutputFile = ""
            ErrorMessage = $_.Exception.Message
        }
        
        $processedResults += $result
        Write-Host "    Exception: $($_.Exception.Message)" -ForegroundColor Red
        $failedComparisons++
        
        if (-not $ContinueOnError) {
            Write-Host "Stopping due to error. Use -ContinueOnError to continue processing." -ForegroundColor Red
            break
        }
    }
    
    Write-Host ""
}

# Generate summary statistics
Write-Host ("=" * 80) -ForegroundColor Cyan
Write-Host "BATCH PROCESSING COMPLETE" -ForegroundColor Cyan
Write-Host "Total experiments: $totalExperiments" -ForegroundColor White
Write-Host "Successful: $successfulComparisons" -ForegroundColor Green
Write-Host "Failed: $failedComparisons" -ForegroundColor Red

# Generate summary report
if ($successfulComparisons -gt 0) {
    Write-Host "`nGenerating summary report..." -ForegroundColor Green
    
    $summaryData = @{
        batch_processing_summary = @{
            total_experiments_found = $totalExperiments
            successful_comparisons = $successfulComparisons
            failed_comparisons = $failedComparisons
            output_directory = (Get-Item $OutputDir).FullName
            processed_date = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        }
        processing_details = $processedResults
    }
    
    # Save summary as JSON
    $summaryFile = Join-Path $OutputDir "batch_wd_comparison_summary.json"
    $summaryData | ConvertTo-Json -Depth 10 | Out-File -FilePath $summaryFile -Encoding UTF8
    Write-Host "Summary report saved: $summaryFile" -ForegroundColor Green
    
    # Display quick overview
    Write-Host "`nQuick Overview:" -ForegroundColor Yellow
    $successfulResults = $processedResults | Where-Object { $_.Success }
    
    foreach ($result in $successfulResults) {
        # Try to extract basic info from filename
        if ($result.OutputFile -match "wd_survey_vs_(.+?)_(.+?)_(.+?)_tmp(.+?)_seed(.+?)\.json") {
            $model = $matches[1]
            $variant = $matches[2] 
            $language = $matches[3]
            Write-Host "  $model ($variant, $language): $($result.OutputFile)" -ForegroundColor White
        } else {
            Write-Host "  $($result.ExperimentDirectory): $($result.OutputFile)" -ForegroundColor White
        }
    }
}

if ($failedComparisons -gt 0) {
    Write-Host "`nWarning: $failedComparisons experiments failed. Check the summary report for details." -ForegroundColor Red
    
    # Show failed experiments
    Write-Host "`nFailed Experiments:" -ForegroundColor Red
    $failedResults = $processedResults | Where-Object { -not $_.Success }
    foreach ($failed in $failedResults) {
        Write-Host "  $($failed.ExperimentDirectory): $($failed.ErrorMessage)" -ForegroundColor Red
    }
}

Write-Host "`nAll comparison files saved to: $((Get-Item $OutputDir).FullName)" -ForegroundColor Cyan