<#
.SYNOPSIS
Executes the IPSAE Python script on all AlphaFold job directories, grouped by model index, in reverse job order.

.DESCRIPTION
This script finds subdirectories within the specified root directory that follow
the pattern "job*". It processes all directories first for model index 0, then
all for index 1, up to 4. Within each model index group, jobs run in reverse
order. If JSON and CIF exist and PML is absent, the IPSAE script is launched
as a background job. Progress is tracked across all tasks, with concurrency
limited to avoid overload.

.PARAMETER RootDir
The root directory containing the AlphaFold job subdirectories.
Example: "Z:\xenosignaldb\alphafold_results"

.EXAMPLE
.\Run-AllJobs.ps1 -RootDir "Z:\xenosignaldb\alphafold_results"

.NOTES
Requires PowerShell v3.0 or later.
Ensure that the specified Python executable and IPSAE script paths are correct.
#>
param (
    [Parameter(Mandatory=$true)]
    [string]$RootDir
)

if (-not $RootDir) {
    Write-Error "Error: The -RootDir parameter is required."
    Write-Host "Usage: .\Run-AllJobs.ps1 -RootDir '<path_to_alphafold_results>'"
    exit 1
}

# Paths to executables
$pythonExe  = "C:\Users\alyos\miniconda3\envs\bio_ai\python.exe"
$scriptPath = "Z:\phd\Projects\XenoSignalDB\IPSAE\ipsae.py"

# Collect job directories in reverse order
$jobDirectories = Get-ChildItem -Path $RootDir -Directory |
                  Where-Object { $_.Name -like '*job*' } |
                  Sort-Object Name  # -Descending

# Prepare progress tracking: total tasks = jobs * models
$totalJobsCount    = $jobDirectories.Count * 5  # 5 model indices (0..4)
$processedJobCount = 0

# Concurrency settings
$maxParallelJobs = 8
$runningJobs     = [System.Collections.ArrayList]::new()

# Loop by model index first, then by job
foreach ($modelIndex in 0..4) {
    foreach ($jobDirectory in $jobDirectories) {
        $processedJobCount++
        $progressPercentage = [Math]::Round(($processedJobCount / $totalJobsCount) * 100, 1)

        $jobName     = $jobDirectory.Name
        $jobFullPath = $jobDirectory.FullName

        # Build file paths
        $jsonFile = Join-Path $jobFullPath "fold_${jobName}_full_data_${modelIndex}.json"
        $cifFile  = Join-Path $jobFullPath "fold_${jobName}_model_${modelIndex}.cif"
        $pmlFile  = Join-Path $jobFullPath "fold_${jobName}_model_${modelIndex}_10_10.pml"

        # Skip if PML exists and is non-zero
        if ((Test-Path $pmlFile) -and ((Get-Item $pmlFile).Length -gt 0)) {
            Write-Host "[SKIP] ($progressPercentage%) Valid PML exists for ${jobName}, model ${modelIndex}."
            continue
        }

        # Check inputs
        if ((Test-Path $jsonFile) -and (Test-Path $cifFile)) {
            # Throttle background jobs
            while ($runningJobs.Count -ge $maxParallelJobs) {
                foreach ($j in $runningJobs.ToArray()) {
                    if ($j.State -ne 'Running') { $runningJobs.Remove($j) | Out-Null }
                }
                Start-Sleep -Seconds 1
            }

            Write-Host "[RUN]  ($progressPercentage%) ${jobName}, model ${modelIndex}..."

            # Launch IPSAE as background job
            $job = Start-Job -ScriptBlock {
                param($py, $script, $json, $cif)
                & $py $script $json $cif 10 10 2>&1
            } -ArgumentList $pythonExe, $scriptPath, $jsonFile, $cifFile

            [void]$runningJobs.Add($job)
        } else {
            Write-Warning "[MISS] ($progressPercentage%) Missing JSON/CIF for ${jobName}, model ${modelIndex}."
        }
    }
}

# Wait for all background jobs to complete
Write-Host "Waiting for all jobs to finish..."
foreach ($j in $runningJobs) {
    Wait-Job   $j
    Receive-Job $j
    Remove-Job  $j
}
Write-Host "All jobs completed."
