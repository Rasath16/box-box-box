# Box Box Box - Test Runner (PowerShell)
# Usage: .\test_runner.ps1

$ErrorActionPreference = "Continue"

# Configuration
$TEST_CASES_DIR = "data/test_cases/inputs"
$EXPECTED_OUTPUTS_DIR = "data/test_cases/expected_outputs"
$RUN_COMMAND_FILE = "solution/run_command.txt"

# Check if run command file exists
if (-not (Test-Path $RUN_COMMAND_FILE)) {
    Write-Host "Error: Run command file not found: $RUN_COMMAND_FILE" -ForegroundColor Red
    Write-Host "Please create $RUN_COMMAND_FILE with your run command"
    exit 1
}

$SOLUTION_CMD = (Get-Content $RUN_COMMAND_FILE).Trim()

# Check if test cases directory exists
if (-not (Test-Path $TEST_CASES_DIR)) {
    Write-Host "Error: Test cases directory not found: $TEST_CASES_DIR" -ForegroundColor Red
    exit 1
}

# Get all test files
$TEST_FILES = Get-ChildItem "$TEST_CASES_DIR/test_*.json" | Sort-Object Name
$TOTAL_TESTS = $TEST_FILES.Count

if ($TOTAL_TESTS -eq 0) {
    Write-Host "Error: No test files found in $TEST_CASES_DIR" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "═════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "          Box Box Box - Test Runner" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Solution Command: " -NoNewline
Write-Host "$SOLUTION_CMD" -ForegroundColor Yellow
Write-Host "Test Cases Found: " -NoNewline
Write-Host "$TOTAL_TESTS" -ForegroundColor Yellow
Write-Host ""

# Initialize counters
$PASSED = 0
$FAILED = 0
$ERRORS = 0

# Create temp directory
$TMP_DIR = New-Item -ItemType Directory -Path ([System.IO.Path]::Combine([System.IO.Path]::GetTempPath(), "box_box_box_$(Get-Random)")) -Force | Select-Object -ExpandProperty FullName

# Check if expected outputs exist
$HAS_ANSWERS = Test-Path $EXPECTED_OUTPUTS_DIR

Write-Host "Running tests..." -ForegroundColor Cyan
Write-Host ""

# Run tests
foreach ($TEST_FILE in $TEST_FILES) {
    $TEST_NAME = $TEST_FILE.BaseName
    $TEST_ID = "TEST_" + $TEST_NAME.Replace("test_", "")
    
    $OUTPUT_FILE = Join-Path $TMP_DIR "${TEST_NAME}_output.json"
    $ERROR_FILE = Join-Path $TMP_DIR "${TEST_NAME}_error.log"
    
    try {
        # Read test input and run solution
        $testInput = Get-Content $TEST_FILE.FullName -Raw
        $output = $testInput | Invoke-Expression $SOLUTION_CMD -ErrorVariable cmdError -ErrorAction SilentlyContinue
        $exitCode = $LASTEXITCODE
        
        # Save output
        $output | Out-File $OUTPUT_FILE -Encoding UTF8
        if ($cmdError) {
            $cmdError | Out-File $ERROR_FILE -Encoding UTF8
        }
        
        if ($exitCode -eq 0 -and $output) {
            # Try to parse as JSON
            try {
                $jsonOutput = $output | ConvertFrom-Json
                $PREDICTED = $jsonOutput.finishing_positions -join ","
                
                if ([string]::IsNullOrWhiteSpace($PREDICTED) -or $PREDICTED -eq "null") {
                    Write-Host "✗ $TEST_ID - Invalid output format" -ForegroundColor Red
                    $FAILED++
                } elseif ($HAS_ANSWERS) {
                    # Compare with expected output
                    $ANSWER_FILE = Join-Path $EXPECTED_OUTPUTS_DIR "$TEST_NAME.json"
                    if (Test-Path $ANSWER_FILE) {
                        $expectedJson = Get-Content $ANSWER_FILE -Raw | ConvertFrom-Json
                        $EXPECTED = $expectedJson.finishing_positions -join ","
                        
                        if ($PREDICTED -eq $EXPECTED) {
                            Write-Host "✓ $TEST_ID" -ForegroundColor Green
                            $PASSED++
                        } else {
                            Write-Host "✗ $TEST_ID - Incorrect prediction" -ForegroundColor Red
                            $FAILED++
                        }
                    } else {
                        Write-Host "? $TEST_ID - Output generated (no answer file)" -ForegroundColor Yellow
                        $PASSED++
                    }
                } else {
                    Write-Host "? $TEST_ID - Output generated (no answer key)" -ForegroundColor Yellow
                    $PASSED++
                }
            } catch {
                Write-Host "✗ $TEST_ID - Invalid JSON output" -ForegroundColor Red
                $FAILED++
            }
        } else {
            Write-Host "✗ $TEST_ID - Execution error" -ForegroundColor Red
            if (Test-Path $ERROR_FILE) {
                $errorMsg = (Get-Content $ERROR_FILE | Select-Object -First 1)
                Write-Host "  Error: $errorMsg" -ForegroundColor Red
            }
            $ERRORS++
        }
    } catch {
        Write-Host "✗ $TEST_ID - Error running test" -ForegroundColor Red
        $ERRORS++
    }
}

# Calculate stats
[double]$PASS_RATE = 0
if ($TOTAL_TESTS -gt 0) {
    $PASS_RATE = [Math]::Round(($PASSED * 100 / $TOTAL_TESTS), 1)
}

Write-Host ""
Write-Host "═════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "                    Results" -ForegroundColor Cyan
Write-Host "═════════════════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""
Write-Host "Total Tests:    " -NoNewline
Write-Host "$TOTAL_TESTS" -ForegroundColor Yellow
Write-Host "Passed:         " -NoNewline
Write-Host "$PASSED" -ForegroundColor Green
Write-Host "Failed:         " -NoNewline
Write-Host "$FAILED" -ForegroundColor Red
if ($ERRORS -gt 0) {
    Write-Host "Errors:         " -NoNewline
    Write-Host "$ERRORS" -ForegroundColor Red
}
Write-Host ""
Write-Host "Pass Rate:      " -NoNewline
Write-Host "$PASS_RATE%" -ForegroundColor Green
Write-Host ""

if (-not $HAS_ANSWERS) {
    Write-Host "Note: Running without expected outputs. Only checking output format." -ForegroundColor Yellow
    Write-Host ""
}

# Final message
if ($PASSED -eq $TOTAL_TESTS) {
    Write-Host "Perfect score! All tests passed!" -ForegroundColor Green
} elseif ($PASSED -gt 0) {
    Write-Host "Keep improving! Check failed test cases." -ForegroundColor Yellow
} else {
    Write-Host "No tests passed. Review your implementation." -ForegroundColor Red
}

# Cleanup
Remove-Item -Path $TMP_DIR -Recurse -Force -ErrorAction SilentlyContinue

exit 0

