# Download the public multilingual spam/scam datasets used by the pipeline.
# Requires the Kaggle CLI and a token in the KAGGLE_API_TOKEN environment variable.
# Do not commit your token.

$ErrorActionPreference = "Stop"
if (-not $env:KAGGLE_API_TOKEN) {
    Write-Error "Set KAGGLE_API_TOKEN before running this script."
    exit 1
}

$base = Join-Path $PSScriptRoot "..\data\raw"
$datasets = @(
    @{ ref = "siddharthrane07/scamshield-multilingual-smishing-social-engn"; dir = "scamshield" },
    @{ ref = "mdferozahmedafm/bengali-sms-smishing-dataset"; dir = "bengali" },
    @{ ref = "vinit119/sms-scam-detection-dataset-merged"; dir = "merged" },
    @{ ref = "roberto2057/hindi-sms"; dir = "hindi" },
    @{ ref = "alialzubaidi7/arabic-sms-spam-dataset"; dir = "arabic" },
    @{ ref = "aldoivan/sms-spam-mexico-dataset-en-espaol-mexicano"; dir = "spanish" },
    @{ ref = "kinoux/french-spamham-detection-free"; dir = "french" },
    @{ ref = "kritikgianta/multilanguage-sms-classification-dataset"; dir = "multilang" },
    @{ ref = "crystalbaby/a2pp2p-sms-classifier-is-it-a-bot-or-a-bestie"; dir = "p2p" },
    @{ ref = "ahmadtijjani/phishing-urgency-authority-persuasion"; dir = "phishing\urgency" },
    @{ ref = "uciml/sms-spam-collection-dataset"; dir = "uci" }
)

foreach ($dataset in $datasets) {
    $target = Join-Path $base $dataset.dir
    New-Item -ItemType Directory -Force -Path $target | Out-Null
    Write-Host "Downloading $($dataset.ref) -> $target"
    kaggle datasets download -d $dataset.ref -p $target --unzip
}

Write-Host "Done. Raw data is under $base."
Write-Host "Note: review each upstream license before commercial use."
