# PowerShell execution policy may need to allow scripts
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

& "venv_py310\Scripts\Activate"

$SRC_ROOT = "\\gnict_nas\LG_Chemistry_Site"
$DST_ROOT = "\\gnict_nas\LG_Chemistry_S1K2\defect\_report"

$LINE  = "SSBR"
$GRADE = "FFFFF"

# 날짜 배열
$DATES = @(
    "2025-11-21"
    # "2025-11-22"
)

$BATCH_SIZE = 9

# Python 실행
python report.py `
    --src-root $SRC_ROOT `
    --dst-root $DST_ROOT `
    --line $LINE `
    --grade $GRADE `
    --dates $DATES `
    --batch-size $BATCH_SIZE