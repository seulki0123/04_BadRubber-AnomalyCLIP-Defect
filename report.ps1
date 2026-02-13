# PowerShell execution policy may need to allow scripts
# Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

# 가상환경 활성화 (Linux / WSL 기준)
source venv_py38/bin/activate

# 경로 설정
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
python src/report.py `
    --src-root $SRC_ROOT `
    --dst-root $DST_ROOT `
    --line $LINE `
    --grade $GRADE `
    --dates $DATES `
    --batch-size $BATCH_SIZE