Here is the raw Markdown code.

```markdown
# Auto Marker

Automated exam grading application using a provided answer key.

## Installation

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

```

## Workflow

### 1. Extract Student IDs

Use `detect_student_id.py` to process the exam bundle. Use `--known-student-ids` to improve accuracy.

**Option A: List Mode (Extract IDs to text file)**

```bash
python3 detect_student_id.py list C481Final.pdf \
    --pages-per-attempt 1 \
    --known-student-ids known.ids \
    -o C481.student.ids

```

**Option B: Split Mode (Split PDF by Student ID)**

```bash
python3 detect_student_id.py split C481Final.pdf \
    --pages-per-attempt 1 \
    --known-student-ids known.ids \
    -o C481Final.split

```

*Output: A directory containing `{student_id}.pdf` files.*

### 2. Mark Exams

Use `auto_mark.py` to grade the attempts.

**Method A: Directory (Recommended)**
Grades a directory of split files. Supports multi-threading (`-j`).

```bash
python3 auto_mark.py mark C481Final.split/ key.pdf -o final_grades.csv

```

**Method B: Single File**
Grades the original bundle using the ID list from Step 1.

```bash
python3 auto_mark.py mark --single-file C481Final.pdf key.pdf \
    --student-ids C481.student.ids \
    -o final_grades.csv

```

---

## Multipass VM Setup

**Quick Start**

```bash
source setup-multipass-ubuntu-24.04-LTS.sh

```

**Manual Setup**

```bash
multipass launch 24.04 -n auto-grader-vm
multipass shell auto-grader-vm

# System Deps
sudo apt update && sudo apt install -y python3-pip python3.12-venv libgl1-mesa-dev

# Python Setup
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

# Run
python3 auto_mark.py mark attempt.pdf key.pdf
streamlit run webapp.py
```
