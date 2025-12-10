# Auto Marker

This application automatically grades exams based on a provided answer key.

## Running the application

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv venv
    ```

2.  **Activate the virtual environment:**

    ```bash
    source venv/bin/activate
    ```

3.  **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**

    ```bash
    streamlit run webapp.py
    ```

The application will then be available at the URL displayed in your terminal.

## Command-line interface (CLI)

The application also provides a command-line interface for marking exams.

### `mark`

Marks a single exam file or a directory depending on what was passed as the file argument.

```bash
python3 auto_mark.py mark <file> <answer_file>
```

Running in a multipass vm
```bash
multipass launch 24.04 -n auto-grader-vm
multipass shell auto-grader-vm

sudo apt update
sudo apt install python3-pip python3.12-venv
sudo apt-get install -y libgl1-mesa-dev

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt

python3 auto_mark.py mark file-to-mark.pdf answer-file.pdf
streamlit run webapp.py
```
 or simply

```bash
source setup-multipass-ubuntu-24.04-LTS.sh
```
