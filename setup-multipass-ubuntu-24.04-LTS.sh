set -ex

sudo apt update
sudo apt install python3-pip python3.12-venv
sudo apt-get install -y libgl1-mesa-dev

python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
