scp -P [PORT] BTCUSDT.zip root@[POD IP ADDRESS]:/workspace/

apt-get update && apt-get install -y unzip

mkdir -p /workspace/Ultramarin/data

mv /workspace/BTCUSDT.zip ~/Ultramarin/data/

cd ~/Ultramarin/data/

unzip BTCUSDT.zip -x "__MACOSX/*"

unzip data/BTCUSDT.zip -d data/
