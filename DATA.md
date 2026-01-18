scp -P 22122 BTCUSDT.zip root@69.30.85.115:/workspace/

apt-get update && apt-get install -y unzip

mkdir -p /workspace/Ultramarin/data

mv /workspace/BTCUSDT.zip ~/Ultramarin/data/

cd ~/Ultramarin/data/

unzip BTCUSDT.zip -x "__MACOSX/*"

unzip data/BTCUSDT.zip -d data/
