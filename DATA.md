scp -P [PORT] BTCUSDT.zip root@[POD IP ADDRESS]:/workspace/

At the bottom of your "Connect Page" you'll see something like this: [194.68.245.213]:[22151] -> 22, which is [POD IP ADDRESS]:[PORT]

apt-get update && apt-get install -y unzip

mkdir -p /workspace/Ultramarin/data

mv /workspace/BTCUSDT.zip ~/Ultramarin/data/

cd ~/Ultramarin/data/

unzip BTCUSDT.zip -x "__MACOSX/*"

unzip data/BTCUSDT.zip -d data/
