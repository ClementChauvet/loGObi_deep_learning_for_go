cd "${0%/*}"
wget https://www.lamsade.dauphine.fr/~cazenave/project2022.zip
mkdir ../project_files
unzip project2022.zip -d ../project_files
wget https://www.lamsade.dauphine.fr/~cazenave/games.1000000.data.zip
unzip games.1000000.data.zip
mv games.1000000.data ../project_files/games.data
rm games.1000000.data.zip
rm project2022.zip
pip install pybind11
cd ../project_files
bash ./compile.sh