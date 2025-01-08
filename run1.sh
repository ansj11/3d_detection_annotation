
for i in $(seq 8491 7 8554)
do 
	echo $i
	python process8.py --input metaloop_20241126205435/metaloop_data/dicts/000${i}.json
done
