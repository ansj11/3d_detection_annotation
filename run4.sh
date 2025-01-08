
for i in $(seq 1484 7 1512)
do 
	echo $i
	python process8.py --input metaloop_20241126205435/metaloop_data/dicts/000${i}.json
done
