
for i in $(seq 6769 7 8148)
do 
	echo $i
	python -u process9.py --input metaloop_20241126210108/metaloop_data/dicts/000${i}.json
done
