
stats.csv: query.sql
	sqlite3  example.db < $<

example.db: build_db.sh
	./build_db.sh

