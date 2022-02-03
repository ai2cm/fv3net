
stats.csv: query_prog_run_location.sql example.db
	sqlite3  example.db < $<

example.db: build_db.sh
	./build_db.sh

