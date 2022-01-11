#!/bin/bash

rm -f example.db
sqlite3 example.db < 0_create_tables.sql
python3 2_wandb_to_sqlite.py
sqlite3 example.db < 3_views.sql

