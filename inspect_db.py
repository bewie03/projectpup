import psycopg2

def inspect_database():
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    conn = psycopg2.connect(DATABASE_URL)
    cur = conn.cursor()
    
    # List all tables
    cur.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'public'
    """)
    tables = cur.fetchall()
    print("\nTables in database:")
    for table in tables:
        print(f"\n=== {table[0]} ===")
        
        # Get column info
        cur.execute(f"""
            SELECT column_name, data_type 
            FROM information_schema.columns 
            WHERE table_name = '{table[0]}'
        """)
        columns = cur.fetchall()
        print("\nColumns:")
        for col in columns:
            print(f"{col[0]}: {col[1]}")
            
        # Get row count
        cur.execute(f"SELECT COUNT(*) FROM {table[0]}")
        count = cur.fetchone()[0]
        print(f"\nTotal rows: {count}")
        
        # Show sample data
        if count > 0:
            cur.execute(f"SELECT * FROM {table[0]} LIMIT 3")
            rows = cur.fetchall()
            print("\nSample data:")
            for row in rows:
                print(row)
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    inspect_database()
