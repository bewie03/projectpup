import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def migrate_database():
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    
    # Convert postgres:// to postgresql:// if needed
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    print("Dropping existing tables...")
    cur.execute("""
        DROP TABLE IF EXISTS trackers;
    """)
    
    print("Creating tables with composite primary key (policy_id, channel_id)...")
    cur.execute("""
        CREATE TABLE trackers (
            policy_id VARCHAR(56),
            channel_id BIGINT,
            token_name VARCHAR(255),
            image_url VARCHAR(255),
            threshold FLOAT NOT NULL DEFAULT 1000,
            track_transfers BOOLEAN NOT NULL DEFAULT true,
            last_block BIGINT NOT NULL DEFAULT 0,
            trade_notifications INTEGER NOT NULL DEFAULT 0,
            transfer_notifications INTEGER NOT NULL DEFAULT 0,
            token_info JSONB,
            PRIMARY KEY (policy_id, channel_id)
        );
    """)
    
    print("Migration completed successfully!")
    cur.close()
    conn.close()

def check_database():
    DATABASE_URL = "postgres://uf96h0a7396t3j:p98406daed2890173604432daf725ecedc22819e058d1019c79681d6c84a65501@cd27da2sn4hj7h.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/dbu70voivfu6u8"
    
    # Convert postgres:// to postgresql:// if needed
    if DATABASE_URL.startswith('postgres://'):
        DATABASE_URL = DATABASE_URL.replace('postgres://', 'postgresql://', 1)

    print("Connecting to database...")
    conn = psycopg2.connect(DATABASE_URL)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    
    print("\nChecking table structure...")
    cur.execute("""
        SELECT column_name, data_type, character_maximum_length, 
               column_default, is_nullable
        FROM information_schema.columns 
        WHERE table_name = 'trackers'
        ORDER BY ordinal_position;
    """)
    columns = cur.fetchall()
    print("\nTable structure:")
    print("Column Name | Data Type | Max Length | Default | Nullable")
    print("-" * 70)
    for col in columns:
        col_name, data_type, max_len, default, nullable = col
        print(f"{col_name:<11} | {data_type:<9} | {max_len or 'N/A':<10} | {default or 'N/A':<8} | {nullable:<8}")

    print("\nChecking primary key...")
    cur.execute("""
        SELECT a.attname
        FROM   pg_index i
        JOIN   pg_attribute a ON a.attrelid = i.indrelid
                            AND a.attnum = ANY(i.indkey)
        WHERE  i.indrelid = 'trackers'::regclass
        AND    i.indisprimary;
    """)
    pk = cur.fetchall()
    print("Primary key columns:", [p[0] for p in pk])

    print("\nChecking existing data...")
    cur.execute("SELECT * FROM trackers")
    data = cur.fetchall()
    if data:
        print(f"\nFound {len(data)} trackers:")
        cur.execute("SELECT * FROM trackers LIMIT 0")
        colnames = [desc[0] for desc in cur.description]
        for row in data:
            print("\nTracker:")
            for i, val in enumerate(row):
                print(f"  {colnames[i]}: {val}")
    else:
        print("No trackers found in database")

    print("\nDatabase check completed!")
    cur.close()
    conn.close()

if __name__ == "__main__":
    # migrate_database()  # Comment out migration
    check_database()
